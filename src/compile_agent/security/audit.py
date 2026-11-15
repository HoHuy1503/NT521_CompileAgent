import base64
import json
import logging
import os
import posixpath
import re
import subprocess
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from compile_agent.config import MAKE_JOBS
from compile_agent.reporting.logs import find_compiled_artifacts, smoke_check_artifact

CUSTOM_SEMGREP_RULES_CONTAINER = "/opt/rules/c_security.yaml"
HARDENING_WRAPPER_FLAGS = (
    "-O2 -g -fstack-protector-strong -D_FORTIFY_SOURCE=2 -fPIC -pie -Wl,-z,relro,-z,now"
)
SANITIZER_WRAPPER_FLAGS = (
    "-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer"
)

_EXPLORATORY_CMD_PREFIXES = (
    "tree",
    "ls",
    "cat",
    "head",
    "tail",
    "less",
    "more",
    "find",
    "file",
    "ldd",
    "readelf",
    "objdump",
    "nm",
    "pwd",
    "which",
    "whereis",
    "echo",
    "grep",
    "rg",
    "wc",
    "apt",
    "apt-get",
    "dpkg",
    "uname",
    "nproc",
    "env",
    "printenv",
    "mkdir",
    "touch",
    "chmod",
    "cp",
    "mv",
    "ln",
    "git ",
    "tar ",
    "unzip ",
    "rm ",
)

_EXIT_CODE_RE = re.compile(r"\[Exit code:\s*(-?\d+)\]\s*$")


def _extract_exit_code(output: str) -> Optional[int]:
    if not output:
        return None
    m = _EXIT_CODE_RE.search(output.strip())
    return int(m.group(1)) if m else None


def _strip_exit_marker(output: str) -> str:
    if not output:
        return ""
    return _EXIT_CODE_RE.sub("", output).strip()


def _exec_in_container(
    shell,
    command: str,
    workdir: str = "/work",
    timeout: int = 900,
) -> Dict[str, Any]:
    """
    Execute a non-interactive command inside the running container via raw `docker exec`
    to bypass SSH PTY line-wrapping and output truncation while keeping the exact
    container state (installed apt packages, /work mount, etc.).
    """
    container_id = getattr(shell, "container_id", None)
    if not container_id:
        wrapped = (
            f"(cd {json.dumps(workdir)} && {command})"
            if workdir and workdir != "/work"
            else command
        )
        out = shell.execute_command(wrapped, record_log=False)
        return {
            "returncode": _extract_exit_code(out) if _extract_exit_code(out) is not None else -1,
            "stdout": _strip_exit_marker(out),
            "stderr": "",
        }

    cmd = ["docker", "exec", "-w", workdir or "/work", container_id, "/bin/bash", "-c", command]
    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
        }
    except subprocess.TimeoutExpired as exc:
        out_s = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        err_s = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        return {
            "returncode": 124,
            "stdout": out_s,
            "stderr": f"{err_s}\nCommand timed out after {timeout}s".strip(),
        }


def _run_container_python(
    shell, script: str, marker: str, timeout: int = 120
) -> Optional[Any]:
    container_id = getattr(shell, "container_id", None)
    if container_id:
        try:
            proc = subprocess.run(
                ["docker", "exec", "-i", "-w", "/work", container_id, "python3", "-"],
                input=script,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
            out = proc.stdout or ""
            if proc.returncode != 0 and not out and proc.stderr:
                logging.warning(
                    "[!] Container python (%s) exited %s: %s",
                    marker,
                    proc.returncode,
                    proc.stderr.strip()[:500],
                )
        except Exception as exc:
            logging.warning("[!] Container python (%s) failed: %s", marker, exc)
            return None
    else:
        payload_b64 = base64.b64encode(script.encode("utf-8")).decode("ascii")
        out = shell.execute_command(
            f"python3 -c \"import base64; exec(base64.b64decode('{payload_b64}'))\"",
            record_log=False,
        )

    for line in out.splitlines():
        s = line.strip()
        if s.startswith(marker):
            try:
                return json.loads(s[len(marker) :])
            except json.JSONDecodeError as err:
                logging.warning("[!] Failed to parse JSON after marker %s: %s", marker, err)
                return None
    return None


def _is_exploratory_command(cmd: str) -> bool:
    s = (cmd or "").strip()
    if not s:
        return True
    low = s.lower()
    if any(low.startswith(p) for p in _EXPLORATORY_CMD_PREFIXES):
        return True
    if any(q in low for q in ("--version", "--help", " -v", "make clean", "distclean")):
        if not any(b in low for b in ("gcc ", "g++ ", "cc ", "c++ ", "cmake ", "./configure", "make")):
            return True
    return False


def _probe_makefile_targets(shell, workdir: str, makefile_flag: str = "") -> Dict[str, bool]:
    res: Dict[str, bool] = {"clean": False, "test": False, "check": False}
    mf = f" {makefile_flag}" if makefile_flag else ""
    for target in ("clean", "test", "check"):
        r = _exec_in_container(
            shell,
            f"make{mf} -n {target} >/dev/null 2>&1",
            workdir=workdir,
            timeout=30,
        )
        res[target] = r["returncode"] == 0
    return res


def detect_build_recipe(shell, tools_log: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inspect both container /work filesystem and the agent's Shell command history
    to construct a clean/build/test recipe for Layer 1 Bear and Layer 3 rebuilds.
    """
    shell_history = list((tools_log or {}).get("Shell") or [])
    shell_entries = [
        (
            item[0].strip(),
            item[1] if len(item) > 1 and isinstance(item[1], str) else "",
        )
        for item in shell_history
        if isinstance(item, (list, tuple)) and item and isinstance(item[0], str)
    ]

    fs_probe_script = r"""
import os, json

cmake_dirs = []
ninja_dirs = []
makefiles = []
test_scripts = []

for r, ds, fs in os.walk('/work'):
    ds[:] = sorted([d for d in ds if d not in ('.git', 'CMakeFiles', '.deps')])
    rel_r = os.path.relpath(r, '/work').replace('\\', '/')
    depth = 0 if rel_r == '.' else rel_r.count('/') + 1
    if 'CMakeCache.txt' in fs:
        cmake_dirs.append((depth, r, 'CTestTestfile.cmake' in fs))
    if 'build.ninja' in fs:
        ninja_dirs.append((depth, r))
    for mf in ('GNUmakefile', 'makefile', 'Makefile'):
        if mf in fs:
            makefiles.append((depth, r, mf))
    for fn in fs:
        low = fn.lower()
        if fn.endswith('.gcc') and 'make' in low:
            makefiles.append((depth + 10, r, fn))
        if fn in ('Makefile', 'makefile', 'GNUmakefile'):
            try:
                with open(os.path.join(r, fn), 'r', errors='ignore') as f:
                    txt = f.read(16000)
                if '\ntest:' in txt or txt.startswith('test:'):
                    test_scripts.append((r, fn, 'test'))
                elif '\ncheck:' in txt or txt.startswith('check:'):
                    test_scripts.append((r, fn, 'check'))
            except OSError:
                pass

cmake_dirs.sort()
ninja_dirs.sort()
makefiles.sort()
print('__AC_FS__' + json.dumps({
    'cmake_dirs': cmake_dirs,
    'ninja_dirs': ninja_dirs,
    'makefiles': makefiles,
    'test_scripts': test_scripts,
}))
"""
    fs_info = _run_container_python(shell, fs_probe_script, "__AC_FS__") or {}
    cmake_dirs = fs_info.get("cmake_dirs", [])
    ninja_dirs = fs_info.get("ninja_dirs", [])
    makefiles = fs_info.get("makefiles", [])

    # 1. CMake recipe
    if cmake_dirs:
        _, bdir, has_ctest = cmake_dirs[0]
        test_cmd = f"ctest --test-dir {json.dumps(bdir)} --output-on-failure" if has_ctest else ""
        if not test_cmd:
            targets = _probe_makefile_targets(shell, bdir)
            if targets["test"]:
                test_cmd = f"cmake --build {json.dumps(bdir)} --target test"
        clean_cmd = (
            f"cmake --build {json.dumps(bdir)} --target clean"
            if not ninja_dirs or ninja_dirs[0][1] != bdir
            else f"ninja -C {json.dumps(bdir)} -t clean"
        )
        return {
            "build_system": "cmake",
            "workdir": bdir,
            "clean_cmd": clean_cmd,
            "build_cmd": f"cmake --build {json.dumps(bdir)} -j{MAKE_JOBS}",
            "compile_db_cmd": "",
            "test_cmd": test_cmd,
            "cmake_dir": bdir,
        }

    # 2. Ninja / Meson recipe
    if ninja_dirs:
        _, bdir = ninja_dirs[0]
        return {
            "build_system": "ninja",
            "workdir": bdir,
            "clean_cmd": f"ninja -C {json.dumps(bdir)} -t clean",
            "build_cmd": f"ninja -C {json.dumps(bdir)} -j{MAKE_JOBS}",
            "compile_db_cmd": f"ninja -C {json.dumps(bdir)} -t compdb > /tmp/compile_commands.json",
            "test_cmd": "",
        }

    # 3. Extract explicit make or compiler invocation from agent shell history
    exit0_make_workdir = "/work"
    exit0_make_cmd = ""
    fallback_make_workdir = "/work"
    fallback_make_cmd = ""
    exit0_direct_cmds: List[str] = []
    fallback_direct_cmds: List[str] = []
    cur_dir = "/work"
    for cmd, obs in shell_entries:
        ec = _extract_exit_code(obs)
        # Approximate interactive cwd transitions cleanly
        cd_only = re.match(r"^\s*cd\s+([^\s;&|]+)\s*$", cmd)
        if cd_only:
            if ec in (None, 0):
                tgt = cd_only.group(1).strip("'\"")
                if tgt.startswith("/"):
                    cur_dir = posixpath.normpath(tgt)
                elif tgt == "..":
                    cur_dir = posixpath.dirname(cur_dir) or "/"
                else:
                    cur_dir = posixpath.normpath(posixpath.join(cur_dir, tgt))
            continue

        # Split chained command segments for cd + make patterns
        cd_chain = re.search(r"(?:^|[;&|]\s*)cd\s+([^\s;&|]+)\s*(?:&&|;)", cmd)
        seg_dir = cur_dir
        if cd_chain:
            tgt = cd_chain.group(1).strip("'\"")
            seg_dir = (
                posixpath.normpath(tgt)
                if tgt.startswith("/")
                else posixpath.normpath(posixpath.join(cur_dir, tgt))
            )
            if not cmd.lstrip().startswith("(") and ec in (None, 0):
                cur_dir = seg_dir

        if ec not in (None, 0):
            continue

        # Look for make invocation in command (preserving leading VAR=val env assignments)
        make_match = re.search(
            r"(?:[A-Za-z_][A-Za-z0-9_]*=(?:\"[^\"]*\"|'[^']*'|[^\s;&|]+)\s+)*\bmake\b[^;&|\n]*",
            cmd,
        )
        if make_match:
            m_seg = make_match.group(0).strip()
            if not any(x in m_seg for x in (" clean", " distclean", " mrproper", " --version", " -n ")):
                # Resolve -C dir if present
                c_flag = re.search(r"-C\s+([^\s]+)", m_seg)
                if c_flag:
                    c_tgt = c_flag.group(1).strip("'\"")
                    resolved_dir = (
                        posixpath.normpath(c_tgt)
                        if c_tgt.startswith("/")
                        else posixpath.normpath(posixpath.join(seg_dir, c_tgt))
                    )
                    m_seg = re.sub(r"\s*-C\s+[^\s]+", "", m_seg).strip()
                else:
                    resolved_dir = seg_dir
                # Strip test/check target from build command so build and test stay separate
                m_seg = re.sub(r"\s+(?:test|check)(?=\s|$)", "", m_seg).strip()
                if re.search(r"(?:^|\s)make(?:\s|$)", m_seg):
                    if ec == 0:
                        exit0_make_cmd = m_seg
                        exit0_make_workdir = resolved_dir
                    else:
                        fallback_make_cmd = m_seg
                        fallback_make_workdir = resolved_dir
                continue

        if not _is_exploratory_command(cmd):
            if re.search(r"\b(?:gcc|g\+\+|cc|c\+\+|clang|clang\+\+|ar)\b", cmd):
                entry_cmd = f"(cd {json.dumps(cur_dir)} && {cmd})"
                if ec == 0:
                    exit0_direct_cmds.append(entry_cmd)
                else:
                    fallback_direct_cmds.append(entry_cmd)

    last_make_cmd = exit0_make_cmd or fallback_make_cmd
    last_make_workdir = exit0_make_workdir if exit0_make_cmd else fallback_make_workdir
    direct_compile_cmds = exit0_direct_cmds or fallback_direct_cmds

    # Verify whether detected make workdir actually exists & extract -f makefile
    if last_make_cmd:
        f_match = re.search(r"-f\s+([^\s]+)", last_make_cmd)
        mf_flag = f"-f {f_match.group(1)}" if f_match else ""
        # Ensure -j is normalized while keeping leading VAR=val assignments intact
        if "-j" not in last_make_cmd:
            last_make_cmd = re.sub(r"\bmake\b", f"make -j{MAKE_JOBS}", last_make_cmd, count=1)

        exists_res = _exec_in_container(
            shell,
            f"test -d {json.dumps(last_make_workdir)}",
            workdir="/work",
            timeout=15,
        )
        if exists_res["returncode"] == 0:
            targets = _probe_makefile_targets(shell, last_make_workdir, mf_flag)
            if targets["clean"]:
                clean_cmd = f"make{(' ' + mf_flag) if mf_flag else ''} clean"
            else:
                clean_cmd = (
                    "find . -maxdepth 4 \\( -name '*.o' -o -name '*.lo' -o -name '*.obj' "
                    "-o -path '*/_o/*.o' \\) -delete"
                )
            test_cmd = ""
            if targets["test"]:
                test_cmd = f"make{(' ' + mf_flag) if mf_flag else ''} test"
            elif targets["check"]:
                test_cmd = f"make{(' ' + mf_flag) if mf_flag else ''} check"
            return {
                "build_system": "make",
                "workdir": last_make_workdir,
                "clean_cmd": clean_cmd,
                "build_cmd": last_make_cmd,
                "compile_db_cmd": "",
                "test_cmd": test_cmd,
            }

    # Fallback to shallowest Makefile discovered in /work
    if makefiles:
        _, mdir, mfile = makefiles[0]
        mf_flag = "" if mfile in ("Makefile", "makefile", "GNUmakefile") else f"-f {mfile}"
        targets = _probe_makefile_targets(shell, mdir, mf_flag)
        clean_cmd = (
            f"make{(' ' + mf_flag) if mf_flag else ''} clean"
            if targets["clean"]
            else "find . -maxdepth 4 \\( -name '*.o' -o -name '*.lo' -o -name '*.obj' -o -path '*/_o/*.o' \\) -delete"
        )
        test_cmd = ""
        if targets["test"]:
            test_cmd = f"make{(' ' + mf_flag) if mf_flag else ''} test"
        elif targets["check"]:
            test_cmd = f"make{(' ' + mf_flag) if mf_flag else ''} check"
        return {
            "build_system": "make",
            "workdir": mdir,
            "clean_cmd": clean_cmd,
            "build_cmd": f"make -j{MAKE_JOBS}{(' ' + mf_flag) if mf_flag else ''}",
            "compile_db_cmd": "",
            "test_cmd": test_cmd,
        }

    # 4. Direct compiler commands
    if direct_compile_cmds:
        return {
            "build_system": "direct",
            "workdir": "/work",
            "clean_cmd": "find /work -maxdepth 4 -name '*.o' -delete",
            "build_cmd": " && ".join(direct_compile_cmds),
            "compile_db_cmd": "",
            "test_cmd": "",
        }

    return {
        "build_system": "unknown",
        "workdir": "/work",
        "clean_cmd": "",
        "build_cmd": "",
        "compile_db_cmd": "",
        "test_cmd": "",
    }


def _run_semgrep_in_container(
    shell, log_path: str, proj_name: str
) -> Dict[str, Any]:
    os.makedirs(os.path.join(log_path, "semgrep"), exist_ok=True)
    host_out = os.path.join(log_path, "semgrep", f"{proj_name}.semgrep.json")

    configs = [CUSTOM_SEMGREP_RULES_CONTAINER, "p/c"]
    base_excludes = (
        "--exclude .git --exclude CMakeFiles --exclude build --exclude _o "
        "--exclude '*.min.js'"
    )

    def _exec_semgrep(cfg_list: List[str], disable_metrics: bool = False) -> Optional[int]:
        cfg_args = " ".join(f"--config {json.dumps(c)}" for c in cfg_list)
        metrics_flag = "--metrics=off " if disable_metrics else ""
        cmd = (
            f"rm -f /tmp/ac_semgrep.json && "
            f"semgrep scan {cfg_args} {base_excludes} --timeout 15 --max-target-bytes 2000000 {metrics_flag}"
            f"--json --output /tmp/ac_semgrep.json --quiet /work >/tmp/ac_semgrep.log 2>&1"
        )
        res = _exec_in_container(shell, cmd, workdir="/work", timeout=300)
        return res["returncode"]

    ec = _exec_semgrep(configs)
    used_configs = list(configs)
    status = "completed"

    # Semgrep exits 0 (no findings) or 1 (findings with --error, though without --error it returns 0);
    # nonzero usually means registry timeout / network failure for p/c
    exists_check = _exec_in_container(shell, "test -s /tmp/ac_semgrep.json", timeout=15)
    if ec not in (0, 1) or exists_check["returncode"] != 0:
        logging.warning(
            "[!] Semgrep with registry rules (p/c) returned ec=%s; falling back to local custom rules.",
            ec,
        )
        used_configs = [CUSTOM_SEMGREP_RULES_CONTAINER]
        ec = _exec_semgrep(used_configs, disable_metrics=True)
        status = "fallback_local_rules"

    parse_script = r"""
import json, os

p = '/tmp/ac_semgrep.json'
if not os.path.exists(p) or os.path.getsize(p) == 0:
    print('__AC_SG__' + json.dumps({'ok': False, 'error': 'missing semgrep output'}))
else:
    with open(p, 'r', encoding='utf-8', errors='replace') as f:
        data = json.load(f)
    results = data.get('results', [])
    errors = data.get('errors', [])
    by_sev = {}
    by_rule = {}
    items = []
    for r in results:
        extra = r.get('extra', {})
        sev = str(extra.get('severity', 'UNKNOWN')).upper()
        cid = str(r.get('check_id', ''))
        short_id = cid.split('.')[-1] if '.' in cid else cid
        by_sev[sev] = by_sev.get(sev, 0) + 1
        by_rule[short_id] = by_rule.get(short_id, 0) + 1
        path = str(r.get('path', ''))
        if path.startswith('/work/'):
            path = path[len('/work/'):]
        meta = extra.get('metadata', {}) or {}
        if len(items) < 200:
            items.append({
                'check_id': cid,
                'rule': short_id,
                'severity': sev,
                'path': path,
                'line': r.get('start', {}).get('line', 0),
                'col': r.get('start', {}).get('col', 0),
                'message': (extra.get('message', '') or '').strip(),
                'cwe': meta.get('cwe', ''),
            })
    print('__AC_SG__' + json.dumps({
        'ok': True,
        'total_findings': len(results),
        'by_severity': by_sev,
        'by_rule': by_rule,
        'errors_count': len(errors),
        'findings': items,
    }))
"""
    parsed = _run_container_python(shell, parse_script, "__AC_SG__")
    _exec_in_container(shell, "rm -f /tmp/ac_semgrep.json /tmp/ac_semgrep.log", timeout=15)

    if not parsed or not parsed.get("ok"):
        err_report = {
            "status": "error",
            "configs_used": used_configs,
            "total_findings": 0,
            "by_severity": {},
            "by_rule": {},
            "findings": [],
        }
        with open(host_out, "w", encoding="utf-8") as fp:
            json.dump(err_report, fp, indent=2, ensure_ascii=False)
        return err_report

    summary = {
        "status": status,
        "configs_used": used_configs,
        "report_path": host_out,
        "total_findings": parsed["total_findings"],
        "by_severity": parsed["by_severity"],
        "by_rule": parsed["by_rule"],
        "errors_count": parsed["errors_count"],
        "findings": parsed["findings"],
    }
    with open(host_out, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    return summary


def _run_cppcheck_in_container(
    shell,
    log_path: str,
    proj_name: str,
    recipe: Dict[str, Any],
    has_baseline_binaries: bool = True,
) -> Dict[str, Any]:
    os.makedirs(os.path.join(log_path, "cppcheck"), exist_ok=True)
    host_out = os.path.join(log_path, "cppcheck", f"{proj_name}.cppcheck.json")

    _exec_in_container(
        shell,
        "rm -f /tmp/ac_compile_commands.json /tmp/compile_commands.json /tmp/ac_cppcheck.xml",
        timeout=15,
    )

    compile_db_path = ""
    compile_db_entries = 0
    workdir = recipe.get("workdir") or "/work"
    build_sys = recipe.get("build_system", "unknown")

    if has_baseline_binaries:
        # Locate or generate compile_commands.json
        if build_sys == "cmake":
            cmake_dir = recipe.get("cmake_dir") or workdir
            _exec_in_container(
                shell,
                f"cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON {json.dumps(cmake_dir)} >/dev/null 2>&1 || true",
                workdir=workdir,
                timeout=180,
            )
            candidate = f"{cmake_dir.rstrip('/')}/compile_commands.json"
            cnt_info = _run_container_python(
                shell,
                (
                    "import json, os\n"
                    f"p = {json.dumps(candidate)}\n"
                    "n = 0\n"
                    "if os.path.exists(p):\n"
                    "    try:\n"
                    "        d = json.load(open(p, 'r', encoding='utf-8', errors='replace'))\n"
                    "        if isinstance(d, list): n = len(d)\n"
                    "    except Exception: pass\n"
                    "print('__AC_DB__' + json.dumps({'path': p, 'count': n}))\n"
                ),
                "__AC_DB__",
            )
            if cnt_info and cnt_info.get("count", 0) > 0:
                compile_db_path = cnt_info["path"]
                compile_db_entries = cnt_info["count"]
        elif build_sys == "ninja" and recipe.get("compile_db_cmd"):
            _exec_in_container(shell, recipe["compile_db_cmd"], workdir=workdir, timeout=120)
            cnt_info = _run_container_python(
                shell,
                (
                    "import json, os\n"
                    "p = '/tmp/compile_commands.json'\n"
                    "n = 0\n"
                    "if os.path.exists(p):\n"
                    "    try:\n"
                    "        d = json.load(open(p, 'r', encoding='utf-8', errors='replace'))\n"
                    "        if isinstance(d, list):\n"
                    "            d = [e for e in d if 'CMakeFiles' not in str(e.get('file', ''))]\n"
                    "            n = len(d)\n"
                    "            json.dump(d, open(p, 'w', encoding='utf-8'))\n"
                    "    except Exception: pass\n"
                    "print('__AC_DB__' + json.dumps({'path': p, 'count': n}))\n"
                ),
                "__AC_DB__",
            )
            if cnt_info and cnt_info.get("count", 0) > 0:
                compile_db_path = cnt_info["path"]
                compile_db_entries = cnt_info["count"]

        if compile_db_entries == 0 and recipe.get("build_cmd"):
            clean_cmd = recipe.get("clean_cmd", "")
            build_cmd = recipe.get("build_cmd", "")
            # Run bear inside workdir with wrapper disabled to capture accurate compile DB
            seq = ""
            if clean_cmd:
                seq += f"({clean_cmd} >/dev/null 2>&1 || true) && "
            seq += (
                f"ENABLE_COMPILER_WRAPPER=0 WRAPPER_OPTI='' "
                f"bear --output /tmp/ac_compile_commands.json -- {build_cmd} >/tmp/ac_bear.log 2>&1"
            )
            _exec_in_container(shell, seq, workdir=workdir, timeout=900)
            cnt_info = _run_container_python(
                shell,
                (
                    "import json, os\n"
                    "p = '/tmp/ac_compile_commands.json'\n"
                    "n = 0\n"
                    "if os.path.exists(p):\n"
                    "    try:\n"
                    "        d = json.load(open(p, 'r', encoding='utf-8', errors='replace'))\n"
                    "        if isinstance(d, list):\n"
                    "            d = [e for e in d if 'CMakeFiles' not in str(e.get('file', ''))]\n"
                    "            n = len(d)\n"
                    "            json.dump(d, open(p, 'w', encoding='utf-8'))\n"
                    "    except Exception: pass\n"
                    "print('__AC_DB__' + json.dumps({'path': p, 'count': n}))\n"
                ),
                "__AC_DB__",
            )
            if cnt_info and cnt_info.get("count", 0) > 0:
                compile_db_path = cnt_info["path"]
                compile_db_entries = cnt_info["count"]

    cpp_platform_flags = (
        "--platform=unix64 -D__x86_64__=1 -D__linux__=1 -D__GNUC__=13 "
        "-D__BYTE_ORDER__=__ORDER_LITTLE_ENDIAN__ "
        "-D__ORDER_LITTLE_ENDIAN__=1234 -D__ORDER_BIG_ENDIAN__=4321"
    )
    if compile_db_entries > 0 and compile_db_path:
        mode = "compile_commands"
        cpp_cmd = (
            f"cppcheck {cpp_platform_flags} --project={json.dumps(compile_db_path)} "
            "--enable=warning,style,performance,portability "
            "--xml --xml-version=2 --suppress=missingIncludeSystem --inline-suppr "
            f"-j{MAKE_JOBS} 2> /tmp/ac_cppcheck.xml"
        )
    else:
        mode = "directory_scan"
        cpp_cmd = (
            f"cppcheck {cpp_platform_flags} --enable=warning,style,performance,portability "
            "--xml --xml-version=2 --suppress=missingIncludeSystem --inline-suppr "
            "-i /work/.git -i /work/CMakeFiles -i /work/build "
            f"-j{MAKE_JOBS} /work 2> /tmp/ac_cppcheck.xml"
        )

    _exec_in_container(shell, f"{cpp_cmd} || true", workdir=workdir, timeout=600)

    parse_script = r"""
import json, os, xml.etree.ElementTree as ET

p = '/tmp/ac_cppcheck.xml'
if not os.path.exists(p) or os.path.getsize(p) == 0:
    print('__AC_CPP__' + json.dumps({'ok': False, 'error': 'empty xml'}))
else:
    try:
        tree = ET.parse(p)
        root = tree.getroot()
        errors_el = root.find('errors')
        by_sev = {}
        findings = []
        seen = set()
        if errors_el is not None:
            for err in errors_el.findall('error'):
                eid = err.attrib.get('id', '')
                if eid in ('toomanyconfigs', 'missingInclude', 'missingIncludeSystem'):
                    continue
                sev = err.attrib.get('severity', 'unknown')
                msg = err.attrib.get('msg', '')
                verbose = err.attrib.get('verbose', '')
                cwe = err.attrib.get('cwe', '')
                loc = err.find('location')
                fpath = loc.attrib.get('file', '') if loc is not None else ''
                line = int(loc.attrib.get('line', '0') or 0) if loc is not None else 0
                if fpath.startswith('/work/'):
                    fpath = fpath[len('/work/'):]
                key = (eid, sev, fpath, line, msg)
                if key in seen:
                    continue
                seen.add(key)
                by_sev[sev] = by_sev.get(sev, 0) + 1
                if len(findings) < 200:
                    findings.append({
                        'id': eid,
                        'severity': sev,
                        'msg': msg,
                        'verbose': verbose,
                        'cwe': cwe,
                        'file': fpath,
                        'line': line,
                    })
        print('__AC_CPP__' + json.dumps({
            'ok': True,
            'total_findings': len(seen),
            'by_severity': by_sev,
            'findings': findings,
        }))
    except Exception as exc:
        print('__AC_CPP__' + json.dumps({'ok': False, 'error': str(exc)}))
"""
    parsed = _run_container_python(shell, parse_script, "__AC_CPP__")
    _exec_in_container(
        shell,
        "rm -f /tmp/ac_compile_commands.json /tmp/compile_commands.json /tmp/ac_cppcheck.xml /tmp/ac_bear.log",
        timeout=15,
    )

    if not parsed or not parsed.get("ok"):
        err_report = {
            "status": "error",
            "mode": mode,
            "compile_db_entries": compile_db_entries,
            "total_findings": 0,
            "by_severity": {},
            "findings": [],
            "error": (parsed or {}).get("error", "parse_failed"),
        }
        with open(host_out, "w", encoding="utf-8") as fp:
            json.dump(err_report, fp, indent=2, ensure_ascii=False)
        return err_report

    summary = {
        "status": "completed",
        "mode": mode,
        "compile_db_entries": compile_db_entries,
        "report_path": host_out,
        "total_findings": parsed["total_findings"],
        "by_severity": parsed["by_severity"],
        "findings": parsed["findings"],
    }
    with open(host_out, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    return summary


def run_layer1_sast(
    shell,
    local_path: str,
    log_path: str,
    proj_name: str,
    recipe: Dict[str, Any],
    has_baseline_binaries: bool = True,
) -> Dict[str, Any]:
    """
    Layer 1: Run Semgrep (custom C rules + p/c) and compile_commands.json + Cppcheck
    inside the container against /work.
    """
    semgrep_summary = _run_semgrep_in_container(shell, log_path, proj_name)
    cppcheck_summary = _run_cppcheck_in_container(
        shell,
        log_path,
        proj_name,
        recipe,
        has_baseline_binaries=has_baseline_binaries,
    )
    return {
        "semgrep": semgrep_summary,
        "cppcheck": cppcheck_summary,
    }


def audit_elf_binaries(
    shell,
    local_path: str,
    target_files: Optional[List[str]] = None,
    label: str = "baseline",
    ignore_relpaths: Optional[set] = None,
) -> Dict[str, Any]:
    """
    Layer 2: Audit compiled ELF artifacts inside the container using readelf
    for NX, Stack Canary, PIE, RELRO, FORTIFY_SOURCE, RPATH/RUNPATH, symbols,
    and sanitizer instrumentation.
    """
    artifacts_info = find_compiled_artifacts(
        local_path, target_files, shell=shell, ignore_relpaths=ignore_relpaths
    )
    matched = list(artifacts_info.get("matched_targets") or [])
    all_art = list(artifacts_info.get("all_artifacts") or [])

    # Prefer linked binaries (executables and .so), fall back to .a/.o only if no linked binaries exist
    ordered: List[str] = []
    seen = set()
    for p in matched + all_art:
        if p not in seen:
            seen.add(p)
            ordered.append(p)

    linked = [p for p in ordered if not p.endswith((".o", ".a", ".lo"))]
    archives = [p for p in ordered if p.endswith(".a")]
    selected = (linked or archives or ordered)[:15]

    if not selected:
        return {
            "label": label,
            "binary_count": 0,
            "average_score": 0.0,
            "binaries": [],
        }

    audit_script = f"""
import json, os, re, subprocess

rels = {json.dumps(selected)}
_FORTIFY_RE = re.compile(r"^__[a-z0-9_]+_chk(?:@|$)")

def run_readelf(args):
    try:
        r = subprocess.run(["readelf"] + args, capture_output=True, text=True, timeout=15)
        return r.stdout + "\\n" + r.stderr
    except Exception as exc:
        return ""

results = []
seen_real = set()
for rel in rels:
    path = os.path.join("/work", rel)
    if not os.path.exists(path):
        continue
    real_p = os.path.realpath(path)
    if real_p in seen_real:
        continue
    seen_real.add(real_p)
    with open(path, "rb") as f:
        magic = f.read(8)

    if magic in (b"!<arch>\\n", b"!<thin>\\n"):
        sym_out = run_readelf(["-s", "-W", path])
        has_canary = "__stack_chk_fail" in sym_out or "__stack_chk_guard" in sym_out
        fort_syms = sorted(set(
            tok for tok in re.findall(r"__[a-z0-9_]+_chk(?:@[A-Za-z0-9_.]+)?", sym_out)
            if not tok.startswith("__stack_chk")
        ))
        sec_out = run_readelf(["-S", "-W", path])
        nx = True if ".note.GNU-stack" in sec_out else None
        asan = "__asan_init" in sym_out
        ubsan = "__ubsan_handle" in sym_out
        results.append({{
            "artifact": rel,
            "kind": "static_archive",
            "elf_type": "AR",
            "nx": "Enabled (obj)" if nx else "Unknown",
            "canary": "Enabled" if has_canary else "Disabled",
            "pie": "N/A (archive)",
            "relro": "N/A (archive)",
            "fortify": f"Enabled ({{len(fort_syms)}} _chk)" if fort_syms else "None (0 _chk)",
            "fortify_symbols": fort_syms[:20],
            "rpath": "None",
            "stripped": "Unknown",
            "has_debug_info": ".debug_info" in sec_out,
            "asan": asan,
            "ubsan": ubsan,
            "score": None,
        }})
        continue

    if magic[:4] != b"\\x7fELF":
        continue

    hdr = run_readelf(["-h", "-W", path])
    phdr = run_readelf(["-l", "-W", path])
    dyn = run_readelf(["-d", "-W", path])
    sym = run_readelf(["-s", "-W", path])
    sec = run_readelf(["-S", "-W", path])

    elf_type = "UNKNOWN"
    m_type = re.search(r"Type:\\s+([A-Z0-9_]+)", hdr)
    if m_type:
        elf_type = m_type.group(1)

    # NX
    nx_enabled = False
    m_stack = re.search(r"GNU_STACK\\s+.*?(RWE|RWX|R\\s*W\\s*E|RW|R)", phdr)
    if m_stack:
        nx_enabled = "E" not in m_stack.group(1) and "X" not in m_stack.group(1)
    elif elf_type == "REL" and ".note.GNU-stack" in sec:
        nx_enabled = True

    # Canary
    has_canary = "__stack_chk_fail" in sym or "__stack_chk_guard" in sym

    # RELRO
    has_relro = "GNU_RELRO" in phdr
    has_bind_now = bool(re.search(r"\\b(BIND_NOW|NOW)\\b", dyn))
    if elf_type in ("DYN", "EXEC"):
        if has_relro and has_bind_now:
            relro_str = "Full RELRO"
        elif has_relro:
            relro_str = "Partial RELRO"
        else:
            relro_str = "No RELRO"
    else:
        relro_str = "N/A (REL)"

    # PIE
    has_interp = "INTERP" in phdr
    has_df1_pie = bool(re.search(r"\\(FLAGS_1\\).*\\bPIE\\b", dyn))
    is_so_name = ".so" in os.path.basename(rel)
    if elf_type == "EXEC":
        pie_str = "No PIE"
    elif elf_type == "DYN":
        if has_df1_pie or (has_interp and not is_so_name):
            pie_str = "PIE Enabled"
        else:
            pie_str = "DSO (PIC)"
    elif elf_type == "REL":
        pie_str = "REL Object"
    else:
        pie_str = elf_type

    # FORTIFY_SOURCE
    fort_syms = []
    for line in sym.splitlines():
        parts = line.split()
        if parts:
            sname = parts[-1]
            if _FORTIFY_RE.match(sname) and not sname.startswith("__stack_chk"):
                fort_syms.append(sname.split("@")[0])
    fort_syms = sorted(set(fort_syms))

    # RPATH / RUNPATH
    rpaths = re.findall(r"\\((?:RPATH|RUNPATH)\\)\\s+Library [^:]+:\\s+\\[([^\\]]+)\\]", dyn)
    rpath_str = "; ".join(rpaths) if rpaths else "None"

    # Symbols / Stripped
    stripped = ".symtab" not in sec
    has_debug = ".debug_info" in sec

    # Sanitizers
    asan = "__asan_init" in sym or "libasan" in dyn
    ubsan = "__ubsan_handle" in sym or "libubsan" in dyn

    # Hardening score (0-100) for linked ELF
    score = None
    kind = "shared_library" if (elf_type == "DYN" and pie_str.startswith("DSO")) else "executable"
    if elf_type in ("DYN", "EXEC"):
        score = 0
        if nx_enabled:
            score += 20
        if has_canary:
            score += 20
        if pie_str in ("PIE Enabled", "DSO (PIC)"):
            score += 20
        if relro_str == "Full RELRO":
            score += 20
        elif relro_str == "Partial RELRO":
            score += 10
        if fort_syms:
            score += 20

    results.append({{
        "artifact": rel,
        "kind": kind if elf_type != "REL" else "relocatable",
        "elf_type": elf_type,
        "nx": "Enabled" if nx_enabled else "Disabled",
        "canary": "Enabled" if has_canary else "Disabled",
        "pie": pie_str,
        "relro": relro_str,
        "fortify": f"Enabled ({{len(fort_syms)}} _chk)" if fort_syms else "Disabled (0 _chk)",
        "fortify_symbols": fort_syms[:20],
        "rpath": rpath_str,
        "stripped": stripped,
        "has_debug_info": has_debug,
        "asan": asan,
        "ubsan": ubsan,
        "score": score,
    }})

if any(r.get("elf_type") in ("EXEC", "DYN") for r in results):
    results = [r for r in results if r.get("elf_type") != "REL"]

print("__AC_AUD__" + json.dumps(results))
"""
    binaries = _run_container_python(shell, audit_script, "__AC_AUD__") or []
    scored = [b["score"] for b in binaries if isinstance(b.get("score"), (int, float))]
    avg_score = round(sum(scored) / len(scored), 2) if scored else 0.0
    return {
        "label": label,
        "binary_count": len(binaries),
        "average_score": avg_score,
        "binaries": binaries,
    }


def _build_before_after_diff(
    before_audit: Dict[str, Any], after_audit: Dict[str, Any]
) -> Dict[str, Any]:
    before_list = list(before_audit.get("binaries") or [])
    after_list = list(after_audit.get("binaries") or [])

    before_by_rel = {b["artifact"]: b for b in before_list}
    before_by_base = {os.path.basename(b["artifact"]): b for b in before_list}

    comparisons = []
    for after in after_list:
        rel = after["artifact"]
        prev = before_by_rel.get(rel) or before_by_base.get(os.path.basename(rel))
        if not prev:
            comparisons.append(
                {
                    "artifact": rel,
                    "before": None,
                    "after": after,
                    "score_delta": None,
                }
            )
            continue
        b_score = prev.get("score")
        a_score = after.get("score")
        delta = (
            round(a_score - b_score, 2)
            if isinstance(b_score, (int, float)) and isinstance(a_score, (int, float))
            else None
        )
        comparisons.append(
            {
                "artifact": rel,
                "before": {
                    "nx": prev.get("nx"),
                    "canary": prev.get("canary"),
                    "pie": prev.get("pie"),
                    "relro": prev.get("relro"),
                    "fortify": prev.get("fortify"),
                    "score": b_score,
                },
                "after": {
                    "nx": after.get("nx"),
                    "canary": after.get("canary"),
                    "pie": after.get("pie"),
                    "relro": after.get("relro"),
                    "fortify": after.get("fortify"),
                    "score": a_score,
                },
                "score_delta": delta,
            }
        )

    b_avg = before_audit.get("average_score", 0.0)
    a_avg = after_audit.get("average_score", 0.0)
    return {
        "before_avg_score": b_avg,
        "after_avg_score": a_avg,
        "avg_score_delta": round(a_avg - b_avg, 2),
        "binaries": comparisons,
    }


def _rebuild_with_wrapper(
    shell, recipe: Dict[str, Any], wrapper_flags: str
) -> Dict[str, Any]:
    workdir = recipe.get("workdir") or "/work"
    clean_cmd = recipe.get("clean_cmd", "")
    build_cmd = recipe.get("build_cmd", "")
    if not build_cmd:
        return {"status": "skipped", "reason": "No build command detected"}

    obj_cleanup = (
        "find . -maxdepth 4 \\( -name '*.o' -o -name '*.lo' -o -name '*.obj' "
        "-o -path '*/_o/*.o' \\) -delete >/dev/null 2>&1 || true"
    )
    if clean_cmd:
        clean_step = f"(({clean_cmd}) >/dev/null 2>&1 || true) && ({obj_cleanup}) && "
    else:
        clean_step = f"({obj_cleanup}) && "
    full_cmd = (
        f"{clean_step}"
        f"ENABLE_COMPILER_WRAPPER=1 WRAPPER_OPTI={json.dumps(wrapper_flags)} {build_cmd}"
    )
    res = _exec_in_container(shell, full_cmd, workdir=workdir, timeout=900)
    ec = res["returncode"]
    combined = ((res.get("stdout") or "") + "\n" + (res.get("stderr") or "")).strip()
    return {
        "status": "completed" if ec == 0 else "failed",
        "exit_code": ec,
        "build_cmd": build_cmd,
        "workdir": workdir,
        "wrapper_flags": wrapper_flags,
        "output_tail": combined[-2000:],
    }


def _parse_sanitizer_output(text: str) -> Dict[str, Any]:
    asan_errors: List[str] = []
    ubsan_errors: List[str] = []
    seen_asan = set()
    seen_ubsan = set()

    for line in (text or "").splitlines():
        s = line.strip()
        m_asan = re.search(r"ERROR:\s+AddressSanitizer:\s*([^\n]+)", s)
        if m_asan:
            msg = m_asan.group(0)
            if msg not in seen_asan:
                seen_asan.add(msg)
                asan_errors.append(msg)
        m_ub = re.search(r"[^\s:]+\.c(?:c|pp|xx)?:\d+:\d+:\s+runtime error:\s*([^\n]+)", s)
        if m_ub:
            msg = m_ub.group(0)
            if msg not in seen_ubsan:
                seen_ubsan.add(msg)
                ubsan_errors.append(msg)
        elif "runtime error:" in s and s not in seen_ubsan:
            seen_ubsan.add(s)
            ubsan_errors.append(s)

    return {
        "asan_count": len(asan_errors),
        "ubsan_count": len(ubsan_errors),
        "asan_errors": asan_errors[:30],
        "ubsan_errors": ubsan_errors[:30],
    }


def _run_gdb_backtrace(shell, binary_path: str, args: str = "", workdir: str = "/work") -> str:
    cmd = (
        f"timeout 10 gdb -nx -q -batch "
        f"-ex 'set pagination off' "
        f"-ex 'run {args}' "
        f"-ex 'bt' "
        f"-ex 'quit' {json.dumps(binary_path)} 2>&1 || true"
    )
    res = _exec_in_container(shell, cmd, workdir=workdir, timeout=20)
    combined = ((res.get("stdout") or "") + "\n" + (res.get("stderr") or "")).strip()
    return combined[-2000:]


def _collect_sanitizer_ld_library_path(
    san_audit: Dict[str, Any], artifacts_info: Optional[Dict[str, Any]] = None, workdir: str = "/work"
) -> str:
    dirs: List[str] = []
    for b in (san_audit.get("binaries") or []):
        rel = str(b.get("artifact") or "")
        base = posixpath.basename(rel)
        if b.get("kind") == "shared_library" or base.endswith(".so") or ".so." in base:
            d = posixpath.dirname(f"/work/{rel.lstrip('/')}")
            if d and d not in dirs:
                dirs.append(d)
    for rel in ((artifacts_info or {}).get("all_artifacts") or []):
        base = posixpath.basename(str(rel))
        if base.endswith(".so") or ".so." in base:
            d = posixpath.dirname(f"/work/{str(rel).lstrip('/')}")
            if d and d not in dirs:
                dirs.append(d)
    for extra in (workdir, "/work"):
        if extra and extra not in dirs:
            dirs.append(extra)
    return ":".join(dirs)


def _run_sanitizer_dynamic_checks(
    shell,
    local_path: str,
    target_files: Optional[List[str]],
    recipe: Dict[str, Any],
    san_audit: Dict[str, Any],
    ignore_relpaths: Optional[set] = None,
) -> Dict[str, Any]:
    workdir = recipe.get("workdir") or "/work"
    found = find_compiled_artifacts(
        local_path, target_files, shell=shell, ignore_relpaths=ignore_relpaths
    )
    ld_library_path = _collect_sanitizer_ld_library_path(san_audit, found, workdir=workdir)
    safe_ld = ld_library_path.replace('"', '\\"').replace("$", "\\$")
    san_env = (
        f"ENABLE_COMPILER_WRAPPER=1 WRAPPER_OPTI={json.dumps(SANITIZER_WRAPPER_FLAGS)} "
        f'LD_LIBRARY_PATH="{safe_ld}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}" '
        "ASAN_OPTIONS=detect_leaks=0:halt_on_error=0:print_stacktrace=1 "
        "UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=0"
    )
    test_cmd = recipe.get("test_cmd", "")
    runs: List[Dict[str, Any]] = []
    combined_output = []

    if test_cmd:
        res = _exec_in_container(
            shell,
            f"{san_env} timeout 60 {test_cmd} 2>&1",
            workdir=workdir,
            timeout=90,
        )
        ec = res["returncode"]
        cleaned = (res.get("stdout") or "").strip()
        combined_output.append(cleaned)
        runs.append(
            {
                "kind": "test_suite",
                "command": test_cmd,
                "exit_code": ec,
                "output_tail": cleaned[-3000:],
            }
        )

    # Also locate compiled executable binaries for smoke / trigger runs when no test_cmd
    # or when executables may expose additional sanitizer findings
    executables = [
        b["artifact"]
        for b in (san_audit.get("binaries") or [])
        if b.get("kind") == "executable"
    ]
    if not executables:
        executables = [
            p for p in (found.get("matched_targets") or found.get("all_artifacts") or [])
            if not p.endswith((".a", ".o", ".lo", ".so")) and ".so." not in os.path.basename(p)
        ]

    for rel in executables[:3]:
        cpath = f"/work/{rel}"
        # Skip smoke invocation if test suite already ran and tested it, unless test_cmd was empty
        args = "--test" if (not test_cmd and "demo" in os.path.basename(rel).lower()) else "--help"
        if not test_cmd:
            res = _exec_in_container(
                shell,
                f"{san_env} timeout 10 {json.dumps(cpath)} {args} 2>&1",
                workdir=workdir,
                timeout=20,
            )
            ec = res["returncode"]
            cleaned = (res.get("stdout") or "").strip()
            if "AddressSanitizer" not in cleaned and "runtime error:" not in cleaned and ec not in (0,):
                # Retry without args (e.g., unit test binaries like ./test or ./test.elf that take no flags)
                res2 = _exec_in_container(
                    shell,
                    f"{san_env} timeout 10 {json.dumps(cpath)} 2>&1",
                    workdir=workdir,
                    timeout=20,
                )
                ec2 = res2["returncode"]
                cleaned2 = (res2.get("stdout") or "").strip()
                if cleaned2:
                    ec, cleaned = ec2, cleaned2
                    args = ""
            combined_output.append(cleaned)
            gdb_bt = ""
            if ec is not None and ec >= 128 and "AddressSanitizer" not in cleaned:
                gdb_bt = _run_gdb_backtrace(shell, cpath, args, workdir=workdir)
            runs.append(
                {
                    "kind": "executable_smoke",
                    "artifact": rel,
                    "args": args,
                    "exit_code": ec,
                    "gdb_backtrace": gdb_bt,
                    "output_tail": cleaned[-2000:],
                }
            )

    parsed_san = _parse_sanitizer_output("\n".join(combined_output))
    return {
        "asan_instrumented_count": sum(
            1 for b in (san_audit.get("binaries") or []) if b.get("asan")
        ),
        "ubsan_instrumented_count": sum(
            1 for b in (san_audit.get("binaries") or []) if b.get("ubsan")
        ),
        "runs": runs,
        **parsed_san,
    }


def run_security_pipeline(
    shell,
    local_path: str,
    log_path: str,
    proj_name: str,
    target_files: Optional[List[str]],
    tools_log: Dict[str, Any],
    security_mode: str = "all",
    ignore_relpaths: Optional[set] = None,
) -> Dict[str, Any]:
    """
    Execute the 3-layer security pipeline once per project inside the active Docker container:
      Layer 1: Build-aware SAST (Semgrep + Bear/CMake compile_commands.json + Cppcheck)
      Layer 2: Baseline ELF binary hardening audit via readelf
      Layer 3: Hardening rebuild (+ Before/After comparison) and Sanitizer (ASan/UBSan) rebuild & run
    """
    mode = (security_mode or "all").lower()
    if mode == "none":
        return {"mode": "none", "status": "skipped"}

    os.makedirs(os.path.join(log_path, "security"), exist_ok=True)
    report: Dict[str, Any] = {
        "mode": mode,
        "status": "completed",
        "recipe": {},
        "layer1_sast": {},
        "layer2_hardening_baseline": {},
        "layer3_hardening_rebuild": {},
        "layer3_sanitizer": {},
    }

    # Always inspect build recipe and baseline binaries before any clean/rebuild
    recipe = detect_build_recipe(shell, tools_log)
    report["recipe"] = recipe

    baseline_audit = audit_elf_binaries(
        shell, local_path, target_files, label="baseline", ignore_relpaths=ignore_relpaths
    )
    report["layer2_hardening_baseline"] = baseline_audit

    if mode in ("sast", "all"):
        report["layer1_sast"] = run_layer1_sast(
            shell,
            local_path,
            log_path,
            proj_name,
            recipe,
            has_baseline_binaries=(baseline_audit.get("binary_count", 0) > 0),
        )

    if mode in ("sanitizer", "all"):
        if baseline_audit.get("binary_count", 0) == 0:
            report["layer3_sanitizer"] = {
                "build": {"status": "skipped", "reason": "No baseline ELF/archive artifacts compiled"},
                "audit_sanitizer": {"label": "sanitizer", "binary_count": 0, "average_score": 0.0, "binaries": []},
                "asan_instrumented_count": 0,
                "ubsan_instrumented_count": 0,
                "runs": [],
                "asan_count": 0,
                "ubsan_count": 0,
                "asan_errors": [],
                "ubsan_errors": [],
            }
        else:
            san_build = _rebuild_with_wrapper(shell, recipe, SANITIZER_WRAPPER_FLAGS)
            if san_build.get("status") == "completed":
                san_audit = audit_elf_binaries(
                    shell, local_path, target_files, label="sanitizer", ignore_relpaths=ignore_relpaths
                )
                dyn = _run_sanitizer_dynamic_checks(
                    shell, local_path, target_files, recipe, san_audit, ignore_relpaths=ignore_relpaths
                )
            else:
                san_audit = {"label": "sanitizer", "binary_count": 0, "average_score": 0.0, "binaries": []}
                dyn = {"asan_instrumented_count": 0, "ubsan_instrumented_count": 0, "runs": [], "asan_count": 0, "ubsan_count": 0, "asan_errors": [], "ubsan_errors": []}
            report["layer3_sanitizer"] = {
                "build": san_build,
                "audit_sanitizer": san_audit,
                **dyn,
            }

    if mode in ("hardening", "all"):
        if baseline_audit.get("binary_count", 0) == 0:
            report["layer3_hardening_rebuild"] = {
                "build": {"status": "skipped", "reason": "No baseline ELF/archive artifacts compiled"},
                "audit_after": {"label": "after_hardening", "binary_count": 0, "average_score": 0.0, "binaries": []},
                "smoke_check_after": {"checked_count": 0, "passed": False, "missing_libs": [], "details": []},
                "before_vs_after": _build_before_after_diff(baseline_audit, {"binaries": [], "average_score": 0.0}),
            }
        else:
            rebuild_info = _rebuild_with_wrapper(shell, recipe, HARDENING_WRAPPER_FLAGS)
            after_audit = (
                audit_elf_binaries(
                    shell, local_path, target_files, label="after_hardening", ignore_relpaths=ignore_relpaths
                )
                if rebuild_info.get("status") == "completed"
                else {"label": "after_hardening", "binary_count": 0, "average_score": 0.0, "binaries": []}
            )
            hardened_smoke = (
                smoke_check_artifact(shell, local_path, target_files, ignore_relpaths=ignore_relpaths)
                if rebuild_info.get("status") == "completed"
                else {"checked_count": 0, "passed": False, "missing_libs": [], "details": []}
            )
            report["layer3_hardening_rebuild"] = {
                "build": rebuild_info,
                "audit_after": after_audit,
                "smoke_check_after": hardened_smoke,
                "before_vs_after": _build_before_after_diff(baseline_audit, after_audit),
            }

    out_file = os.path.join(log_path, "security", f"{proj_name}.security.json")
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)
    report["report_path"] = out_file

    logging.info(
        "[Security][%s] mode=%s semgrep=%s cppcheck=%s baseline_score=%.1f after_score=%s asan=%s ubsan=%s",
        proj_name,
        mode,
        report.get("layer1_sast", {}).get("semgrep", {}).get("total_findings", "N/A"),
        report.get("layer1_sast", {}).get("cppcheck", {}).get("total_findings", "N/A"),
        baseline_audit.get("average_score", 0.0),
        report.get("layer3_hardening_rebuild", {}).get("audit_after", {}).get("average_score", "N/A"),
        report.get("layer3_sanitizer", {}).get("asan_count", "N/A"),
        report.get("layer3_sanitizer", {}).get("ubsan_count", "N/A"),
    )
    return report
