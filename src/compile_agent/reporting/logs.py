import base64
import csv
import datetime
import json
import logging
import os
import posixpath
import subprocess
from typing import Optional

from compile_agent.config import *


def is_elf(file):
    try:
        if not os.path.isfile(file) or os.path.getsize(file) <= 0:
            return False
        with open(file, "rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


def is_archive(file):
    try:
        if not os.path.isfile(file) or os.path.getsize(file) <= 0:
            return False
        with open(file, "rb") as f:
            return f.read(8) in {b"!<arch>\n", b"!<thin>\n"}
    except OSError:
        return False


_PROBE_DIR_NAMES = frozenset({
    ".git", "CMakeFiles", ".deps", "autom4te.cache",
    "CompilerIdC", "CompilerIdCXX", "CheckTypeSize", "CMakeTmp",
})
_PROBE_FILE_PREFIXES = (
    "conftest",
    "CMakeDetermineCompilerABI",
    "CMakeCCompilerId",
    "CMakeCXXCompilerId",
)


def _is_build_system_probe_artifact(rel_path: str) -> bool:
    """Return True if rel_path belongs to a CMake/Autoconf compiler probe or internal cache."""
    if not rel_path:
        return False
    norm = rel_path.replace("\\", "/").strip("/")
    parts = [p for p in norm.split("/") if p and p != "."]
    if not parts:
        return False
    for part in parts[:-1]:
        if part in _PROBE_DIR_NAMES:
            return True
    fname = parts[-1]
    if fname in _PROBE_DIR_NAMES:
        return True
    if fname.startswith(_PROBE_FILE_PREFIXES):
        return True
    return False


def _matches_target_name(actual_filename: str, target_basename: str) -> bool:
    """Match compiled artifact filename against expected target basename."""
    if not actual_filename or not target_basename:
        return False
    actual_base = os.path.basename(actual_filename.replace("\\", "/"))
    target_base = os.path.basename(target_basename.replace("\\", "/"))
    if actual_base == target_base:
        return True
    actual_norm = actual_base.lower().replace("-", "_")
    target_norm = target_base.lower().replace("-", "_")
    if actual_norm == target_norm:
        return True
    if target_base.lower().endswith(".so"):
        if actual_base.startswith(target_base + ".") or actual_norm.startswith(target_norm + "."):
            return True
    return False


def snapshot_existing_artifacts(local_path: str) -> set:
    """Scan local_path on host before compilation and return relative paths of pre-existing ELF/archive files."""
    existing = set()
    if not local_path or not os.path.isdir(local_path):
        return existing
    for root, dirs, files in os.walk(local_path):
        dirs[:] = [d for d in dirs if d != ".git"]
        for fname in files:
            fpath = os.path.join(root, fname)
            if os.path.islink(fpath):
                continue
            if is_elf(fpath) or is_archive(fpath):
                rel = os.path.relpath(fpath, local_path).replace("\\", "/").lstrip("./")
                existing.add(rel)
    return existing


def find_compiled_artifacts(
    local_path: str,
    target_files: Optional[list] = None,
    shell=None,
    ignore_relpaths: Optional[set] = None,
) -> dict:
    target_names = sorted(
        {os.path.basename(t.replace("\\", "/")) for t in (target_files or []) if t}
    )
    ignore_set = {
        p.replace("\\", "/").lstrip("./")
        for p in (ignore_relpaths or set())
        if p
    }
    if shell is not None:
        try:
            payload_src = (
                "import os, json\n"
                f"targets = {json.dumps(target_names)}\n"
                f"ignores = set({json.dumps(sorted(ignore_set))})\n"
                "probe_dirs = {'.git', 'CMakeFiles', '.deps', 'autom4te.cache', 'CompilerIdC', 'CompilerIdCXX', 'CheckTypeSize', 'CMakeTmp'}\n"
                "probe_prefixes = ('conftest', 'CMakeDetermineCompilerABI', 'CMakeCCompilerId', 'CMakeCXXCompilerId')\n"
                "def _is_probe(rel_p):\n"
                "    parts = [x for x in rel_p.replace('\\\\', '/').strip('/').split('/') if x and x != '.']\n"
                "    if not parts:\n"
                "        return False\n"
                "    if any(x in probe_dirs for x in parts[:-1]):\n"
                "        return True\n"
                "    fn = parts[-1]\n"
                "    return fn in probe_dirs or fn.startswith(probe_prefixes)\n"
                "def _match_target(fn, tb):\n"
                "    if fn == tb:\n"
                "        return True\n"
                "    fn_n = fn.lower().replace('-', '_')\n"
                "    tb_n = tb.lower().replace('-', '_')\n"
                "    if fn_n == tb_n:\n"
                "        return True\n"
                "    if tb.lower().endswith('.so') and (fn.startswith(tb + '.') or fn_n.startswith(tb_n + '.')):\n"
                "        return True\n"
                "    return False\n"
                "matched, non_obj, objs = [], [], []\n"
                "for r, ds, fs in os.walk('/work'):\n"
                "    ds[:] = [d for d in ds if d not in ('.git', 'CMakeFiles', '.deps', 'autom4te.cache')]\n"
                "    for fn in fs:\n"
                "        p = os.path.join(r, fn)\n"
                "        try:\n"
                "            if os.path.isfile(p) and os.path.getsize(p) > 0:\n"
                "                rel = os.path.relpath(p, '/work').replace('\\\\', '/').lstrip('./')\n"
                "                if _is_probe(rel) or rel in ignores:\n"
                "                    continue\n"
                "                with open(p, 'rb') as f:\n"
                "                    h = f.read(8)\n"
                "                if h[:4] == b'\\x7fELF' or h in (b'!<arch>\\n', b'!<thin>\\n'):\n"
                "                    if any(_match_target(fn, tb) for tb in targets):\n"
                "                        matched.append(rel)\n"
                "                    if not fn.endswith(('.o', '.lo')):\n"
                "                        non_obj.append(rel)\n"
                "                    else:\n"
                "                        objs.append(rel)\n"
                "        except OSError:\n"
                "            pass\n"
                "all_a = sorted(set(matched + non_obj))[:50] or sorted(objs)[:50]\n"
                "print('__AC_ART__' + json.dumps({'matched_targets': sorted(matched), 'all_artifacts': all_a}))\n"
            )
            container_id = getattr(shell, "container_id", None)
            if container_id:
                proc = subprocess.run(
                    ["docker", "exec", "-i", "-w", "/work", container_id, "python3", "-"],
                    input=payload_src,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=60,
                )
                out = proc.stdout or ""
            else:
                payload_b64 = base64.b64encode(payload_src.encode("utf-8")).decode("ascii")
                out = shell.execute_command(
                    f"python3 -c \"import base64; exec(base64.b64decode('{payload_b64}'))\"",
                    record_log=False,
                )
            for line in out.splitlines():
                s = line.strip()
                if s.startswith("__AC_ART__"):
                    parsed = json.loads(s[len("__AC_ART__") :])
                    if isinstance(parsed, dict) and "matched_targets" in parsed and "all_artifacts" in parsed:
                        return parsed
        except Exception as err:
            logging.warning(f"[!] Container artifact scan failed, falling back to host scan: {err}")

    all_non_obj = []
    all_objs = []
    matched_targets = []
    for root, dirs, files in os.walk(local_path):
        dirs[:] = [
            d for d in dirs if d not in (".git", "CMakeFiles", ".deps", "autom4te.cache")
        ]
        for file in files:
            fp = os.path.join(root, file)
            rel = os.path.relpath(fp, local_path).replace("\\", "/").lstrip("./")
            if _is_build_system_probe_artifact(rel) or rel in ignore_set:
                continue
            if is_elf(fp) or is_archive(fp):
                if any(_matches_target_name(file, tb) for tb in target_names):
                    matched_targets.append(rel)
                if not file.endswith((".o", ".lo")):
                    all_non_obj.append(rel)
                else:
                    all_objs.append(rel)
    all_artifacts = sorted(set(matched_targets + all_non_obj))[:50] or sorted(all_objs)[:50]
    return {"all_artifacts": all_artifacts, "matched_targets": sorted(matched_targets)}


def is_compiled(local_path, target_files, strict=False, shell=None, precomputed_found=None):
    if not target_files:
        return None
    found = precomputed_found if precomputed_found is not None else find_compiled_artifacts(local_path, target_files, shell=shell)
    valid_matched = [
        p for p in (found.get("matched_targets") or [])
        if not _is_build_system_probe_artifact(p)
    ]
    matched_basenames = [os.path.basename(p.replace("\\", "/")) for p in valid_matched]
    target_basenames = [os.path.basename(f.replace("\\", "/")) for f in target_files if f]
    if not target_basenames:
        return None
    if strict:
        return all(any(_matches_target_name(cb, tb) for cb in matched_basenames) for tb in target_basenames)
    return any(any(_matches_target_name(cb, tb) for cb in matched_basenames) for tb in target_basenames)


def _build_container_ld_library_path(artifacts: list, extra_dirs=None) -> str:
    dirs = []
    for rel in artifacts or []:
        base = posixpath.basename(rel)
        if base.endswith(".so") or ".so." in base:
            d = posixpath.dirname(f"/work/{rel.lstrip('/')}")
            if d and d not in dirs:
                dirs.append(d)
    for d in extra_dirs or []:
        if d and d not in dirs:
            dirs.append(d)
    if "/work" not in dirs:
        dirs.append("/work")
    return ":".join(dirs)


def smoke_check_artifact(
    shell,
    local_path: str,
    target_files: Optional[list] = None,
    precomputed_found: Optional[dict] = None,
    ignore_relpaths: Optional[set] = None,
):
    ignore_set = {
        p.replace("\\", "/").lstrip("./")
        for p in (ignore_relpaths or set())
        if p
    }
    found = (
        precomputed_found
        if precomputed_found is not None
        else find_compiled_artifacts(local_path, target_files, shell=shell, ignore_relpaths=ignore_set)
    )
    matched = [
        p for p in (found.get("matched_targets") or [])
        if not _is_build_system_probe_artifact(p)
        and p.replace("\\", "/").lstrip("./") not in ignore_set
    ]
    all_art = [
        p for p in (found.get("all_artifacts") or [])
        if not _is_build_system_probe_artifact(p)
        and p.replace("\\", "/").lstrip("./") not in ignore_set
    ]
    if matched and all(p.endswith((".o", ".lo", ".a")) for p in matched):
        extra_linked = [
            p for p in all_art if not p.endswith((".o", ".lo", ".a")) and p not in matched
        ][:2]
        candidates = (matched + extra_linked)[:10]
    else:
        candidates = (matched if matched else all_art[:5])[:8]
    only_intermediate_objects = bool(
        not matched and candidates and all(c.lower().endswith((".o", ".lo")) for c in candidates)
    )
    lib_dirs_base = _build_container_ld_library_path(all_art)
    container_id = getattr(shell, "container_id", None)

    def _run_probe(cmd: str) -> str:
        if container_id:
            proc = subprocess.run(
                ["docker", "exec", "-w", "/work", container_id, "/bin/bash", "-c", cmd],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
                stdin=subprocess.DEVNULL,
            )
            return (proc.stdout or "") + (proc.stderr or "")
        return shell.execute_command(cmd, record_log=False)

    details = []
    missing_libs = []
    for rel in candidates[:10]:
        container_fp = f"/work/{rel}"
        file_out = ""
        ldd_out = ""
        try:
            file_out = _run_probe(f'file -L "{container_fp}"')
        except Exception as err:
            file_out = f"ERR: {err}"
        file_desc = ""
        for line in file_out.splitlines():
            stripped = line.strip()
            if stripped.startswith(f"{container_fp}:"):
                file_desc = stripped.split(":", 1)[1]
                break
        if not file_desc:
            for line in file_out.splitlines():
                stripped = line.strip()
                if "file -L" in stripped or stripped.startswith("[Exit code:") or "#" in stripped:
                    continue
                if ":" in stripped:
                    file_desc = stripped.split(":", 1)[1]
                    break
        file_ok = ("ELF" in file_desc) or ("ar archive" in file_desc)

        is_dynamic_candidate = "ELF" in file_out and not rel.endswith((".a", ".o", ".lo"))
        has_missing_lib = False
        if is_dynamic_candidate:
            ld_path = _build_container_ld_library_path(
                all_art,
                extra_dirs=[posixpath.dirname(container_fp)],
            ) or lib_dirs_base
            try:
                ldd_out = _run_probe(
                    f'LD_LIBRARY_PATH="{ld_path}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}" ldd "{container_fp}" 2>&1 || true'
                )
            except Exception as err:
                ldd_out = f"ERR: {err}"
            for line in ldd_out.splitlines():
                if "not found" in line:
                    has_missing_lib = True
                    missing_libs.append(f"{rel}: {line.strip()}")

        details.append(
            {
                "artifact": rel,
                "file_ok": file_ok,
                "missing_lib": has_missing_lib,
                "ok": file_ok and not has_missing_lib,
                "file_output": file_out.strip(),
                "ldd_output": ldd_out.strip(),
            }
        )

    return {
        "checked_count": len(details),
        "passed": bool(details) and all(d["ok"] for d in details) and not only_intermediate_objects,
        "missing_libs": missing_libs,
        "details": details,
    }


def assemble_all_to_txt(template, tools, input, output, intermediate_steps):
    tool_names = ", ".join([tool.name for tool in tools])
    tool_info = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
    agent_scratchpad = ""
    for idx, (action, value) in enumerate(intermediate_steps):
        model_response = getattr(action, "log", str(action))
        tool_response = value
        agent_scratchpad += "\nThought: " + str(model_response) if idx != 0 else str(model_response)
        agent_scratchpad += "\nObservation: " + str(tool_response)
    agent_scratchpad += "\nThought: " + str(output)
    result = template.format(
        tools=tool_info,
        tool_names=tool_names,
        input=input,
        agent_scratchpad=agent_scratchpad,
    )
    return result


def assemble_all_to_json(template, tools, input, output, intermediate_steps):
    tool_names = ", ".join([tool.name for tool in tools])
    tool_info = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
    init_question = template.format(
        tools=tool_info,
        tool_names=tool_names,
        input=input,
        agent_scratchpad="",
    )
    _pair = [init_question]
    log_list = []
    for idx, (action, value) in enumerate(intermediate_steps):
        model_response = getattr(action, "log", str(action))
        _pair.append(model_response)
        log_list.append(_pair)
        tool_response = value
        _pair = [str(tool_response)]
    final_output = str(output)
    _pair.append(final_output)
    log_list.append(_pair)
    return log_list


def save_logs(
    log_path,
    proj_name,
    question,
    tools,
    res,
    tools_logs,
    local_path,
    checker_ok,
    model_ok,
    semgrep_summary=None,
    build_metadata=None,
):
    os.makedirs(log_path, exist_ok=True)

    res = res or {}
    output_text = res.get("output", "")
    intermediate_steps = res.get("intermediate_steps", [])
    run_id = res.get("run_id", "unknown")

    # logging process
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    log_local_file = os.path.join(log_path, f"{proj_name}.log.{now}.txt")
    logging.info(f"[-] Saving logs to {log_local_file}")
    with open(log_local_file, "w", encoding="utf-8", errors="replace") as file:
        s = assemble_all_to_txt(template, tools, question, output_text, intermediate_steps)
        file.write(s)

    log_local_file_json = log_local_file.replace(".txt", ".json")
    logging.info(f"[-] Saving logs to {log_local_file_json}")
    build_info = {
        "model_result": model_ok,
        "checker_result": checker_ok,
    }
    if build_metadata:
        build_info.update(build_metadata)
    with open(log_local_file_json, "w", encoding="utf-8", errors="replace") as file:
        s = assemble_all_to_json(template, tools, question, output_text, intermediate_steps)
        json.dump(
            {
                "autocompiler": s,
                "tools": tools_logs,
                "build": build_info,
                "security": semgrep_summary,
            },
            file,
            indent="\t",
            ensure_ascii=False,
        )

    # logging result into the current run's log_path so concurrent/different runs stay isolated
    log_url = LOG_URL_TEMPLATE.format(run_id=run_id) if LOG_URL_TEMPLATE else ""
    process_id = os.getpid()
    process_csv_file = os.path.join(log_path, f"{process_id}.csv")
    with open(process_csv_file, "a", encoding="utf-8", errors="replace", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([run_id, log_url, log_local_file, local_path, model_ok, checker_ok])
