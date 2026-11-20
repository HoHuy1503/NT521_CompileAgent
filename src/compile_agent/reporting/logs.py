import base64
import csv
import datetime
import glob
import json
import logging
import os
import posixpath
import re
import subprocess
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

try:
    import fcntl
except ImportError:
    fcntl = None

try:
    import msvcrt
except ImportError:
    msvcrt = None

from compile_agent.config import *

RESULTS_JSONL_FILENAME = "results.jsonl"
_LOG_NAME_TS_RE = re.compile(
    r"^(?P<project>.+?)\.log\.(?P<ts>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d+)\.json$"
)


@contextmanager
def _file_lock(target_path: str, timeout: float = 15.0):
    """Cross-platform advisory file lock (flock on POSIX, msvcrt on Windows) for safe JSONL appends."""
    lock_path = f"{os.path.abspath(target_path)}.lock"
    parent = os.path.dirname(lock_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fp = open(lock_path, "a+b")
    locked = False
    try:
        deadline = time.monotonic() + max(0.5, float(timeout))
        while True:
            try:
                if fcntl is not None:
                    fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                elif msvcrt is not None:
                    fp.seek(0)
                    msvcrt.locking(fp.fileno(), msvcrt.LK_NBLCK, 1)
                locked = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    logging.warning("[!] File lock timeout on %s; proceeding unlocked.", lock_path)
                    break
                time.sleep(0.02)
        yield
    finally:
        if locked:
            try:
                if fcntl is not None:
                    fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
                elif msvcrt is not None:
                    fp.seek(0)
                    msvcrt.locking(fp.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        try:
            fp.close()
        except OSError:
            pass


def atomic_write_text(path: str, content: str) -> None:
    """Write text to a temp file in the same directory, fsync, and atomically replace target."""
    abs_path = os.path.abspath(path)
    parent = os.path.dirname(abs_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{abs_path}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    try:
        with open(tmp_path, "w", encoding="utf-8", errors="replace", newline="") as fp:
            fp.write(content)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp_path, abs_path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_json(path: str, data: Any, indent: Any = 2) -> None:
    """Serialize JSON and persist atomically with fsync to prevent corruption on power loss."""
    payload = json.dumps(data, indent=indent, ensure_ascii=False)
    atomic_write_text(path, payload)


def append_jsonl(path: str, entry: dict) -> None:
    """
    Append one JSON object as a single line with cross-process locking, trailing-newline recovery
    (in case a prior power loss left a truncated line without '\\n'), flush, and fsync.
    """
    abs_path = os.path.abspath(path)
    parent = os.path.dirname(abs_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    line_bytes = (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8", errors="replace")
    with _file_lock(abs_path):
        with open(abs_path, "a+b") as fp:
            fp.seek(0, os.SEEK_END)
            size = fp.tell()
            if size > 0:
                fp.seek(-1, os.SEEK_END)
                last_byte = fp.read(1)
                fp.seek(0, os.SEEK_END)
                if last_byte != b"\n":
                    fp.write(b"\n")
            fp.write(line_bytes)
            fp.flush()
            os.fsync(fp.fileno())


def iter_jsonl_records(path: str, repair_truncated_tail: bool = False) -> List[dict]:
    """
    Read valid JSON dict records from a JSONL file, gracefully skipping any corrupted/truncated
    lines left by power outages. When repair_truncated_tail=True, rewrites atomically if any
    corrupted lines were detected so future reads stay clean.
    """
    if not path or not os.path.isfile(path):
        return []
    records: List[dict] = []
    valid_raw_lines: List[str] = []
    bad_count = 0
    with _file_lock(path):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fp:
                for line_no, raw_line in enumerate(fp, start=1):
                    stripped = raw_line.strip()
                    if not stripped:
                        continue
                    try:
                        obj = json.loads(stripped)
                        if isinstance(obj, dict):
                            records.append(obj)
                            valid_raw_lines.append(stripped)
                        else:
                            bad_count += 1
                    except json.JSONDecodeError as err:
                        bad_count += 1
                        logging.warning(
                            "[!] Skipping corrupted/truncated JSONL line %d in %s (%s)",
                            line_no,
                            path,
                            err,
                        )
        except OSError as err:
            logging.warning("[!] Could not read JSONL file %s: %s", path, err)
            return []

        if bad_count > 0 and repair_truncated_tail:
            repaired_content = ("\n".join(valid_raw_lines) + "\n") if valid_raw_lines else ""
            try:
                atomic_write_text(path, repaired_content)
                logging.info(
                    "[+] Repaired %s by removing %d corrupted/truncated line(s); kept %d valid record(s).",
                    path,
                    bad_count,
                    len(records),
                )
            except OSError as err:
                logging.warning("[!] Failed to repair JSONL %s: %s", path, err)

    return records


def cleanup_stale_tmp_files(log_path: str) -> int:
    """Remove leftover *.tmp.* files from interrupted atomic writes after a crash/power loss."""
    if not log_path or not os.path.isdir(log_path):
        return 0
    removed = 0
    for fp in glob.glob(os.path.join(log_path, "*.tmp.*")):
        try:
            if os.path.isfile(fp):
                os.remove(fp)
                removed += 1
        except OSError:
            pass
    return removed


def load_results_jsonl_index(log_path: str) -> Dict[str, List[dict]]:
    """Return mapping project_name -> list of results.jsonl records sorted newest-first."""
    jsonl_path = os.path.join(log_path, RESULTS_JSONL_FILENAME)
    records = iter_jsonl_records(jsonl_path, repair_truncated_tail=False)
    index: Dict[str, List[dict]] = {}
    for rec in records:
        build = rec.get("build") if isinstance(rec.get("build"), dict) else {}
        proj = str(rec.get("project") or build.get("project") or "").strip()
        if not proj:
            continue
        index.setdefault(proj, []).append(rec)
    for proj in index:
        index[proj].sort(key=lambda r: str(r.get("timestamp") or ""), reverse=True)
    return index


def backfill_results_jsonl(log_path: str) -> int:
    """
    Backfill existing per-project *.log.*.json files in log_path into results.jsonl
    without duplicating already-recorded (project, timestamp) pairs.
    """
    if not log_path or not os.path.isdir(log_path):
        return 0
    jsonl_path = os.path.join(log_path, RESULTS_JSONL_FILENAME)
    existing = iter_jsonl_records(jsonl_path, repair_truncated_tail=True)
    seen = {
        (
            str(r.get("project") or (r.get("build") or {}).get("project") or ""),
            str(r.get("timestamp") or ""),
        )
        for r in existing
    }

    added = 0
    for fp in sorted(glob.glob(os.path.join(log_path, "*.log.*.json"))):
        fname = os.path.basename(fp)
        m = _LOG_NAME_TS_RE.match(fname)
        if m:
            fallback_proj = m.group("project")
            ts_key = m.group("ts")
        else:
            fallback_proj = fname.split(".log.")[0]
            ts_key = f"{os.path.getmtime(fp):020.6f}"
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, dict) or not isinstance(data.get("build"), dict):
            continue
        build = data["build"]
        proj = str(build.get("project") or fallback_proj)
        if (proj, ts_key) in seen:
            continue

        auto = data.get("autocompiler") or []
        final_tail = ""
        if isinstance(auto, list) and auto:
            last_pair = auto[-1]
            tail_str = " ".join(str(x) for x in last_pair) if isinstance(last_pair, list) else str(last_pair)
            final_tail = tail_str[-300:]

        entry = {
            "schema_version": 1,
            "timestamp": ts_key,
            "recorded_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "run_id": "",
            "project": proj,
            "url": "",
            "log_file": fp.replace("\\", "/"),
            "txt_log_file": fp[:-5].replace("\\", "/") + ".txt",
            "local_path": "",
            "final_output_tail": final_tail,
            "build": build,
        }
        append_jsonl(jsonl_path, entry)
        seen.add((proj, ts_key))
        added += 1
    return added


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
    build_metadata=None,
    project_url: str = "",
):
    os.makedirs(log_path, exist_ok=True)

    res = res or {}
    output_text = res.get("output", "")
    intermediate_steps = res.get("intermediate_steps", [])
    run_id = res.get("run_id", "unknown")

    # Atomic per-project TXT + JSON logs (safe against power loss mid-write)
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    log_local_file = os.path.join(log_path, f"{proj_name}.log.{now}.txt")
    logging.info(f"[-] Saving logs to {log_local_file}")
    txt_content = assemble_all_to_txt(template, tools, question, output_text, intermediate_steps)
    atomic_write_text(log_local_file, txt_content)

    log_local_file_json = log_local_file.replace(".txt", ".json")
    logging.info(f"[-] Saving logs to {log_local_file_json}")
    build_info = {
        "model_result": model_ok,
        "checker_result": checker_ok,
    }
    if build_metadata:
        build_info.update(build_metadata)
    if "project" not in build_info:
        build_info["project"] = proj_name
    auto_json = assemble_all_to_json(template, tools, question, output_text, intermediate_steps)
    atomic_write_json(
        log_local_file_json,
        {
            "autocompiler": auto_json,
            "tools": tools_logs,
            "build": build_info,
        },
        indent="\t",
    )

    # Crash-safe append-only JSONL ledger (1 line = 1 completed repo, locked + fsync)
    final_tail = ""
    if auto_json:
        last_pair = auto_json[-1]
        tail_str = " ".join(str(x) for x in last_pair) if isinstance(last_pair, list) else str(last_pair)
        final_tail = tail_str[-300:]
    else:
        final_tail = str(output_text or "")[-300:]

    jsonl_entry = {
        "schema_version": 1,
        "timestamp": now,
        "recorded_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "project": proj_name,
        "url": project_url or "",
        "log_file": log_local_file_json.replace("\\", "/"),
        "txt_log_file": log_local_file.replace("\\", "/"),
        "local_path": local_path,
        "final_output_tail": final_tail,
        "build": build_info,
    }
    append_jsonl(os.path.join(log_path, RESULTS_JSONL_FILENAME), jsonl_entry)

    # logging result into the current run's log_path so concurrent/different runs stay isolated
    log_url = LOG_URL_TEMPLATE.format(run_id=run_id) if LOG_URL_TEMPLATE else ""
    process_id = os.getpid()
    process_csv_file = os.path.join(log_path, f"{process_id}.csv")
    with open(process_csv_file, "a", encoding="utf-8", errors="replace", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([run_id, log_url, log_local_file, local_path, model_ok, checker_ok])
        file.flush()
        os.fsync(file.fileno())
