import base64
import csv
import datetime
import json
import logging
import os

from Config import *


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


def find_compiled_artifacts(local_path, target_files=None, shell=None):
    target_names = sorted({os.path.basename(t) for t in (target_files or [])})
    if shell is not None:
        try:
            payload_src = (
                "import os, json\n"
                f"targets = set({json.dumps(target_names)})\n"
                "matched, non_obj, objs = [], [], []\n"
                "for r, ds, fs in os.walk('/work'):\n"
                "    ds[:] = [d for d in ds if d != '.git']\n"
                "    for fn in fs:\n"
                "        p = os.path.join(r, fn)\n"
                "        try:\n"
                "            if os.path.isfile(p) and os.path.getsize(p) > 0:\n"
                "                with open(p, 'rb') as f:\n"
                "                    h = f.read(8)\n"
                "                if h[:4] == b'\\x7fELF' or h in (b'!<arch>\\n', b'!<thin>\\n'):\n"
                "                    rel = os.path.relpath(p, '/work').replace('\\\\', '/')\n"
                "                    if fn in targets:\n"
                "                        matched.append(rel)\n"
                "                    if not fn.endswith('.o'):\n"
                "                        non_obj.append(rel)\n"
                "                    else:\n"
                "                        objs.append(rel)\n"
                "        except OSError:\n"
                "            pass\n"
                "all_a = sorted(set(matched + non_obj))[:50] or sorted(objs)[:50]\n"
                "print('__AC_ART__' + json.dumps({'matched_targets': sorted(matched), 'all_artifacts': all_a}))\n"
            )
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

    all_artifacts = []
    matched_targets = []
    target_set = set(target_names)
    for root, dirs, files in os.walk(local_path):
        dirs[:] = [d for d in dirs if d != ".git"]
        for file in files:
            fp = os.path.join(root, file)
            if is_elf(fp) or is_archive(fp):
                rel = os.path.relpath(fp, local_path).replace("\\", "/")
                all_artifacts.append(rel)
                if file in target_set:
                    matched_targets.append(rel)
    return {"all_artifacts": sorted(all_artifacts)[:50], "matched_targets": sorted(matched_targets)}


def is_compiled(local_path, target_files, strict=False, shell=None, precomputed_found=None):
    if not target_files:
        return None
    found = precomputed_found if precomputed_found is not None else find_compiled_artifacts(local_path, target_files, shell=shell)
    matched_names = {os.path.basename(p) for p in found["matched_targets"]}
    target_basenames = [os.path.basename(f) for f in target_files]
    if strict:
        return all(name in matched_names for name in target_basenames)
    return any(name in matched_names for name in target_basenames)


def smoke_check_artifact(shell, local_path, target_files=None, precomputed_found=None):
    found = precomputed_found if precomputed_found is not None else find_compiled_artifacts(local_path, target_files, shell=shell)
    candidates = found["matched_targets"] if found["matched_targets"] else found["all_artifacts"][:5]
    details = []
    missing_libs = []
    for rel in candidates[:5]:
        container_fp = f"/work/{rel}"
        file_out = ""
        ldd_out = ""
        try:
            file_out = shell.execute_command(f'file -L "{container_fp}"', record_log=False)
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

        is_dynamic_candidate = "ELF" in file_out and not rel.endswith((".a", ".o"))
        has_missing_lib = False
        if is_dynamic_candidate:
            try:
                ldd_out = shell.execute_command(f'ldd "{container_fp}" 2>&1 || true', record_log=False)
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
        "passed": bool(details) and all(d["ok"] for d in details),
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
    os.makedirs(PROCESS_LOG_CSV_PATH, exist_ok=True)

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

    # logging result
    log_url = LOG_URL_TEMPLATE.format(run_id=run_id) if LOG_URL_TEMPLATE else ""
    process_id = os.getpid()
    process_csv_file = os.path.join(PROCESS_LOG_CSV_PATH, f"{process_id}.csv")
    with open(process_csv_file, "a", encoding="utf-8", errors="replace", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([run_id, log_url, log_local_file, local_path, model_ok, checker_ok])
