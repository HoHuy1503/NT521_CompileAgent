import csv
import glob
import json
import logging
import multiprocessing
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import uuid

import click
from langchain.agents import Tool, create_react_agent
from langchain.prompts import PromptTemplate

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from compile_agent.config import *
from compile_agent.agent.executor import CompilationAgentExecutor, SafeReActSingleInputOutputParser
from compile_agent.infra.downloader import copy_project, download_project
from compile_agent.reporting.logs import (
    RESULTS_JSONL_FILENAME,
    append_jsonl,
    backfill_results_jsonl,
    cleanup_stale_tmp_files,
    find_compiled_artifacts,
    is_compiled,
    iter_jsonl_records,
    load_results_jsonl_index,
    save_logs,
    smoke_check_artifact,
    snapshot_existing_artifacts,
)
from compile_agent.agent.error_solver import ErrorSolver
from compile_agent.agent.navigator import CompileNavigator
from compile_agent.infra.docker_tools import InteractiveDockerShell

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)

prompt = PromptTemplate.from_template(template)
question_template = model_decision_template


def _force_rmtree(path: str):
    if not path or not os.path.exists(path):
        return

    def _on_error(func, func_path, _exc_info):
        try:
            os.chmod(func_path, stat.S_IWRITE | stat.S_IREAD)
        except OSError:
            pass
        try:
            func(func_path)
        except OSError:
            pass

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_on_error)
    else:
        shutil.rmtree(path, onerror=_on_error)
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


def parse_model_decision(res: dict) -> str:
    if not res:
        return "False"
    for candidate in (res.get("final_answer", ""), res.get("output", "")):
        text = strip_thinking_tags(candidate or "")
        matches = re.findall(r"\bCOMPILATION[-_](SUCCESS|FAIL|UNCERTAIN)\b", text, flags=re.IGNORECASE)
        if matches:
            verdict = matches[-1].upper()
            if verdict == "SUCCESS":
                return "True"
            if verdict == "UNCERTAIN":
                return "UNCERTAIN"
            return "False"
    combined_low = (
        str(res.get("final_answer", "")) + "\n" + str(res.get("output", ""))
    ).lower()
    if "agent stopped due to iteration limit or time limit" in combined_low:
        return "UNCERTAIN"
    return "False"


def _is_uncertain_or_loop_log(data: dict, build: dict) -> bool:
    if str(build.get("model_result", "")).upper() == "UNCERTAIN":
        return True
    tail_candidates = []
    if data.get("final_output_tail"):
        tail_candidates.append(str(data.get("final_output_tail")))
    auto = data.get("autocompiler")
    if isinstance(auto, list) and auto:
        last_pair = auto[-1]
        tail_candidates.append(
            " ".join(str(x) for x in last_pair) if isinstance(last_pair, list) else str(last_pair)
        )
    low_tail = " ".join(tail_candidates).lower()
    if (
        "early stop: loop detected" in low_tail
        or "terminated early due to repeated action/parsing loop" in low_tail
        or "agent stopped due to iteration limit or time limit" in low_tail
    ):
        return True
    return False


def _record_satisfies_existing(
    data: dict,
    retry_uncertain: bool,
) -> bool:
    if not (isinstance(data, dict) and isinstance(data.get("build"), dict)):
        return False
    build = data.get("build") or {}
    compile_ok = (
        bool(build.get("checker_non_strict"))
        or str(build.get("checker_result", "")).startswith("True")
        or str(build.get("model_result", "")) == "True"
        or str(data.get("final_Decision", "")) == "COMPILATION-SUCCESS"
    )
    if retry_uncertain and not compile_ok and _is_uncertain_or_loop_log(data, build):
        return False
    return True


def _has_existing_log(
    log_path: str,
    proj_name: str,
    retry_uncertain: bool = False,
    jsonl_index: dict = None,
) -> bool:
    index = jsonl_index if jsonl_index is not None else load_results_jsonl_index(log_path)
    for rec in index.get(proj_name, []):
        if _record_satisfies_existing(rec, retry_uncertain):
            return True

    pattern = os.path.join(log_path, f"{glob.escape(proj_name)}.log.*.json")
    for fp in sorted(glob.glob(pattern), reverse=True):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if _record_satisfies_existing(data, retry_uncertain):
                return True
        except Exception:
            continue
    return False


def _append_error_jsonl(log_path: str, filename: str, entry: dict) -> None:
    os.makedirs(log_path, exist_ok=True)
    err_file = os.path.join(log_path, filename)
    append_jsonl(err_file, entry)


def _record_download_error(log_path: str, proj_name: str, url: str, err: Exception) -> None:
    _append_error_jsonl(
        log_path,
        "download_errors.jsonl",
        {
            "project": proj_name,
            "url": url,
            "error": str(err),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


def _record_transient_llm_error(
    log_path: str, proj_name: str, url: str, err: Exception, attempt: int = 1
) -> None:
    _append_error_jsonl(
        log_path,
        "llm_transient_errors.jsonl",
        {
            "project": proj_name,
            "url": url,
            "attempt": attempt,
            "error": str(err),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


def _record_container_error(
    log_path: str, proj_name: str, url: str, err: Exception, attempt: int = 1
) -> None:
    _append_error_jsonl(
        log_path,
        "container_errors.jsonl",
        {
            "project": proj_name,
            "url": url,
            "attempt": attempt,
            "error": str(err),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


def _open_docker_shell_with_retry(
    local_path: str, max_attempts: int = 2
) -> InteractiveDockerShell:
    stuck_timeout = max(30, int(os.getenv("SHELL_STUCK_TIMEOUT", "240") or "240"))
    last_err = None
    for idx in range(max(1, int(max_attempts))):
        try:
            return InteractiveDockerShell(
                local_path=local_path,
                stuck_timeout=stuck_timeout,
                cmd_timeout=1200,
                use_proxy=USE_PROXY,
                pre_exec=USE_PROXY,
            )
        except Exception as err:
            last_err = err
            if idx + 1 < max_attempts:
                logging.warning(
                    "[!] InteractiveDockerShell init failed (%s); retrying once after 2s...",
                    err,
                )
                time.sleep(2.0)
    raise last_err


def _merge_process_csvs(csv_dir: str) -> int:
    """
    Merge any temporary <pid>.csv files inside csv_dir into statistics.csv (deduplicating by run_id)
    and delete the merged temp files. Safe to call both on startup crash-recovery and in finally.
    """
    if not csv_dir or not os.path.isdir(csv_dir):
        return 0
    stats_path = os.path.join(csv_dir, "statistics.csv")
    seen_run_ids = set()
    if os.path.isfile(stats_path):
        try:
            with open(stats_path, "r", encoding="utf-8", errors="replace", newline="") as fp:
                for row in csv.reader(fp):
                    if row and row[0]:
                        seen_run_ids.add(row[0])
        except OSError:
            pass

    rows_to_append = []
    for filename in sorted(os.listdir(csv_dir)):
        if filename.endswith(".csv") and filename[:-4].isdigit():
            file_path = os.path.join(csv_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as fp:
                    for row in csv.reader(fp):
                        if not row:
                            continue
                        run_id = row[0] if row[0] else None
                        if run_id and run_id in seen_run_ids:
                            continue
                        if run_id:
                            seen_run_ids.add(run_id)
                        rows_to_append.append(row)
            except OSError:
                continue
            finally:
                try:
                    os.remove(file_path)
                except OSError:
                    pass

    if rows_to_append:
        with open(stats_path, "a", encoding="utf-8", errors="replace", newline="") as fp:
            csv_writer = csv.writer(fp)
            for row in rows_to_append:
                csv_writer.writerow(row)
            fp.flush()
            os.fsync(fp.fileno())
    return len(rows_to_append)


_cleanup_stale_process_csvs = _merge_process_csvs


def start_compile(
    dataset_base_path,
    log_path,
    clean_copied_project,
    download_proxy,
    strict_checker,
    retry,
    disable_navigator,
    disable_error_solver,
    skip_existing,
    projects,
    retry_uncertain=False,
):
    DATASET_BASE_PATH = None
    if os.environ.get("DATASET_BASE_PATH"):
        DATASET_BASE_PATH = os.environ.get("DATASET_BASE_PATH")
    if dataset_base_path:
        DATASET_BASE_PATH = dataset_base_path
    if not DATASET_BASE_PATH:
        DATASET_BASE_PATH = os.path.join(_REPO_ROOT, "dataset")
    os.makedirs(DATASET_BASE_PATH, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    if not download_proxy and USE_PROXY and PROXY:
        download_proxy = PROXY

    if disable_navigator and disable_error_solver:
        ablation_mode = "baseline_no_nav_no_solver"
    elif disable_navigator:
        ablation_mode = "no_navigator"
    elif disable_error_solver:
        ablation_mode = "no_error_solver"
    else:
        ablation_mode = "full"

    consecutive_container_failures = 0
    total_projects = len(projects)
    jsonl_index = load_results_jsonl_index(log_path) if skip_existing else {}

    for proj_idx, info in enumerate(projects, start=1):
        proj_name = info["name"]
        if skip_existing and _has_existing_log(
            log_path,
            proj_name,
            retry_uncertain=retry_uncertain,
            jsonl_index=jsonl_index,
        ):
            logging.info(
                "[-] [%d/%d] Skipping %s (existing completed record found in %s)",
                proj_idx,
                total_projects,
                proj_name,
                log_path,
            )
            continue
        logging.info("[+] [%d/%d] Starting project: %s", proj_idx, total_projects, proj_name)

        for attempt_idx in range(retry + 1):
            url = info["url"]
            files = info.get("files", None)
            raw_proj_path = info.get("local_path")
            if raw_proj_path:
                if os.path.isabs(raw_proj_path) or os.path.exists(raw_proj_path):
                    proj_path = os.path.abspath(raw_proj_path)
                else:
                    proj_path = os.path.abspath(os.path.join(_REPO_ROOT, raw_proj_path))
            else:
                proj_path = os.path.join(DATASET_BASE_PATH, proj_name)
            local_path = None
            pre_existing_artifacts = set()
            try:
                if not os.path.exists(proj_path) or not os.listdir(proj_path):
                    download_project(url, proj_path, download_proxy)
                local_path = copy_project(proj_path, dest_parent_dir=DATASET_BASE_PATH)
                pre_existing_artifacts = snapshot_existing_artifacts(local_path)
                if pre_existing_artifacts:
                    logging.info(
                        "[%s] Snapshotted %d pre-existing binary/archive file(s) to ignore.",
                        proj_name,
                        len(pre_existing_artifacts),
                    )
            except Exception as dl_err:
                logging.error("[!] Failed to prepare project %s (%s): %s", proj_name, url, dl_err)
                _record_download_error(log_path, proj_name, url, dl_err)
                break

            os.environ["LOCAL_PATH"] = local_path
            checker_ok = f"False(strict={str(strict_checker)})"
            transient_llm_aborted = False
            container_aborted = False
            container_started = False

            try:
                llm = create_chat_llm()

                res = {
                    "final_answer": "",
                    "intermediate_steps": [],
                    "output": "",
                    "run_id": str(uuid.uuid4()),
                }
                checker_non_strict = False
                checker_strict = False
                question = build_decision_question(
                    proj_name,
                    disable_navigator=disable_navigator,
                    disable_error_solver=disable_error_solver,
                )
                smoke_check = {"checked_count": 0, "passed": False, "missing_libs": [], "details": []}
                artifacts_info = {"all_artifacts": [], "matched_targets": []}
                attempt_start = time.time()

                with _open_docker_shell_with_retry(local_path) as shell:
                    container_started = True
                    consecutive_container_failures = 0
                    GetIns = CompileNavigator(local_path=local_path, project_name=proj_name)
                    Dis = ErrorSolver(project_name=proj_name)
                    tools = [
                        Tool(
                            name="Shell",
                            description=shell.execute_command.__doc__,
                            func=shell.execute_command,
                        )
                    ]
                    if not disable_navigator:
                        tools.append(
                            Tool(
                                name="CompileNavigator",
                                description=GetIns.get_instructions.__doc__,
                                func=GetIns.get_instructions,
                            )
                        )
                    if not disable_error_solver:
                        tools.append(
                            Tool(
                                name="ErrorSolver",
                                description=Dis.discussion.__doc__,
                                func=Dis.discussion,
                            )
                        )
                    agent = create_react_agent(
                        llm=llm,
                        tools=tools,
                        prompt=prompt,
                        output_parser=SafeReActSingleInputOutputParser(),
                        stop_sequence=resolve_react_stop_sequences(),
                    )
                    agent_max_execution_time = max(
                        60, int(os.getenv("AGENT_MAX_EXECUTION_TIME", "1800") or "1800")
                    )
                    agent_executor = CompilationAgentExecutor(
                        agent=agent,
                        tools=tools,
                        verbose=True,
                        return_intermediate_steps=True,
                        max_iterations=30,
                        max_execution_time=agent_max_execution_time,
                        handle_parsing_errors=True,
                    )

                    shell_call_count = 0
                    try:
                        for step in agent_executor.iter({"input": question}, include_run_info=True):
                            if "intermediate_step" in step:
                                istep = step["intermediate_step"]
                                if isinstance(istep, list):
                                    res["intermediate_steps"].extend(istep)
                                else:
                                    res["intermediate_steps"].append(istep)
                                continue
                            if "output" in step:
                                messages = step.get("messages", [])
                                msg_content = messages[0].content if messages else ""
                                run_info = step.get("__run")
                                run_id = str(run_info.run_id) if run_info else res["run_id"]
                                final_ans = step.get("output", "")
                                res = {
                                    "final_answer": final_ans,
                                    "intermediate_steps": (
                                        step.get("intermediate_steps") or res["intermediate_steps"]
                                    ),
                                    "output": msg_content or final_ans,
                                    "run_id": run_id,
                                }
                        shell_call_count = len(shell.logger)
                        artifacts_info = find_compiled_artifacts(
                            local_path, files, shell=shell, ignore_relpaths=pre_existing_artifacts
                        )
                        checker_non_strict = bool(
                            is_compiled(local_path, files, strict=False, precomputed_found=artifacts_info)
                        )
                        checker_strict = bool(
                            is_compiled(local_path, files, strict=True, precomputed_found=artifacts_info)
                        )
                        checker_res = checker_strict if strict_checker else checker_non_strict
                        checker_ok = f"{str(checker_res)}(strict={str(strict_checker)})"
                        if artifacts_info["all_artifacts"]:
                            smoke_check = smoke_check_artifact(
                                shell,
                                local_path,
                                files,
                                precomputed_found=artifacts_info,
                                ignore_relpaths=pre_existing_artifacts,
                            )
                        record_repo_llm_success()
                    except Exception as e:
                        if is_fatal_llm_error(e):
                            logging.critical(
                                "[FATAL] Unrecoverable LLM/API error on %s: %s. Stopping batch without writing fake failure log.",
                                proj_name,
                                e,
                            )
                            raise
                        if is_transient_llm_error(e):
                            try:
                                record_repo_llm_failure(e)
                            except FatalLLMError as cb_err:
                                logging.critical(
                                    "[FATAL] Circuit breaker tripped on %s: %s. Stopping batch without writing fake failure log.",
                                    proj_name,
                                    cb_err,
                                )
                                raise
                            transient_llm_aborted = True
                            logging.warning(
                                "[!] Transient LLM error on %s (%s); skipping save_logs so --skip-existing will retry later.",
                                proj_name,
                                e,
                            )
                            _record_transient_llm_error(
                                log_path, proj_name, url, e, attempt=attempt_idx + 1
                            )
                        else:
                            shell_call_count = len(shell.logger)
                            logging.exception(f"[!] Compilation failed with error: {e}")
                            checker_ok = f"False(strict={str(strict_checker)})"

                    if not transient_llm_aborted:
                        elapsed_seconds = round(time.time() - attempt_start, 2)
                        model_ok = parse_model_decision(res)

                        logging.info(
                            "[-] Project %s [%s], model result: %s, checker result: %s, non-strict=%s, strict=%s, smoke=%s, time=%.2fs",
                            proj_name,
                            ablation_mode,
                            model_ok,
                            checker_ok,
                            checker_non_strict,
                            checker_strict,
                            smoke_check.get("passed", False),
                            elapsed_seconds,
                        )

                        tools_log = {
                            "Shell": list(shell.logger[:shell_call_count]),
                            "CompileNavigator": GetIns.logger,
                            "ErrorSolver": Dis.logger,
                        }
                        build_metadata = {
                            "project": proj_name,
                            "ablation_mode": ablation_mode,
                            "strict_checker": bool(strict_checker),
                            "checker_non_strict": checker_non_strict,
                            "checker_strict": checker_strict,
                            "elapsed_seconds": elapsed_seconds,
                            "iterations": len(res.get("intermediate_steps", [])),
                            "tool_calls": {
                                "Shell": shell_call_count,
                                "CompileNavigator": len(GetIns.logger),
                                "ErrorSolver": len(Dis.logger),
                            },
                            "matched_targets": artifacts_info["matched_targets"],
                            "all_artifacts": artifacts_info["all_artifacts"][:50],
                            "smoke_check": smoke_check,
                        }

                        save_logs(
                            log_path,
                            proj_name,
                            question,
                            tools,
                            res,
                            tools_log,
                            local_path,
                            checker_ok,
                            model_ok,
                            build_metadata=build_metadata,
                            project_url=url,
                        )
            except Exception as repo_err:
                if is_fatal_llm_error(repo_err):
                    raise
                if is_transient_llm_error(repo_err):
                    try:
                        record_repo_llm_failure(repo_err)
                    except FatalLLMError as cb_err:
                        logging.critical(
                            "[FATAL] Circuit breaker tripped on %s: %s. Stopping batch without writing fake failure log.",
                            proj_name,
                            cb_err,
                        )
                        raise
                    transient_llm_aborted = True
                    logging.warning(
                        "[!] Transient LLM error before/outside agent loop on %s (%s); skipping fake log so --skip-existing will retry later.",
                        proj_name,
                        repo_err,
                    )
                    _record_transient_llm_error(
                        log_path, proj_name, url, repo_err, attempt=attempt_idx + 1
                    )
                else:
                    if not container_started:
                        consecutive_container_failures += 1
                    logging.exception(
                        "[!] Container/runtime error for %s (attempt %d/%d): %s",
                        proj_name,
                        attempt_idx + 1,
                        retry + 1,
                        repo_err,
                    )
                    _record_container_error(
                        log_path, proj_name, url, repo_err, attempt=attempt_idx + 1
                    )
                    container_aborted = True
                    checker_ok = f"False(strict={str(strict_checker)})"
                    if consecutive_container_failures >= 3:
                        docker_ok = False
                        try:
                            info_proc = subprocess.run(
                                ["docker", "info"],
                                capture_output=True,
                                text=True,
                                timeout=15,
                                stdin=subprocess.DEVNULL,
                            )
                            docker_ok = info_proc.returncode == 0
                        except Exception:
                            docker_ok = False
                        if not docker_ok:
                            raise RuntimeError(
                                "Docker daemon appears offline after 3 consecutive container failures; aborting batch."
                            ) from repo_err
            finally:
                if clean_copied_project and local_path and os.path.exists(local_path):
                    logging.info(f"[+] Clear project compilation path {local_path}")
                    _force_rmtree(local_path)

            if transient_llm_aborted or container_aborted:
                break
            if checker_ok.startswith("False"):
                continue
            else:
                break


@click.command()
@click.option(
    "-j",
    "--json_path",
    type=str,
    required=True,
    help="the path to the project file to be compiled",
)
@click.option(
    "-p",
    "--dataset_base_path",
    type=str,
    required=True,
    help="the base path to store the projects need to compile",
)
@click.option(
    "-l",
    "--log_path",
    type=str,
    required=True,
    help="the path to store the logs of compilation",
)
@click.option(
    "-c",
    "--clean_copied_project",
    type=bool,
    required=True,
    help="whether to clear the new project path after the compilation, set to True when debugging",
)
@click.option(
    "-r",
    "--retry",
    type=int,
    default=0,
    help="the times to retry when fail to pass checker",
)
@click.option(
    "-s",
    "--strict_checker",
    type=bool,
    default=False,
    help="if True, only return True when all target files are in the file list, otherwise return True when any target file is in the file list",
)
@click.option(
    "--download_proxy",
    type=str,
    default=None,
    help="pass to git --config http.proxy=${}, e.g., socks5://127.0.0.1:29999, if do not set: do not use proxy",
)
@click.option(
    "--multi_process",
    type=bool,
    default=False,
    help="whether to enable the multi-process to compile the projects",
)
@click.option(
    "--max-workers",
    type=int,
    default=2,
    show_default=True,
    help="maximum concurrent worker processes when --multi_process is True",
)
@click.option(
    "--disable-navigator",
    is_flag=True,
    default=False,
    help="disable CompileNavigator tool for ablation experiments",
)
@click.option(
    "--disable-error-solver",
    is_flag=True,
    default=False,
    help="disable ErrorSolver tool for ablation experiments",
)
@click.option(
    "--skip-existing",
    is_flag=True,
    default=False,
    help="skip projects that already have a valid JSON log in log_path",
)
@click.option(
    "--retry-uncertain",
    is_flag=True,
    default=False,
    help="when combined with --skip-existing, re-run projects whose prior log ended in UNCERTAIN / loop early-stop without passing checker",
)
@click.option(
    "--start-index",
    type=int,
    default=1,
    show_default=True,
    help="1-based starting index into json_path (applied before --limit)",
)
@click.option(
    "--limit",
    type=int,
    default=0,
    show_default=True,
    help="maximum number of projects to run from the selected slice (0 = all)",
)
@click.option(
    "--projects-filter",
    type=str,
    default="",
    help="comma-separated list of project names to run from json_path",
)
def main(
    json_path,
    dataset_base_path,
    log_path,
    clean_copied_project,
    download_proxy,
    strict_checker,
    retry,
    multi_process,
    max_workers,
    disable_navigator,
    disable_error_solver,
    skip_existing,
    retry_uncertain,
    start_index,
    limit,
    projects_filter,
):
    """
    Compile the projects from source code with the help of AI assistant
    """
    with open(json_path, "r", encoding="utf-8") as fp:
        projects = json.load(fp)

    if projects_filter.strip():
        wanted_raw = [name.strip() for name in projects_filter.split(",") if name.strip()]
        wanted_lower = {name.lower() for name in wanted_raw}
        matched_lower = set()
        filtered = []
        for p in projects:
            pname = str(p.get("name", ""))
            if pname.lower() in wanted_lower:
                filtered.append(p)
                matched_lower.add(pname.lower())
        unmatched = [name for name in wanted_raw if name.lower() not in matched_lower]
        if unmatched:
            logging.warning(
                "[!] Requested project(s) not found in %s: %s",
                json_path,
                ", ".join(unmatched),
            )
        projects = filtered

    if start_index > 1:
        projects = projects[start_index - 1 :]
    if limit and limit > 0:
        projects = projects[:limit]

    if not projects:
        logging.warning("[!] No matching projects found to compile.")
        return

    os.makedirs(log_path, exist_ok=True)
    removed_tmps = cleanup_stale_tmp_files(log_path)
    if removed_tmps:
        logging.info("[+] Cleaned up %d interrupted *.tmp.* file(s) in %s.", removed_tmps, log_path)
    recovered_csv_rows = _merge_process_csvs(log_path)
    if recovered_csv_rows:
        logging.info(
            "[+] Recovered %d unmerged statistics.csv row(s) from prior interrupted run in %s.",
            recovered_csv_rows,
            log_path,
        )

    # Repair any truncated trailing JSONL lines from a prior power cut and sync legacy logs
    for jsonl_name in (
        RESULTS_JSONL_FILENAME,
        "download_errors.jsonl",
        "llm_transient_errors.jsonl",
        "container_errors.jsonl",
    ):
        iter_jsonl_records(os.path.join(log_path, jsonl_name), repair_truncated_tail=True)
    backfilled = backfill_results_jsonl(log_path)
    if backfilled:
        logging.info(
            "[+] Backfilled %d existing *.log.*.json record(s) into %s.",
            backfilled,
            os.path.join(log_path, RESULTS_JSONL_FILENAME),
        )

    if not skip_existing:
        existing_logs = glob.glob(os.path.join(log_path, "*.log.*.json"))
        existing_jsonl = iter_jsonl_records(os.path.join(log_path, RESULTS_JSONL_FILENAME))
        existing_count = max(len(existing_logs), len(existing_jsonl))
        if existing_count:
            logging.warning(
                "[!] Found %d existing log record(s) in %s, but --skip-existing is False; "
                "starting from project #1 of the selected list (%s). Pass --skip-existing to resume.",
                existing_count,
                log_path,
                projects[0].get("name", "?"),
            )

    base_tuple = (
        dataset_base_path,
        log_path,
        clean_copied_project,
        download_proxy,
        strict_checker,
        retry,
        disable_navigator,
        disable_error_solver,
        skip_existing,
    )
    try:
        if multi_process:
            os.environ["AC_MULTI_WORKER"] = "1"
            worker_count = max(1, min(len(projects), max_workers or 2))
            args = [base_tuple + ([project], retry_uncertain) for project in projects]
            with multiprocessing.Pool(processes=worker_count) as pool:
                pool.starmap(start_compile, args)
        else:
            args = base_tuple + (projects, retry_uncertain)
            start_compile(*args)
    finally:
        # Merge temporary <pid>.csv inside log_path into statistics.csv and clean up temp files
        _merge_process_csvs(log_path)


if __name__ == "__main__":
    main()
