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

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from compile_agent.config import *
from compile_agent.agent.executor import CompilationAgentExecutor, SafeReActSingleInputOutputParser
from compile_agent.infra.downloader import copy_project, download_project
from compile_agent.reporting.logs import (
    find_compiled_artifacts,
    is_compiled,
    save_logs,
    smoke_check_artifact,
    snapshot_existing_artifacts,
)
from compile_agent.agent.error_solver import ErrorSolver
from compile_agent.agent.navigator import CompileNavigator
from compile_agent.security.audit import run_security_pipeline
from compile_agent.infra.docker_tools import InteractiveDockerShell

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)

os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

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


def _has_existing_log(log_path: str, proj_name: str, security_mode: str = "none") -> bool:
    pattern = os.path.join(log_path, f"{glob.escape(proj_name)}.log.*.json")
    sec_mode = (security_mode or "none").strip().lower()
    for fp in glob.glob(pattern):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not (isinstance(data, dict) and isinstance(data.get("build"), dict)):
                continue
            if sec_mode != "none":
                build = data.get("build") or {}
                compile_ok = (
                    bool(build.get("checker_non_strict"))
                    or str(build.get("checker_result", "")).startswith("True")
                    or str(build.get("model_result", "")) == "True"
                    or str(data.get("final_Decision", "")) == "COMPILATION-SUCCESS"
                )
                if compile_ok:
                    sec_obj = data.get("security")
                    has_inline_sec = (
                        isinstance(sec_obj, dict)
                        and bool(sec_obj)
                        and sec_obj.get("status") not in ("skipped_retry_pending", "error", "", None)
                        and not (sec_obj.get("mode") == "none" and sec_obj.get("status") == "skipped")
                    )
                    sec_file = os.path.join(log_path, "security", f"{proj_name}.security.json")
                    if not (has_inline_sec or os.path.exists(sec_file)):
                        continue
            return True
        except Exception:
            continue
    return False


def _append_error_jsonl(log_path: str, filename: str, entry: dict) -> None:
    os.makedirs(log_path, exist_ok=True)
    err_file = os.path.join(log_path, filename)
    with open(err_file, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry, ensure_ascii=False) + "\n")


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
    last_err = None
    for idx in range(max(1, int(max_attempts))):
        try:
            return InteractiveDockerShell(
                local_path=local_path,
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


def _cleanup_stale_process_csvs(csv_dir: str) -> None:
    if not csv_dir or not os.path.isdir(csv_dir):
        return
    for filename in os.listdir(csv_dir):
        if filename.endswith(".csv") and filename[:-4].isdigit():
            try:
                os.remove(os.path.join(csv_dir, filename))
            except OSError:
                pass


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
    security_mode,
    projects,
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

    for info in projects:
        proj_name = info["name"]
        if skip_existing and _has_existing_log(log_path, proj_name, security_mode=security_mode):
            logging.info("[-] Skipping %s (existing JSON log found in %s)", proj_name, log_path)
            continue

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
                                "[!] Transient LLM error on %s (%s); skipping save_logs/Stage 3 so --skip-existing will retry later.",
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
                            "security_mode": security_mode,
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

                        # Run the 3-layer security pipeline ONCE per project on the final attempt
                        # (either compilation passed, or retries are exhausted) while the container is open.
                        is_final_attempt = (attempt_idx == retry) or (not checker_ok.startswith("False"))
                        if is_final_attempt:
                            try:
                                security_report = run_security_pipeline(
                                    shell=shell,
                                    local_path=local_path,
                                    log_path=log_path,
                                    proj_name=proj_name,
                                    target_files=files,
                                    tools_log=tools_log,
                                    security_mode=security_mode,
                                    ignore_relpaths=pre_existing_artifacts,
                                )
                            except Exception as sec_err:
                                logging.exception(
                                    "[!] Security pipeline failed for %s: %s", proj_name, sec_err
                                )
                                security_report = {
                                    "mode": security_mode,
                                    "status": "error",
                                    "error": str(sec_err),
                                }
                        else:
                            security_report = {
                                "mode": security_mode,
                                "status": "skipped_retry_pending",
                                "attempt": attempt_idx + 1,
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
                            security_report,
                            build_metadata=build_metadata,
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
    "--projects-filter",
    type=str,
    default="",
    help="comma-separated list of project names to run from json_path",
)
@click.option(
    "--security-mode",
    type=click.Choice(["none", "sast", "hardening", "sanitizer", "all"], case_sensitive=False),
    default="all",
    show_default=True,
    help="Stage 3 security pipeline mode (none, sast, hardening, sanitizer, all)",
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
    projects_filter,
    security_mode,
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

    if not projects:
        logging.warning("[!] No matching projects found to compile.")
        return

    os.makedirs(log_path, exist_ok=True)
    _cleanup_stale_process_csvs(log_path)

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
        security_mode,
    )
    try:
        if multi_process:
            os.environ["AC_MULTI_WORKER"] = "1"
            worker_count = max(1, min(len(projects), max_workers or 2))
            args = [base_tuple + ([project],) for project in projects]
            with multiprocessing.Pool(processes=worker_count) as pool:
                pool.starmap(start_compile, args)
        else:
            args = base_tuple + (projects,)
            start_compile(*args)
    finally:
        # Merge temporary <pid>.csv inside log_path into statistics.csv and clean up temp files
        merge_data = []
        for filename in sorted(os.listdir(log_path)):
            if filename.endswith(".csv") and filename[:-4].isdigit():
                file_path = os.path.join(log_path, filename)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as fp:
                        csv_reader = csv.reader(fp)
                        csv_data = list(csv_reader)
                        merge_data.append(csv_data)
                finally:
                    try:
                        os.remove(file_path)
                    except OSError:
                        pass

        if merge_data:
            stats_path = os.path.join(log_path, "statistics.csv")
            with open(stats_path, "a", encoding="utf-8", errors="replace", newline="") as fp:
                csv_writer = csv.writer(fp)
                for data in merge_data:
                    for row in data:
                        csv_writer.writerow(row)


if __name__ == "__main__":
    main()
