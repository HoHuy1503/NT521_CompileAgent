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

from Config import *
from CustomAgentExecutor import CompilationAgentExecutor, SafeReActSingleInputOutputParser
from DownloadProject import copy_project, download_project
from Logs import find_compiled_artifacts, is_compiled, save_logs, smoke_check_artifact
from MultiAgentDiscussion import ErrorSolver
from MultiAgentGetInstructions import CompileNavigator
from Tools import InteractiveDockerShell

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


def run_semgrep(project_path, log_path, project_name):
    os.makedirs(log_path, exist_ok=True)
    report_path = os.path.join(log_path, f"{project_name}-semgrep.json")

    if shutil.which("semgrep") is None:
        logging.warning("[!] Semgrep CLI not found on host; skipping Semgrep scan.")
        return {"status": "skipped", "reason": "semgrep CLI not found"}

    try:
        result = subprocess.run(
            [
                "semgrep",
                "scan",
                "--config",
                "p/security-audit",
                "--json",
                "--output",
                report_path,
                project_path,
            ],
            capture_output=True,
            text=True,
        )
    except OSError as err:
        logging.warning("Semgrep could not be executed: %s", err)
        return {"status": "skipped", "reason": str(err)}

    if result.returncode not in (0, 1):
        logging.error("Semgrep failed: %s", result.stderr or result.stdout)
        return None

    try:
        with open(report_path, encoding="utf-8") as report_file:
            report = json.load(report_file)
        findings = report.get("results", [])
        summary = {
            "status": "completed",
            "report_path": report_path,
            "finding_count": len(findings),
            "severity_counts": {},
        }

        for finding in findings:
            severity = finding.get("extra", {}).get("severity", "UNKNOWN")
            summary["severity_counts"][severity] = (
                summary["severity_counts"].get(severity, 0) + 1
            )

        logging.info(
            "Semgrep scanned %s: %d findings; report: %s",
            project_name,
            summary["finding_count"],
            report_path,
        )
        return summary
    except (OSError, json.JSONDecodeError) as error:
        logging.error("Could not read Semgrep report: %s", error)
        return None


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
        matches = re.findall(r"\bCOMPILATION-(SUCCESS|FAIL|UNCERTAIN)\b", text)
        if matches:
            verdict = matches[-1]
            if verdict == "SUCCESS":
                return "True"
            if verdict == "UNCERTAIN":
                return "UNCERTAIN"
            return "False"
    return "False"


def _has_existing_log(log_path: str, proj_name: str) -> bool:
    pattern = os.path.join(log_path, f"{proj_name}.log.*.json")
    for fp in glob.glob(pattern):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("build"), dict):
                return True
        except Exception:
            continue
    return False


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
):
    DATASET_BASE_PATH = None
    if os.environ.get("DATASET_BASE_PATH"):
        DATASET_BASE_PATH = os.environ.get("DATASET_BASE_PATH")
    if dataset_base_path:
        DATASET_BASE_PATH = dataset_base_path
    if not DATASET_BASE_PATH:
        DATASET_BASE_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "dataset")
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

    for info in projects:
        proj_name = info["name"]
        if skip_existing and _has_existing_log(log_path, proj_name):
            logging.info("[-] Skipping %s (existing JSON log found in %s)", proj_name, log_path)
            continue

        for _ in range(retry + 1):
            url = info["url"]
            files = info.get("files", None)
            proj_path = info.get("local_path", os.path.join(DATASET_BASE_PATH, proj_name))
            if not os.path.exists(proj_path) or not os.listdir(proj_path):
                download_project(url, proj_path, download_proxy)

            local_path = copy_project(proj_path)
            os.environ["LOCAL_PATH"] = local_path

            llm = create_chat_llm()

            res = {
                "final_answer": "",
                "intermediate_steps": [],
                "output": "",
                "run_id": str(uuid.uuid4()),
            }
            checker_non_strict = False
            checker_strict = False
            checker_ok = f"False(strict={str(strict_checker)})"
            question = build_decision_question(
                proj_name,
                disable_navigator=disable_navigator,
                disable_error_solver=disable_error_solver,
            )
            smoke_check = {"checked_count": 0, "passed": False, "missing_libs": [], "details": []}
            artifacts_info = {"all_artifacts": [], "matched_targets": []}
            attempt_start = time.time()

            with InteractiveDockerShell(
                local_path=local_path,
                cmd_timeout=1200,
                use_proxy=USE_PROXY,
                pre_exec=USE_PROXY,
            ) as shell:
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
                agent_executor = CompilationAgentExecutor(
                    agent=agent,
                    tools=tools,
                    verbose=True,
                    return_intermediate_steps=True,
                    max_iterations=30,
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
                    artifacts_info = find_compiled_artifacts(local_path, files, shell=shell)
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
                            shell, local_path, files, precomputed_found=artifacts_info
                        )
                except Exception as e:
                    shell_call_count = len(shell.logger)
                    logging.exception(f"[!] Compilation failed with error: {e}")
                    checker_ok = f"False(strict={str(strict_checker)})"

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
                semgrep_report = run_semgrep(proj_path, log_path, proj_name)
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
                    semgrep_report,
                    build_metadata=build_metadata,
                )

            if clean_copied_project and os.path.exists(local_path):
                logging.info(f"[+] Clear project compilation path {local_path}")
                _force_rmtree(local_path)

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
def main(
    json_path,
    dataset_base_path,
    log_path,
    clean_copied_project,
    download_proxy,
    strict_checker,
    retry,
    multi_process,
    disable_navigator,
    disable_error_solver,
    skip_existing,
    projects_filter,
):
    """
    Compile the projects from source code with the help of AI assistant
    """
    with open(json_path, "r", encoding="utf-8") as fp:
        projects = json.load(fp)

    if projects_filter.strip():
        wanted = {name.strip() for name in projects_filter.split(",") if name.strip()}
        projects = [p for p in projects if p.get("name") in wanted]

    if not projects:
        logging.warning("[!] No matching projects found to compile.")
        return

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
    if multi_process:
        args = [base_tuple + ([project],) for project in projects]
        with multiprocessing.Pool(processes=len(projects)) as pool:
            pool.starmap(start_compile, args)
    else:
        args = base_tuple + (projects,)
        start_compile(*args)

    os.makedirs(PROCESS_LOG_CSV_PATH, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # Merge temporary <pid>.csv into statistics.csv and clean up temp files
    merge_data = []
    for filename in os.listdir(PROCESS_LOG_CSV_PATH):
        if filename.endswith(".csv") and filename[:-4].isdigit():
            file_path = os.path.join(PROCESS_LOG_CSV_PATH, filename)
            with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as fp:
                csv_reader = csv.reader(fp)
                csv_data = list(csv_reader)
                merge_data.append(csv_data)
            os.remove(file_path)

    if merge_data:
        stats_path = os.path.join(log_path, "statistics.csv")
        with open(stats_path, "a", encoding="utf-8", errors="replace", newline="") as fp:
            csv_writer = csv.writer(fp)
            for data in merge_data:
                for row in data:
                    csv_writer.writerow(row)


if __name__ == "__main__":
    main()
