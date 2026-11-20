import csv
import datetime
import glob
import io
import json
import os
import re
import sys
from statistics import mean

import click

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from compile_agent.reporting.logs import (
    RESULTS_JSONL_FILENAME,
    atomic_write_json,
    atomic_write_text,
    iter_jsonl_records,
)

LOG_NAME_RE = re.compile(r"^(?P<project>.+?)\.log\.(?P<ts>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d+)\.json$")


def _parse_boolish(val) -> bool:
    if isinstance(val, bool):
        return val
    s = str(val or "").strip()
    return s.startswith("True")


def _classify_diagnosis(rec: dict) -> str:
    checker_ns = rec["checker_non_strict"]
    model_ok = rec["model_result"] == "True"
    solver_calls = rec["error_solver_calls"]
    if checker_ns and model_ok:
        return "Success (Recovered w/ ErrorSolver)" if solver_calls > 0 else "Success"
    if checker_ns and not model_ok:
        return "False Negative (Model!=True / Checker=True)"
    if not checker_ns and model_ok:
        return "False Positive (Model=True / Checker=False)"
    if solver_calls > 0:
        return "Failed (After ErrorSolver)"
    return "Compilation Failed"


def _build_eval_record(
    data: dict,
    fallback_proj: str,
    ts_key: str,
    log_file: str,
    default_mode: str,
) -> dict:
    build = data["build"]
    tools = data.get("tools", {}) or {}
    proj = build.get("project") or data.get("project") or fallback_proj
    mode = build.get("ablation_mode") or default_mode
    tool_calls = build.get("tool_calls") or {}
    shell_calls = int(tool_calls.get("Shell", len(tools.get("Shell", []))))
    nav_calls = int(tool_calls.get("CompileNavigator", len(tools.get("CompileNavigator", []))))
    solver_calls = int(tool_calls.get("ErrorSolver", len(tools.get("ErrorSolver", []))))

    smoke = build.get("smoke_check") or {}
    checker_raw = str(build.get("checker_result", ""))
    checker_ns = (
        bool(build["checker_non_strict"])
        if "checker_non_strict" in build
        else _parse_boolish(checker_raw)
    )
    checker_strict = (
        bool(build["checker_strict"])
        if "checker_strict" in build
        else (_parse_boolish(checker_raw) if "strict=True" in checker_raw else checker_ns)
    )
    matched = build.get("matched_targets") or []
    auto_msgs = data.get("autocompiler") or []
    iterations = int(build.get("iterations", max(0, len(auto_msgs) - 1)))

    rec = {
        "project": proj,
        "mode": mode,
        "timestamp": ts_key,
        "log_file": str(log_file or "").replace("\\", "/"),
        "model_result": str(build.get("model_result", "False")),
        "checker_non_strict": checker_ns,
        "checker_strict": checker_strict,
        "smoke_passed": bool(smoke.get("passed", False)),
        "smoke_checked": int(smoke.get("checked_count", 0)),
        "missing_libs": "; ".join(smoke.get("missing_libs", [])),
        "matched_targets": ", ".join(matched),
        "iterations": iterations,
        "elapsed_seconds": float(build.get("elapsed_seconds", 0.0)),
        "shell_calls": shell_calls,
        "navigator_calls": nav_calls,
        "error_solver_calls": solver_calls,
    }
    rec["diagnosis"] = _classify_diagnosis(rec)
    return rec


def load_dedup_records(log_dir: str, default_mode: str) -> list:
    if not log_dir or not os.path.isdir(log_dir):
        return []
    latest = {}

    # 1. Load from crash-safe results.jsonl ledger (supports JSONL-only runs or resumed batches)
    jsonl_path = os.path.join(log_dir, RESULTS_JSONL_FILENAME)
    for row in iter_jsonl_records(jsonl_path, repair_truncated_tail=False):
        if not isinstance(row.get("build"), dict):
            continue
        proj = str(row.get("project") or row["build"].get("project") or "").strip()
        if not proj:
            continue
        ts_key = str(row.get("timestamp") or "")
        log_fp = str(row.get("log_file") or jsonl_path)
        rec = _build_eval_record(row, proj, ts_key, log_fp, default_mode)
        key = (rec["project"], rec["mode"])
        if key not in latest or rec["timestamp"] >= latest[key]["timestamp"]:
            latest[key] = rec

    # 2. Load per-project *.log.*.json files (merges seamlessly & deduplicates by newest timestamp)
    for fp in glob.glob(os.path.join(log_dir, "*.log.*.json")):
        fname = os.path.basename(fp)
        m = LOG_NAME_RE.match(fname)
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

        rec = _build_eval_record(data, fallback_proj, ts_key, fp, default_mode)
        key = (rec["project"], rec["mode"])
        if key not in latest or rec["timestamp"] > latest[key]["timestamp"]:
            latest[key] = rec

    return sorted(latest.values(), key=lambda r: (r["mode"], r["project"].lower()))


def compute_summary(records: list) -> dict:
    n = len(records)
    if n == 0:
        return {"count": 0}
    model_true = sum(1 for r in records if r["model_result"] == "True")
    ns_true = sum(1 for r in records if r["checker_non_strict"])
    strict_true = sum(1 for r in records if r["checker_strict"])
    smoke_true = sum(
        1
        for r in records
        if r["smoke_passed"] and (r["checker_non_strict"] or r.get("matched_targets"))
    )
    fp_cnt = sum(1 for r in records if r["model_result"] == "True" and not r["checker_non_strict"])
    fn_cnt = sum(1 for r in records if r["model_result"] != "True" and r["checker_non_strict"])
    return {
        "count": n,
        "model_true": model_true,
        "model_rate": round(100.0 * model_true / n, 1),
        "csr_non_strict_count": ns_true,
        "csr_non_strict": round(100.0 * ns_true / n, 1),
        "csr_strict_count": strict_true,
        "csr_strict": round(100.0 * strict_true / n, 1),
        "smoke_pass_count": smoke_true,
        "smoke_pass_rate": round(100.0 * smoke_true / n, 1),
        "false_positives": fp_cnt,
        "false_negatives": fn_cnt,
        "avg_time_s": round(mean(r["elapsed_seconds"] for r in records), 2),
        "total_time_s": round(sum(r["elapsed_seconds"] for r in records), 2),
        "avg_iterations": round(mean(r["iterations"] for r in records), 2),
        "avg_shell_calls": round(mean(r["shell_calls"] for r in records), 2),
        "avg_navigator_calls": round(mean(r["navigator_calls"] for r in records), 2),
        "avg_error_solver_calls": round(mean(r["error_solver_calls"] for r in records), 2),
        "total_navigator_calls": sum(r["navigator_calls"] for r in records),
        "total_error_solver_calls": sum(r["error_solver_calls"] for r in records),
    }


def write_csv(path: str, rows: list, fieldnames: list) -> None:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fieldnames})
    atomic_write_text(path, buf.getvalue())


def build_markdown_report(full_records: list, ablation_records: list) -> tuple:
    full_sum = compute_summary(full_records)
    lines = ["# CompileAgent Benchmark & Ablation Evaluation Report\n"]

    if full_records:
        lines.append("## 1. Full CompileAgent Benchmark Results\n")
        lines.append(
            "| Project | Model Result | Checker (Non-Strict) | Checker (Strict) | Smoke Check (`file`/`ldd`) | Matched Artifacts | Iterations | Time (s) | Shell | Navigator | ErrorSolver | Diagnosis |"
        )
        lines.append("| :--- | :-: | :-: | :-: | :-: | :--- | :-: | :-: | :-: | :-: | :-: | :--- |")
        for r in full_records:
            smoke_str = f"{'PASS' if r['smoke_passed'] else 'FAIL'} ({r['smoke_checked']})"
            lines.append(
                f"| `{r['project']}` | {r['model_result']} | {r['checker_non_strict']} | {r['checker_strict']} | {smoke_str} | `{r['matched_targets'] or '-'}` | {r['iterations']} | {r['elapsed_seconds']:.2f} | {r['shell_calls']} | {r['navigator_calls']} | {r['error_solver_calls']} | {r['diagnosis']} |"
            )

        lines.append("\n### Aggregate Metrics (Full CompileAgent)\n")
        lines.append("| Metric | Value |")
        lines.append("| :--- | :--- |")
        lines.append(f"| Total Projects | {full_sum['count']} |")
        lines.append(f"| Model Self-Reported Success Rate | {full_sum['model_true']}/{full_sum['count']} ({full_sum['model_rate']}%) |")
        lines.append(f"| Checker CSR (Non-Strict, paper metric) | {full_sum['csr_non_strict_count']}/{full_sum['count']} ({full_sum['csr_non_strict']}%) |")
        lines.append(f"| Checker CSR (Strict) | {full_sum['csr_strict_count']}/{full_sum['count']} ({full_sum['csr_strict']}%) |")
        lines.append(f"| Smoke Check Pass Rate (`file` & `ldd`) | {full_sum['smoke_pass_count']}/{full_sum['count']} ({full_sum['smoke_pass_rate']}%) |")
        lines.append(f"| False Positives (`Model=True, Checker=False`) | {full_sum['false_positives']} |")
        lines.append(f"| False Negatives (`Model!=True, Checker=True`) | {full_sum['false_negatives']} |")
        lines.append(f"| Average Compile Time / Total Time | {full_sum['avg_time_s']}s / {full_sum['total_time_s']}s |")
        lines.append(f"| Average Iterations | {full_sum['avg_iterations']} |")
        lines.append(
            f"| Avg Calls (Shell / CompileNavigator / ErrorSolver) | {full_sum['avg_shell_calls']} / {full_sum['avg_navigator_calls']} / {full_sum['avg_error_solver_calls']} (Total Nav={full_sum['total_navigator_calls']}, Solver={full_sum['total_error_solver_calls']}) |"
        )

    ablation_rows = []
    if ablation_records:
        full_by_proj = {r["project"]: r for r in full_records}
        paired_full = []
        for ab in ablation_records:
            f_rec = full_by_proj.get(ab["project"])
            if f_rec:
                paired_full.append(f_rec)
            ablation_rows.append(
                {
                    "project": ab["project"],
                    "full_checker_non_strict": f_rec["checker_non_strict"] if f_rec else "N/A",
                    "full_checker_strict": f_rec["checker_strict"] if f_rec else "N/A",
                    "full_time_s": f_rec["elapsed_seconds"] if f_rec else "N/A",
                    "full_iterations": f_rec["iterations"] if f_rec else "N/A",
                    "full_nav_calls": f_rec["navigator_calls"] if f_rec else "N/A",
                    "full_solver_calls": f_rec["error_solver_calls"] if f_rec else "N/A",
                    "baseline_checker_non_strict": ab["checker_non_strict"],
                    "baseline_checker_strict": ab["checker_strict"],
                    "baseline_model_result": ab["model_result"],
                    "baseline_time_s": ab["elapsed_seconds"],
                    "baseline_iterations": ab["iterations"],
                    "baseline_shell_calls": ab["shell_calls"],
                }
            )

        sub_full_sum = compute_summary(paired_full)
        ab_sum = compute_summary(ablation_records)

        lines.append("\n## 2. Ablation Study (`Full CompileAgent` vs `Baseline: --disable-navigator --disable-error-solver`)\n")
        lines.append(
            "| Project | Full CSR (Non-Strict / Strict) | Full Iter / Time (s) | Full Nav / Solver Calls | Baseline CSR (Non-Strict / Strict) | Baseline Model | Baseline Iter / Time (s) | Baseline Shell |"
        )
        lines.append("| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |")
        for row in ablation_rows:
            full_csr = (
                f"{row['full_checker_non_strict']} / {row['full_checker_strict']}"
                if row["full_checker_non_strict"] != "N/A"
                else "N/A"
            )
            full_it_t = (
                f"{row['full_iterations']} / {row['full_time_s']:.2f}"
                if row["full_iterations"] != "N/A"
                else "N/A"
            )
            full_ns = (
                f"{row['full_nav_calls']} / {row['full_solver_calls']}"
                if row["full_nav_calls"] != "N/A"
                else "N/A"
            )
            lines.append(
                f"| `{row['project']}` | {full_csr} | {full_it_t} | {full_ns} | {row['baseline_checker_non_strict']} / {row['baseline_checker_strict']} | {row['baseline_model_result']} | {row['baseline_iterations']} / {row['baseline_time_s']:.2f} | {row['baseline_shell_calls']} |"
            )

        if paired_full:
            delta_ns = round(sub_full_sum["csr_non_strict"] - ab_sum["csr_non_strict"], 1)
            lines.append("\n### Ablation Summary (Matched Subset)\n")
            lines.append("| Configuration | Subset Size | CSR (Non-Strict) | CSR (Strict) | Avg Iterations | Avg Time (s) |")
            lines.append("| :--- | :-: | :-: | :-: | :-: | :-: |")
            lines.append(
                f"| Full CompileAgent | {sub_full_sum['count']} | {sub_full_sum['csr_non_strict_count']}/{sub_full_sum['count']} ({sub_full_sum['csr_non_strict']}%) | {sub_full_sum['csr_strict_count']}/{sub_full_sum['count']} ({sub_full_sum['csr_strict']}%) | {sub_full_sum['avg_iterations']} | {sub_full_sum['avg_time_s']} |"
            )
            lines.append(
                f"| Baseline (No Navigator & No ErrorSolver) | {ab_sum['count']} | {ab_sum['csr_non_strict_count']}/{ab_sum['count']} ({ab_sum['csr_non_strict']}%) | {ab_sum['csr_strict_count']}/{ab_sum['count']} ({ab_sum['csr_strict']}%) | {ab_sum['avg_iterations']} | {ab_sum['avg_time_s']} |"
            )
            lines.append(f"\nCSR Gain (Full vs Baseline, Non-Strict): **{delta_ns:+.1f} percentage points**\n")

    return "\n".join(lines) + "\n", ablation_rows


@click.command()
@click.option("--full-dir", type=str, default="outputs/runs/benchmark_full", help="Directory containing Full CompileAgent JSON logs")
@click.option("--ablation-dir", type=str, default="outputs/runs/benchmark_ablation", help="Directory containing Baseline/Ablation JSON logs")
@click.option("--out-dir", type=str, default="outputs/reports/eval", help="Directory to save CSV and Markdown evaluation outputs")
def main(full_dir, ablation_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    full_records = load_dedup_records(full_dir, "full")
    ablation_records = load_dedup_records(ablation_dir, "baseline_no_nav_no_solver")

    detail_fields = [
        "project",
        "mode",
        "model_result",
        "checker_non_strict",
        "checker_strict",
        "smoke_passed",
        "smoke_checked",
        "missing_libs",
        "matched_targets",
        "iterations",
        "elapsed_seconds",
        "shell_calls",
        "navigator_calls",
        "error_solver_calls",
        "diagnosis",
        "timestamp",
        "log_file",
    ]
    all_records = full_records + ablation_records
    details_csv = os.path.join(out_dir, "benchmark_details.csv")
    write_csv(details_csv, all_records, detail_fields)

    md_text, ablation_rows = build_markdown_report(full_records, ablation_records)
    if ablation_rows:
        ab_fields = list(ablation_rows[0].keys())
        write_csv(os.path.join(out_dir, "ablation_comparison.csv"), ablation_rows, ab_fields)

    md_path = os.path.join(out_dir, "benchmark_report.md")
    atomic_write_text(md_path, md_text)

    results_json_path = os.path.join(out_dir, "benchmark_results.json")
    atomic_write_json(
        results_json_path,
        {
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "full_dir": str(full_dir or "").replace("\\", "/"),
            "ablation_dir": str(ablation_dir or "").replace("\\", "/"),
            "full_summary": compute_summary(full_records),
            "ablation_summary": compute_summary(ablation_records) if ablation_records else {},
            "full_records": full_records,
            "ablation_records": ablation_records,
        },
        indent=2,
    )

    print(md_text)
    print(f"Saved details CSV -> {details_csv}")
    print(f"Saved Markdown report -> {md_path}")
    print(f"Saved consolidated JSON -> {results_json_path}")


if __name__ == "__main__":
    main()
