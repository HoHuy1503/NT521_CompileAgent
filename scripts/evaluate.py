import csv
import glob
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


def load_dedup_records(log_dir: str, default_mode: str) -> list:
    if not log_dir or not os.path.isdir(log_dir):
        return []
    latest = {}
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

        build = data["build"]
        tools = data.get("tools", {}) or {}
        proj = build.get("project") or fallback_proj
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
            "log_file": fp.replace("\\", "/"),
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

        key = (proj, mode)
        if key not in latest or rec["timestamp"] > latest[key]["timestamp"]:
            latest[key] = rec

    return sorted(latest.values(), key=lambda r: (r["mode"], r["project"].lower()))


def load_security_records(log_dir: str) -> list:
    if not log_dir:
        return []
    sec_dir = os.path.join(log_dir, "security")
    if not os.path.isdir(sec_dir):
        return []

    records = []
    for fp in sorted(glob.glob(os.path.join(sec_dir, "*.security.json"))):
        proj = os.path.basename(fp)
        if proj.endswith(".security.json"):
            proj = proj[: -len(".security.json")]
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        recipe = data.get("recipe") or {}
        l1 = data.get("layer1_sast") or {}
        sg = l1.get("semgrep") or {}
        cpp = l1.get("cppcheck") or {}
        cpp_sev = cpp.get("by_severity") or {}
        sg_rules = sg.get("by_rule") or {}

        l2 = data.get("layer2_hardening_baseline") or {}
        l3h = data.get("layer3_hardening_rebuild") or {}
        l3h_build = l3h.get("build") or {}
        l3h_after = l3h.get("audit_after") or {}
        l3h_smoke = l3h.get("smoke_check_after") or {}
        l3h_diff = l3h.get("before_vs_after") or {}

        l3s = data.get("layer3_sanitizer") or {}
        l3s_build = l3s.get("build") or {}

        base_avg = l2.get("average_score")
        after_avg = l3h_after.get("average_score")
        delta = l3h_diff.get("avg_score_delta")
        if delta is None and base_avg is not None and after_avg is not None:
            delta = round(float(after_avg) - float(base_avg), 2)

        records.append(
            {
                "project": proj,
                "security_mode": data.get("mode", ""),
                "security_status": data.get("status", ""),
                "build_system": recipe.get("build_system", ""),
                "semgrep_status": sg.get("status", "skipped"),
                "semgrep_findings": int(sg.get("total_findings", 0) or 0),
                "semgrep_errors": int(sg.get("errors_count", 0) or 0),
                "semgrep_rules": ", ".join(f"{k}:{v}" for k, v in sorted(sg_rules.items())),
                "cppcheck_status": cpp.get("status", "skipped"),
                "cppcheck_mode": cpp.get("mode", ""),
                "cppcheck_db_entries": int(cpp.get("compile_db_entries", 0) or 0),
                "cppcheck_findings": int(cpp.get("total_findings", 0) or 0),
                "cppcheck_error": int(cpp_sev.get("error", 0) or 0),
                "cppcheck_warning": int(cpp_sev.get("warning", 0) or 0),
                "cppcheck_style": int(cpp_sev.get("style", 0) or 0),
                "cppcheck_performance": int(cpp_sev.get("performance", 0) or 0),
                "cppcheck_portability": int(cpp_sev.get("portability", 0) or 0),
                "binary_count": int(l2.get("binary_count", 0) or 0),
                "baseline_avg_score": round(float(base_avg), 2) if base_avg is not None else 0.0,
                "after_hardening_avg_score": round(float(after_avg), 2) if after_avg is not None else 0.0,
                "score_delta": round(float(delta), 2) if delta is not None else 0.0,
                "hardening_build_status": l3h_build.get("status", "skipped"),
                "hardened_smoke_passed": bool(l3h_smoke.get("passed", False)),
                "hardened_smoke_checked": int(l3h_smoke.get("checked_count", 0) or 0),
                "sanitizer_build_status": l3s_build.get("status", "skipped"),
                "asan_instrumented_count": int(l3s.get("asan_instrumented_count", 0) or 0),
                "ubsan_instrumented_count": int(l3s.get("ubsan_instrumented_count", 0) or 0),
                "asan_count": int(l3s.get("asan_count", 0) or 0),
                "ubsan_count": int(l3s.get("ubsan_count", 0) or 0),
                "security_report_file": fp.replace("\\", "/"),
            }
        )
    return sorted(records, key=lambda r: r["project"].lower())


def compute_security_summary(sec_records: list) -> dict:
    n = len(sec_records)
    if n == 0:
        return {"count": 0}
    hard_ok = sum(1 for r in sec_records if r["hardening_build_status"] == "completed")
    smoke_ok = sum(1 for r in sec_records if r["hardened_smoke_passed"])
    san_ok = sum(1 for r in sec_records if r["sanitizer_build_status"] == "completed")
    compdb_cnt = sum(1 for r in sec_records if r["cppcheck_mode"] == "compile_commands")
    baseline_pool = [r["baseline_avg_score"] for r in sec_records if r.get("binary_count", 0) > 0]
    hardened_pool = [
        r
        for r in sec_records
        if r.get("binary_count", 0) > 0 and r.get("hardening_build_status") == "completed"
    ]
    avg_baseline = round(mean(baseline_pool), 2) if baseline_pool else 0.0
    avg_after = (
        round(mean(r["after_hardening_avg_score"] for r in hardened_pool), 2)
        if hardened_pool
        else 0.0
    )
    avg_delta = (
        round(mean(r["score_delta"] for r in hardened_pool), 2)
        if hardened_pool
        else 0.0
    )
    return {
        "count": n,
        "semgrep_total_findings": sum(r["semgrep_findings"] for r in sec_records),
        "semgrep_projects_with_findings": sum(1 for r in sec_records if r["semgrep_findings"] > 0),
        "cppcheck_total_findings": sum(r["cppcheck_findings"] for r in sec_records),
        "cppcheck_total_errors": sum(r["cppcheck_error"] for r in sec_records),
        "cppcheck_compile_db_projects": compdb_cnt,
        "avg_baseline_score": avg_baseline,
        "avg_after_score": avg_after,
        "avg_score_delta": avg_delta,
        "hardening_build_ok": hard_ok,
        "hardened_smoke_ok": smoke_ok,
        "sanitizer_build_ok": san_ok,
        "total_asan_findings": sum(r["asan_count"] for r in sec_records),
        "total_ubsan_findings": sum(r["ubsan_count"] for r in sec_records),
    }


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
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def build_markdown_report(
    full_records: list, ablation_records: list, security_records: list = None
) -> tuple:
    full_sum = compute_summary(full_records)
    lines = ["# Stage 2 & Stage 3 Benchmark Evaluation Report\n"]

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

    if security_records:
        sec_sum = compute_security_summary(security_records)
        lines.append("\n## 3. Stage 3 3-Layer Security Pipeline Results\n")
        lines.append(
            "| Project | Build Sys | Semgrep (Findings / ParseErr) | Cppcheck Mode (DB) | Cppcheck (Total / E / W / S) | Hardening Score (Before -> After / Delta) | Hardening Build / Smoke | Sanitizer Instr (ASan/UBSan) | Runtime Findings (ASan / UBSan) |"
        )
        lines.append("| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |")
        for sr in security_records:
            sg_col = f"{sr['semgrep_findings']} / {sr['semgrep_errors']}"
            cpp_mode_col = f"{sr['cppcheck_mode']} ({sr['cppcheck_db_entries']})"
            cpp_col = f"{sr['cppcheck_findings']} ({sr['cppcheck_error']}/{sr['cppcheck_warning']}/{sr['cppcheck_style']})"
            score_col = f"{sr['baseline_avg_score']:.1f} -> {sr['after_hardening_avg_score']:.1f} ({sr['score_delta']:+.1f})"
            hard_col = f"{sr['hardening_build_status']} / {'PASS' if sr['hardened_smoke_passed'] else 'FAIL'}"
            san_instr_col = f"{sr['sanitizer_build_status']} ({sr['asan_instrumented_count']}/{sr['ubsan_instrumented_count']})"
            san_find_col = f"{sr['asan_count']} / {sr['ubsan_count']}"
            lines.append(
                f"| `{sr['project']}` | {sr['build_system'] or '-'} | {sg_col} | {cpp_mode_col} | {cpp_col} | {score_col} | {hard_col} | {san_instr_col} | {san_find_col} |"
            )

        lines.append("\n### Aggregate Metrics (Stage 3 Security Pipeline)\n")
        lines.append("| Metric | Value |")
        lines.append("| :--- | :--- |")
        lines.append(f"| Projects Audited | {sec_sum['count']} |")
        lines.append(f"| Semgrep Total Findings | {sec_sum['semgrep_total_findings']} ({sec_sum['semgrep_projects_with_findings']}/{sec_sum['count']} projects) |")
        lines.append(f"| Cppcheck `compile_commands` Coverage | {sec_sum['cppcheck_compile_db_projects']}/{sec_sum['count']} projects |")
        lines.append(f"| Cppcheck Total Findings (Severity=Error) | {sec_sum['cppcheck_total_findings']} ({sec_sum['cppcheck_total_errors']} errors) |")
        lines.append(
            f"| Binary Hardening Avg Score (Baseline -> Hardened) | {sec_sum['avg_baseline_score']} -> {sec_sum['avg_after_score']} (avg delta {sec_sum['avg_score_delta']:+.2f}) |"
        )
        lines.append(f"| Hardening Rebuild / Smoke Pass Rate | {sec_sum['hardening_build_ok']}/{sec_sum['count']} builds ok, {sec_sum['hardened_smoke_ok']}/{sec_sum['count']} smoke pass |")
        lines.append(f"| Sanitizer Build Success Rate | {sec_sum['sanitizer_build_ok']}/{sec_sum['count']} |")
        lines.append(f"| Runtime Vulnerabilities Caught (ASan / UBSan) | {sec_sum['total_asan_findings']} ASan / {sec_sum['total_ubsan_findings']} UBSan |")

    return "\n".join(lines) + "\n", ablation_rows


@click.command()
@click.option("--full-dir", type=str, default="outputs/runs/stage2_full", help="Directory containing Full CompileAgent JSON logs")
@click.option("--ablation-dir", type=str, default="outputs/runs/stage2_ablation", help="Directory containing Baseline/Ablation JSON logs")
@click.option("--out-dir", type=str, default="outputs/reports/stage2_eval", help="Directory to save CSV and Markdown evaluation outputs")
def main(full_dir, ablation_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    full_records = load_dedup_records(full_dir, "full")
    ablation_records = load_dedup_records(ablation_dir, "baseline_no_nav_no_solver")
    security_records = load_security_records(full_dir)

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

    md_text, ablation_rows = build_markdown_report(
        full_records, ablation_records, security_records=security_records
    )
    if ablation_rows:
        ab_fields = list(ablation_rows[0].keys())
        write_csv(os.path.join(out_dir, "ablation_comparison.csv"), ablation_rows, ab_fields)

    security_csv = None
    if security_records:
        sec_fields = list(security_records[0].keys())
        security_csv = os.path.join(out_dir, "security_summary.csv")
        write_csv(security_csv, security_records, sec_fields)

    md_path = os.path.join(out_dir, "benchmark_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    print(md_text)
    print(f"Saved details CSV -> {details_csv}")
    if security_csv:
        print(f"Saved security CSV -> {security_csv}")
    print(f"Saved Markdown report -> {md_path}")


if __name__ == "__main__":
    main()
