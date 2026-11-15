import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
for path in (SRC_DIR, SCRIPTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from langchain_core.agents import AgentAction, AgentFinish  # noqa: E402

import compile_agent.config as cfg  # noqa: E402
from compile_agent.agent.error_solver import _truncate_mistake_text  # noqa: E402
from compile_agent.agent.executor import (  # noqa: E402
    CompilationAgentExecutor,
    SafeReActSingleInputOutputParser,
    _REPETITION_WARNING,
)
from compile_agent.agent.navigator import AgentTools, CompileNavigator  # noqa: E402
from compile_agent.infra.docker_tools import InteractiveDockerShell  # noqa: E402
from compile_agent.reporting.logs import (  # noqa: E402
    _is_build_system_probe_artifact,
    _matches_target_name,
    find_compiled_artifacts,
    is_compiled,
    smoke_check_artifact,
    snapshot_existing_artifacts,
)
from compile_agent.security.audit import detect_build_recipe  # noqa: E402
import compile as compile_script  # noqa: E402
import evaluate as evaluate_script  # noqa: E402


class TestSafeReActParser(unittest.TestCase):
    def setUp(self):
        self.parser = SafeReActSingleInputOutputParser()

    def test_action_before_hallucinated_final_answer_returns_agent_action(self):
        text = (
            "Thought: Let's compile the project first.\n"
            "Action: Shell\n"
            "Action Input: make -j4\n"
            "Observation: gcc -c main.c\n"
            "Thought: Looks compiled.\n"
            "Final Answer: COMPILATION-SUCCESS"
        )
        parsed = self.parser.parse(text)
        self.assertIsInstance(parsed, AgentAction)
        self.assertEqual(parsed.tool, "Shell")
        self.assertEqual(parsed.tool_input, "make -j4")

    def test_final_answer_before_action_returns_agent_finish(self):
        text = (
            "Thought: Everything built cleanly.\n"
            "Final Answer: COMPILATION-SUCCESS\n"
            "Action: Shell\n"
            "Action Input: ls"
        )
        parsed = self.parser.parse(text)
        self.assertIsInstance(parsed, AgentFinish)
        self.assertEqual(parsed.return_values["output"], "COMPILATION-SUCCESS")


class TestStateEpochRepetitionAndLoopDetection(unittest.TestCase):
    def test_make_allowed_after_state_mutating_actions(self):
        steps = [
            (AgentAction("Shell", "make", "t1"), "fatal error: zlib.h: No such file\n[Exit code: 2]"),
            (AgentAction("Shell", "apt-get install -y zlib1g-dev", "t2"), "Setting up zlib1g-dev\n[Exit code: 0]"),
            (AgentAction("Shell", "make", "t3"), "error: undeclared symbol\n[Exit code: 2]"),
            (AgentAction("Shell", "sed -i 's/bad/good/' Makefile", "t4"), "[Exit code: 0]"),
        ]
        next_make = AgentAction("Shell", "make", "t5")
        self.assertIsNone(CompilationAgentExecutor._check_repetition(next_make, steps))
        self.assertFalse(CompilationAgentExecutor._is_stuck_in_loop(steps))

    def test_repeated_make_within_same_epoch_blocked_and_detected(self):
        steps_two = [
            (AgentAction("Shell", "make", "t1"), "error: missing foo\n[Exit code: 2]"),
            (AgentAction("Shell", "ls -la", "t2"), "Makefile\n[Exit code: 0]"),
            (AgentAction("Shell", "make", "t3"), "error: missing foo\n[Exit code: 2]"),
            (AgentAction("Shell", "cat Makefile", "t4"), "all:\n[Exit code: 0]"),
        ]
        rep = CompilationAgentExecutor._check_repetition(
            AgentAction("Shell", "make", "t5"), steps_two
        )
        self.assertIsNotNone(rep)
        self.assertIn(_REPETITION_WARNING.strip(), rep)

        steps_three = steps_two + [
            (AgentAction("Shell", "make", "t5"), rep),
        ]
        self.assertTrue(CompilationAgentExecutor._is_stuck_in_loop(steps_three))

    def test_alternating_non_mutating_loop_stops_at_six_steps(self):
        steps = [
            (AgentAction("Shell", "ls", "t1"), "a\n[Exit code: 0]"),
            (AgentAction("Shell", "cat README.md", "t2"), "b\n[Exit code: 0]"),
            (AgentAction("Shell", "ls", "t3"), "a\n[Exit code: 0]"),
            (AgentAction("Shell", "cat README.md", "t4"), "b\n[Exit code: 0]"),
            (AgentAction("Shell", "ls", "t5"), "a\n[Exit code: 0]"),
            (AgentAction("Shell", "cat README.md", "t6"), "b\n[Exit code: 0]"),
        ]
        self.assertTrue(CompilationAgentExecutor._is_stuck_in_loop(steps))

    def test_enriched_input_skips_exception_step_and_picks_tree_obs(self):
        steps = [
            (AgentAction("_Exception", "bad", "bad"), "Invalid Format"),
            (AgentAction("Shell", "tree . -L 2", "ok"), "|-- CMakeLists.txt\n|-- src"),
        ]
        enriched = CompilationAgentExecutor._build_enriched_tool_input(
            AgentAction("CompileNavigator", "find build steps", "nav"), steps
        )
        self.assertIn("|-- CMakeLists.txt", enriched)


class TestQuotaClassificationAndKeyPool(unittest.TestCase):
    def tearDown(self):
        cfg._KEY_POOL.configure([])
        cfg.reset_circuit_breaker()

    def test_rpm_429_not_misclassified_as_daily_quota(self):
        rpm_err = RuntimeError(
            "Error code: 429 - You exceeded your current quota, limit: 15 requests per minute, retry in 20s"
        )
        self.assertFalse(cfg._is_quota_exhausted_error(rpm_err))
        self.assertTrue(cfg._is_rate_limit_error(rpm_err))

        daily_err = RuntimeError("429 GenerateRequestsPerDay / PerDay quota exceeded")
        self.assertTrue(cfg._is_quota_exhausted_error(daily_err))

    def test_two_strike_protection_and_success_reset(self):
        cfg._KEY_POOL.configure(["key-A-1234567890", "key-B-1234567890"])
        ambiguous_429 = RuntimeError("Error code: 429 - quota limit reached")
        next_key = cfg._KEY_POOL.mark_exhausted_and_rotate(reason="quota_exhausted", exc=ambiguous_429)
        self.assertEqual(next_key, "key-B-1234567890")
        self.assertEqual(cfg._KEY_POOL.remaining_count(), 2)

        # Success clears strike history
        cfg.record_llm_success()
        self.assertEqual(cfg._KEY_POOL._quota_strikes, {})

        # Two strikes on key-B permanently exhausts it
        cfg._KEY_POOL.mark_exhausted_and_rotate(reason="quota_exhausted", exc=ambiguous_429)
        self.assertEqual(cfg._KEY_POOL.remaining_count(), 2)
        # Rotate back to key-B by hitting strike 1 on key-A (now active) -> brings back key-B cleanly via explicit reset test
        cfg._KEY_POOL.configure(["key-only-123456"])
        cfg._KEY_POOL.mark_exhausted_and_rotate(reason="quota_exhausted", exc=ambiguous_429)
        self.assertEqual(cfg._KEY_POOL.remaining_count(), 1)
        exhausted = cfg._KEY_POOL.mark_exhausted_and_rotate(reason="quota_exhausted", exc=ambiguous_429)
        self.assertIsNone(exhausted)
        self.assertEqual(cfg._KEY_POOL.remaining_count(), 0)

    def test_extra_body_400_fallback_strips_and_retries(self):
        class FakeClient:
            def __init__(self):
                self.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
                self.model_kwargs = {"extra_body": dict(self.extra_body)}
                self.calls = 0

            def invoke(self, _x):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("Error code: 400 - unrecognized request argument: chat_template_kwargs")
                return "ok"

        client = FakeClient()
        cfg._KEY_POOL.configure(["EMPTY"])
        cfg._KEY_POOL.register_client(client)
        res = cfg.call_with_rate_limit_retry(client.invoke, "hello")
        self.assertEqual(res, "ok")
        self.assertEqual(client.calls, 2)
        self.assertIsNone(client.extra_body)
        self.assertNotIn("extra_body", client.model_kwargs)


class TestDockerShellAndStage3Helpers(unittest.TestCase):
    def _dummy_shell(self):
        shell = InteractiveDockerShell.__new__(InteractiveDockerShell)
        shell.last_line = "root@c0mpi1er-c0nta1ner:/work# "
        shell.logger = []
        shell.use_proxy = False
        return shell

    def test_detect_interactive_repl_and_gdb(self):
        self.assertTrue(InteractiveDockerShell._detect_interactive_command("python3"))
        self.assertEqual(InteractiveDockerShell._detect_interactive_command("python3 setup.py build"), "")
        self.assertTrue(InteractiveDockerShell._detect_interactive_command("gdb ./app"))
        self.assertEqual(InteractiveDockerShell._detect_interactive_command("gdb -batch -ex run ./app"), "")

    def test_omit_preserves_key_error_lines(self):
        shell = self._dummy_shell()
        lines = [f"cc -c file{i}.c" for i in range(80)]
        lines[10] = "src/main.c:4:10: fatal error: foo.h: No such file or directory"
        lines[12] = "/usr/bin/ld: main.o: undefined reference to 'bar'"
        lines.append("make: *** [Makefile:10: all] Error 2")
        out = shell.omit("make -j4", "\n".join(lines), 1.0, exit_code=2, record_log=False)
        self.assertIn("[Key Error Lines]:", out)
        self.assertIn("fatal error: foo.h: No such file or directory", out)
        self.assertIn("undefined reference to 'bar'", out)

        mid_err = "\nfatal error: bar.h: No such file or directory\n"
        huge = ("A" * 4500) + mid_err + ("B" * 4500)
        out2 = shell.omit("cmake --build build", huge, 1.0, exit_code=1, record_log=False)
        self.assertIn("[Key Error Lines]:", out2)
        self.assertIn("fatal error: bar.h: No such file or directory", out2)

    def test_truncate_mistake_text_keeps_errors_and_caps_length(self):
        padding = "\n".join(f"noise line {i}" for i in range(400))
        mistake = f"{padding}\nsrc/a.c:12: error: expected ';'\n{padding}"
        self.assertGreater(len(mistake), 4000)
        compacted = _truncate_mistake_text(mistake, limit=4000)
        self.assertLessEqual(len(compacted), 4000)
        self.assertIn("error: expected ';'", compacted)

    def test_navigator_normalizes_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            readme = os.path.join(tmpdir, "README.md")
            with open(readme, "w", encoding="utf-8") as f:
                f.write("")
            atools = AgentTools(tmpdir, "demo", "tree", False)
            res = atools.search_instructions_from_files("./README.md")
            self.assertEqual(res, "File is empty")

    def test_has_existing_log_respects_security_mode_and_glob_chars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proj = "proj[1]"
            fp = os.path.join(tmpdir, f"{proj}.log.2026-01-01-00-00-00-000001.json")
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "build": {"checker_non_strict": True, "checker_result": "True(strict=False)"},
                        "security": {"mode": "none", "status": "skipped"},
                    },
                    f,
                )
            self.assertTrue(compile_script._has_existing_log(tmpdir, proj, security_mode="none"))
            self.assertFalse(compile_script._has_existing_log(tmpdir, proj, security_mode="all"))

            os.makedirs(os.path.join(tmpdir, "security"), exist_ok=True)
            with open(os.path.join(tmpdir, "security", f"{proj}.security.json"), "w", encoding="utf-8") as sf:
                json.dump({"status": "completed"}, sf)
            self.assertTrue(compile_script._has_existing_log(tmpdir, proj, security_mode="all"))

    def test_detect_build_recipe_prefers_exit0_and_keeps_env(self):
        shell = SimpleNamespace(container_id=None, execute_command=lambda cmd, record_log=False: "[Exit code: 0]")
        tools_log = {
            "Shell": [
                ["make BROKEN=1", "error\n[Exit code: 2]", 1.0],
                ["CC=gcc CFLAGS=-O2 make", "ok\n[Exit code: 0]", 2.0],
            ]
        }
        with patch("compile_agent.security.audit._run_container_python", return_value={"cmake_dirs": [], "ninja_dirs": [], "makefiles": []}), \
             patch("compile_agent.security.audit._exec_in_container", return_value={"returncode": 0, "stdout": "", "stderr": ""}), \
             patch("compile_agent.security.audit._probe_makefile_targets", return_value={"clean": False, "test": False, "check": False}):
            recipe = detect_build_recipe(shell, tools_log)
        self.assertEqual(recipe["build_system"], "make")
        self.assertIn("CC=gcc CFLAGS=-O2 make -j", recipe["build_cmd"])
        self.assertNotIn("*.a", recipe["clean_cmd"])
        self.assertNotIn("*.so", recipe["clean_cmd"])

    def test_smoke_check_appends_linked_binaries_for_static_targets(self):
        found = {"matched_targets": ["libfoo.a"], "all_artifacts": ["libfoo.a", "demo_cli", "libfoo.so"]}
        seen_cmds = []

        def fake_exec(cmd, record_log=False):
            seen_cmds.append(cmd)
            if "file -L" in cmd:
                if "libfoo.a" in cmd:
                    return "/work/libfoo.a: current ar archive\n[Exit code: 0]"
                if "demo_cli" in cmd:
                    return "/work/demo_cli: ELF 64-bit LSB executable\n[Exit code: 0]"
                if "libfoo.so" in cmd:
                    return "/work/libfoo.so: ELF 64-bit LSB shared object\n[Exit code: 0]"
            return "[Exit code: 0]"

        shell = SimpleNamespace(container_id=None, execute_command=fake_exec)
        res = smoke_check_artifact(shell, ".", target_files=["libfoo.a"], precomputed_found=found)
        self.assertEqual(res["checked_count"], 3)
        self.assertTrue(res["passed"])


class TestAntiFakeArtifactsAndMetricsAccuracy(unittest.TestCase):
    def tearDown(self):
        cfg._KEY_POOL.configure([])
        cfg.reset_circuit_breaker()

    def test_snapshot_and_probe_filter_exclude_fake_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            elf_bytes = b"\x7fELF" + b"\x00" * 32
            preexisting = os.path.join(tmpdir, "preexisting_tool")
            with open(preexisting, "wb") as f:
                f.write(elf_bytes)

            snap = snapshot_existing_artifacts(tmpdir)
            self.assertEqual(snap, {"preexisting_tool"})

            # Simulate CMake/Autoconf compiler probe binaries + real built binary + intermediate .o
            probe_paths = [
                os.path.join(tmpdir, "build", "CMakeFiles", "3.25.1", "CompilerIdC", "a.out"),
                os.path.join(tmpdir, "build", "CMakeFiles", "CheckTypeSize", "INT.bin"),
                os.path.join(tmpdir, "conftest"),
            ]
            for p in probe_paths:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "wb") as f:
                    f.write(elf_bytes)

            self.assertTrue(_is_build_system_probe_artifact("build/CMakeFiles/3.25.1/CompilerIdC/a.out"))
            self.assertTrue(_is_build_system_probe_artifact("build/CMakeFiles/CheckTypeSize/INT.bin"))
            self.assertTrue(_is_build_system_probe_artifact("conftest"))
            self.assertFalse(_is_build_system_probe_artifact("build/demo_app"))

            obj_path = os.path.join(tmpdir, "build", "main.o")
            app_path = os.path.join(tmpdir, "build", "demo_app")
            with open(obj_path, "wb") as f:
                f.write(elf_bytes)
            with open(app_path, "wb") as f:
                f.write(elf_bytes)

            found = find_compiled_artifacts(tmpdir, target_files=["demo_app"], ignore_relpaths=snap)
            self.assertEqual(found["matched_targets"], ["build/demo_app"])
            self.assertEqual(found["all_artifacts"], ["build/demo_app"])

    def test_matches_target_name_and_is_compiled_aliases(self):
        self.assertTrue(_matches_target_name("linenoise-example", "linenoise_example"))
        self.assertTrue(_matches_target_name("libpcap.so.1.11.1-PRE-GIT", "libpcap.so"))
        self.assertFalse(_matches_target_name("libfoo.so.1", "libpcap.so"))

        with tempfile.TemporaryDirectory() as tmpdir:
            elf_bytes = b"\x7fELF" + b"\x00" * 32
            for name in ("linenoise-example", "libpcap.so.1.11.1-PRE-GIT"):
                with open(os.path.join(tmpdir, name), "wb") as f:
                    f.write(elf_bytes)
            found = find_compiled_artifacts(tmpdir, target_files=["linenoise_example", "libpcap.so"])
            self.assertTrue(
                is_compiled(tmpdir, ["linenoise_example", "libpcap.so"], strict=True, precomputed_found=found)
            )

    def test_smoke_check_rejects_intermediate_objects_only_when_unmatched(self):
        found = {"matched_targets": [], "all_artifacts": ["obj/foo.o", "obj/bar.lo"]}
        shell = SimpleNamespace(
            container_id=None,
            execute_command=lambda cmd, record_log=False: "/work/obj/foo.o: ELF 64-bit LSB relocatable\n[Exit code: 0]",
        )
        res = smoke_check_artifact(shell, ".", target_files=["app_bin"], precomputed_found=found)
        self.assertFalse(res["passed"])

    def test_parse_model_decision_case_insensitive_and_iteration_limit(self):
        self.assertEqual(
            compile_script.parse_model_decision({"final_answer": "All built: compilation_success"}),
            "True",
        )
        self.assertEqual(
            compile_script.parse_model_decision({"output": "Agent stopped due to iteration limit or time limit."}),
            "UNCERTAIN",
        )
        self.assertEqual(
            compile_script.parse_model_decision({"output": "Build failed completely"}),
            "False",
        )

    def test_evaluate_avoids_stage3_dilution_and_unmatched_smoke_count(self):
        sec_records = [
            {
                "project": "ok_repo",
                "binary_count": 2,
                "baseline_avg_score": 40.0,
                "after_hardening_avg_score": 80.0,
                "score_delta": 40.0,
                "hardening_build_status": "completed",
                "hardened_smoke_passed": True,
                "sanitizer_build_status": "completed",
                "cppcheck_mode": "compile_commands",
                "semgrep_findings": 1,
                "cppcheck_findings": 2,
                "cppcheck_error": 1,
                "asan_count": 0,
                "ubsan_count": 0,
            },
            {
                "project": "zero_bin_repo",
                "binary_count": 0,
                "baseline_avg_score": 0.0,
                "after_hardening_avg_score": 0.0,
                "score_delta": 0.0,
                "hardening_build_status": "skipped",
                "hardened_smoke_passed": False,
                "sanitizer_build_status": "skipped",
                "cppcheck_mode": "fallback",
                "semgrep_findings": 0,
                "cppcheck_findings": 0,
                "cppcheck_error": 0,
                "asan_count": 0,
                "ubsan_count": 0,
            },
            {
                "project": "failed_hardening_repo",
                "binary_count": 1,
                "baseline_avg_score": 20.0,
                "after_hardening_avg_score": 0.0,
                "score_delta": -20.0,
                "hardening_build_status": "failed",
                "hardened_smoke_passed": False,
                "sanitizer_build_status": "failed",
                "cppcheck_mode": "fallback",
                "semgrep_findings": 0,
                "cppcheck_findings": 0,
                "cppcheck_error": 0,
                "asan_count": 0,
                "ubsan_count": 0,
            },
        ]
        sec_sum = evaluate_script.compute_security_summary(sec_records)
        self.assertEqual(sec_sum["avg_baseline_score"], 30.0)
        self.assertEqual(sec_sum["avg_after_score"], 80.0)
        self.assertEqual(sec_sum["avg_score_delta"], 40.0)

        stage2_records = [
            {
                "model_result": "False",
                "checker_non_strict": False,
                "checker_strict": False,
                "smoke_passed": True,
                "matched_targets": "",
                "elapsed_seconds": 10.0,
                "iterations": 2,
                "shell_calls": 3,
                "navigator_calls": 0,
                "error_solver_calls": 0,
            }
        ]
        s2_sum = evaluate_script.compute_summary(stage2_records)
        self.assertEqual(s2_sum["smoke_pass_count"], 0)

    def test_local_llm_circuit_breaker_requires_two_failures_and_health_retries(self):
        cfg.reset_circuit_breaker()
        fake_cfg = {
            "provider": "ollama",
            "base_url": "http://127.0.0.1:11434/v1",
            "api_key": "ollama",
            "circuit_breaker_threshold": 3,
        }
        with patch("compile_agent.config.resolve_llm_config", return_value=fake_cfg), \
             patch("compile_agent.config.probe_llm_server_health", return_value=False):
            first = cfg.record_llm_failure(TimeoutError("busy"), check_health=True)
            self.assertFalse(first)
            second = cfg.record_llm_failure(TimeoutError("busy again"), check_health=True)
            self.assertTrue(second)

        class FakeResp:
            status = 200
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False

        calls = {"n": 0}
        class FakeOpener:
            def open(self, req, timeout=10.0):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise TimeoutError("transient stall")
                return FakeResp()

        with patch("compile_agent.config.resolve_llm_config", return_value=fake_cfg), \
             patch("urllib.request.build_opener", return_value=FakeOpener()), \
             patch("compile_agent.config.time.sleep"):
            self.assertTrue(cfg.probe_llm_server_health())
            self.assertEqual(calls["n"], 2)


if __name__ == "__main__":
    unittest.main()
