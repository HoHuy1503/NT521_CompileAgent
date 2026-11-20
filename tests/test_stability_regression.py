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
from compile_agent.agent.error_solver import (  # noqa: E402
    ErrorSolver,
    _truncate_mistake_text,
    clean_output,
    parse_output,
    trans_confidence,
)
from compile_agent.agent.executor import (  # noqa: E402
    CompilationAgentExecutor,
    SafeReActSingleInputOutputParser,
    _REPETITION_WARNING,
)
from compile_agent.agent.navigator import AgentTools  # noqa: E402
from compile_agent.infra.docker_tools import InteractiveDockerShell  # noqa: E402
from compile_agent.reporting.logs import (  # noqa: E402
    RESULTS_JSONL_FILENAME,
    _is_build_system_probe_artifact,
    _matches_target_name,
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
    backfill_results_jsonl,
    cleanup_stale_tmp_files,
    find_compiled_artifacts,
    is_compiled,
    iter_jsonl_records,
    save_logs,
    smoke_check_artifact,
    snapshot_existing_artifacts,
)
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


class TestLocalLLMConfigAndRetry(unittest.TestCase):
    def tearDown(self):
        cfg._EXTRA_BODY_UNSUPPORTED_ENDPOINTS.clear()
        cfg.reset_stop_bypass_cache()
        cfg.reset_circuit_breaker()

    def test_local_providers_resolve_without_api_key(self):
        for provider, expected_url, expected_model in (
            ("llamacpp", "http://localhost:8080/v1", "Qwen 3.8 K_M Q4 27B"),
            ("ollama", "http://localhost:11434/v1", "qwen3:30b"),
            ("vllm", "http://localhost:8000/v1", "Qwen/Qwen3-30B-A3B"),
        ):
            with patch.dict(
                os.environ,
                {
                    "LLM_PROVIDER": provider,
                    "LLM_BASE_URL": "",
                    "OPENAI_BASE_URL": "",
                    "MASTER_MODEL": "",
                    "DISCUSSION_MODELS": "",
                },
                clear=False,
            ):
                resolved = cfg.resolve_llm_config()
                self.assertEqual(resolved["provider"], provider)
                self.assertEqual(resolved["base_url"], expected_url)
                self.assertEqual(resolved["master_model"], expected_model)
                self.assertNotIn("api_key", resolved)
                llm = cfg.create_chat_llm()
                self.assertEqual(llm.openai_api_key.get_secret_value(), "EMPTY")

    def test_extra_body_400_fallback_strips_and_retries(self):
        class FakeClient:
            def __init__(self):
                self.openai_api_base = "http://localhost:8080/v1"
                self.model_name = "qwen3-27b"
                self.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
                self.model_kwargs = {"extra_body": dict(self.extra_body)}
                self.calls = 0

            def invoke(self, _x):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("Error code: 400 - unrecognized request argument: chat_template_kwargs")
                return "ok"

        client = FakeClient()
        res = cfg.call_with_llm_retry(client.invoke, "hello")
        self.assertEqual(res, "ok")
        self.assertEqual(client.calls, 2)
        self.assertIsNone(client.extra_body)
        self.assertNotIn("extra_body", client.model_kwargs)


class TestErrorSolverThreeAgentTwoRoundDebate(unittest.TestCase):
    def test_trans_confidence_and_weighted_voting(self):
        self.assertEqual(trans_confidence(0.5), 0.1)
        self.assertEqual(trans_confidence(0.7), 0.3)
        self.assertEqual(trans_confidence(0.85), 0.5)
        self.assertEqual(trans_confidence(0.95), 0.8)
        self.assertEqual(trans_confidence(1.0), 1.0)

        agent_keys = ["qwen3-27b#1", "qwen3-27b#2", "qwen3-27b#3"]
        results = [
            {
                "qwen3-27b#1_output_0": {
                    "reasoning": "Missing zlib header",
                    "solution": "apt-get update && apt-get install -y zlib1g-dev",
                    "confidence_level": 0.9,
                },
                "qwen3-27b#2_output_0": {
                    "reasoning": "Need zlib dev package",
                    "solution": "apt-get update && apt-get install -y zlib1g-dev",
                    "confidence_level": 0.95,
                },
                "qwen3-27b#3_output_0": {
                    "reasoning": "Try cmake",
                    "solution": "cmake -B build",
                    "confidence_level": 0.6,
                },
            }
        ]
        cleaned = clean_output(results, rounds=0, agent_keys=agent_keys)
        parsed = parse_output(cleaned, rounds=0, threshold=30, agent_keys=agent_keys)
        self.assertEqual(
            parsed[0]["weighted_max_0"],
            "apt-get update && apt-get install -y zlib1g-dev",
        )
        self.assertIn("Majority agents' solution:", parsed[0]["debate_prompt_0"])

    def test_discussion_runs_three_agents_two_rounds_with_single_local_model(self):
        created_agents = []

        class FakeSingleAgent:
            def __init__(self, base_url=None, model_name=None, agent_key=None, temperature=None):
                self.base_url = base_url
                self.model_name = model_name
                self.agent_key = agent_key
                self.temperature = temperature
                created_agents.append((agent_key, model_name, temperature))

            def generate_answer(self, template, question, project_name, chat_history=None, search_budget=None):
                if chat_history is None:
                    return {
                        "reasoning": f"r0 from {self.agent_key}",
                        "solution": "apt-get update && apt-get install -y libssl-dev"
                        if not self.agent_key.endswith("#3")
                        else "make clean",
                        "confidence_level": 0.9 if not self.agent_key.endswith("#3") else 0.5,
                    }
                return {
                    "reasoning": f"r1 consensus from {self.agent_key}",
                    "solution": "apt-get update && apt-get install -y libssl-dev",
                    "confidence_level": 0.95,
                }

            def debate(self, template, question, project_name, results, rounds, search_budget=None):
                for result in results:
                    prev_prompt_key = f"debate_prompt_{rounds - 1}"
                    out_key = f"{self.agent_key}_output_{rounds}"
                    if (out_key not in result) and (prev_prompt_key in result) and len(result[prev_prompt_key]):
                        result[out_key] = self.generate_answer(
                            template=template,
                            question=question,
                            project_name=project_name,
                            chat_history=result[prev_prompt_key],
                            search_budget=search_budget,
                        )
                return results

        fake_cfg = {
            "provider": "llamacpp",
            "base_url": "http://localhost:8080/v1",
            "master_model": "qwen3-27b",
            "discussion_models": ["qwen3-27b"],
        }
        with patch("compile_agent.agent.error_solver.resolve_llm_config", return_value=fake_cfg), \
             patch("compile_agent.agent.error_solver.SingleAgent", FakeSingleAgent), \
             patch("compile_agent.agent.error_solver.time.sleep"):
            solver = ErrorSolver("demo-proj")
            winner = solver.discussion("fatal error: openssl/ssl.h: No such file or directory")

        self.assertEqual(
            created_agents,
            [
                ("qwen3-27b#1", "qwen3-27b", 0.1),
                ("qwen3-27b#2", "qwen3-27b", 0.4),
                ("qwen3-27b#3", "qwen3-27b", 0.7),
            ],
        )
        self.assertEqual(winner, "apt-get update && apt-get install -y libssl-dev")
        self.assertEqual(len(solver.logger), 1)
        round_data = solver.logger[0][1][0]
        self.assertIn("weighted_max_0", round_data)
        self.assertIn("weighted_max_1", round_data)


class TestDockerShellAndHelpers(unittest.TestCase):
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

    def test_has_existing_log_handles_glob_chars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proj = "proj[1]"
            fp = os.path.join(tmpdir, f"{proj}.log.2026-01-01-00-00-00-000001.json")
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "build": {"checker_non_strict": True, "checker_result": "True(strict=False)"},
                    },
                    f,
                )
            self.assertTrue(compile_script._has_existing_log(tmpdir, proj))

    def test_smoke_check_appends_linked_binaries_for_static_targets(self):
        found = {"matched_targets": ["libfoo.a"], "all_artifacts": ["libfoo.a", "demo_cli", "libfoo.so"]}

        def fake_exec(cmd, record_log=False):
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
        cfg.reset_circuit_breaker()

    def test_snapshot_and_probe_filter_exclude_fake_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            elf_bytes = b"\x7fELF" + b"\x00" * 32
            preexisting = os.path.join(tmpdir, "preexisting_tool")
            with open(preexisting, "wb") as f:
                f.write(elf_bytes)

            snap = snapshot_existing_artifacts(tmpdir)
            self.assertEqual(snap, {"preexisting_tool"})

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

    def test_evaluate_ignores_unmatched_smoke_count(self):
        records = [
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
        summary = evaluate_script.compute_summary(records)
        self.assertEqual(summary["smoke_pass_count"], 0)

    def test_local_llm_circuit_breaker_requires_two_failures_and_health_retries(self):
        cfg.reset_circuit_breaker()
        fake_cfg = {
            "provider": "ollama",
            "base_url": "http://127.0.0.1:11434/v1",
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


class TestLocalLlamaCppAndBenchmarkOptimizations(unittest.TestCase):
    def tearDown(self):
        cfg.reset_stop_bypass_cache()
        cfg.reset_circuit_breaker()

    def test_llamacpp_and_openai_compatible_auto_disable_thinking_for_qwen3_variants(self):
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "openai_compatible",
                "LLM_BASE_URL": "http://localhost:8080/v1",
                "MASTER_MODEL": "Qwen 3.8 K_M Q4 27B",
                "LLM_THINKING": "auto",
                "LLM_DISABLE_THINKING": "",
            },
            clear=False,
        ):
            resolved = cfg.resolve_llm_config()
            self.assertEqual(resolved["provider"], "openai_compatible")
            self.assertEqual(resolved["thinking"], "false")
            self.assertTrue(cfg.apply_thinking_directive("Compile this").endswith("/no_think"))
            llm = cfg.create_chat_llm()
            self.assertEqual(
                (llm.extra_body or {}).get("chat_template_kwargs"),
                {"enable_thinking": False},
            )

        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "llama.cpp",
                "LLM_BASE_URL": "http://localhost:8080/v1",
                "MASTER_MODEL": "local-default-model",
                "LLM_THINKING": "auto",
                "LLM_DISABLE_THINKING": "",
            },
            clear=False,
        ):
            resolved = cfg.resolve_llm_config()
            self.assertEqual(resolved["provider"], "llamacpp")
            self.assertEqual(resolved["thinking"], "false")
            llm = cfg.create_chat_llm()
            self.assertEqual(
                (llm.extra_body or {}).get("chat_template_kwargs"),
                {"enable_thinking": False},
            )

    def test_reasoning_safe_chat_openai_salvages_and_bypasses_stop_on_reasoning_content(self):
        from langchain_core.messages import HumanMessage

        cls = cfg._get_reasoning_safe_chat_openai_cls()
        llm = cls(base_url="http://127.0.0.1:8080/v1", model="qwen3-27b", api_key="EMPTY")

        salvage_resp = {
            "id": "chatcmpl-1",
            "model": "qwen3-27b",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "Let's inspect /work.\nAction: Shell\nAction Input: tree . -L 2",
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        res_salvaged = llm._create_chat_result(salvage_resp)
        self.assertIn("Action: Shell", res_salvaged.generations[0].message.content)
        self.assertTrue(res_salvaged.generations[0].message.additional_kwargs.get("reasoning_salvaged"))

        recorded_stops = []

        def fake_create(**payload):
            recorded_stops.append(payload.get("stop"))
            if payload.get("stop"):
                return {
                    "id": "chatcmpl-cut",
                    "model": "qwen3-27b",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "reasoning_content": "I should follow Thought/Action and wait for",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }
            return {
                "id": "chatcmpl-ok",
                "model": "qwen3-27b",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Thought: Check layout.\nAction: Shell\nAction Input: ls -la",
                            "reasoning_content": "done thinking",
                        },
                        "finish_reason": "stop",
                    }
                ],
            }

        with patch.object(llm, "client", SimpleNamespace(create=fake_create)):
            out1 = llm._generate([HumanMessage(content="compile")], stop=["\nObservation:"])
            self.assertIn("Action Input: ls -la", out1.generations[0].message.content)
            self.assertEqual(recorded_stops, [["\nObservation:"], None])

            out2 = llm._generate([HumanMessage(content="compile again")], stop=["\nObservation:"])
            self.assertIn("Action Input: ls -la", out2.generations[0].message.content)
            self.assertEqual(recorded_stops, [["\nObservation:"], None, None])

    def test_parser_empty_response_and_clean_log_and_exception_collapse(self):
        from langchain_core.exceptions import OutputParserException

        parser = SafeReActSingleInputOutputParser()
        with self.assertRaises(OutputParserException) as ctx:
            parser.parse("   ")
        self.assertEqual(ctx.exception.llm_output, "[empty response]")
        self.assertTrue(ctx.exception.send_to_llm)
        self.assertIn("Action Input:", ctx.exception.observation)

        parsed = parser.parse(
            "Thought: List files.\nAction: Shell\nAction Input: ls -la\nObservation: fake_file.c\nThought: Fake"
        )
        self.assertIsInstance(parsed, AgentAction)
        self.assertEqual(parsed.tool_input, "ls -la")
        self.assertNotIn("fake_file.c", parsed.log)

        executor = CompilationAgentExecutor.__new__(CompilationAgentExecutor)
        raw = [
            (AgentAction("_Exception", "obs1", ""), "obs1"),
            (AgentAction("_Exception", "obs2", ""), "obs2"),
            (AgentAction("_Exception", "obs3", ""), "obs3"),
            (AgentAction("_Exception", "obs4", ""), "obs4"),
        ]
        prepared = executor._prepare_intermediate_steps(raw)
        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0][0].log, "[empty response]")
        self.assertTrue(CompilationAgentExecutor._is_stuck_in_loop(raw))

    def test_normalize_make_jobs_handles_quoted_nproc_and_numbers(self):
        cmd1 = 'make -j"$(nproc)" build_libs build_apps'
        norm1 = InteractiveDockerShell._normalize_make_jobs(cmd1, max_jobs=4)
        self.assertEqual(norm1, "make -j4 build_libs build_apps")

        cmd2 = "cmake --build build -j'$(nproc)' && make -j\"16\" all"
        norm2 = InteractiveDockerShell._normalize_make_jobs(cmd2, max_jobs=4)
        self.assertEqual(norm2, "cmake --build build -j4 && make -j4 all")

    def test_has_existing_log_retry_uncertain_behavior(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loop_fp = os.path.join(tmpdir, "zstd.log.2026-11-19-02-31-01-000001.json")
            with open(loop_fp, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "autocompiler": [["Invalid Format", "Early stop: loop detected"]],
                        "build": {"model_result": "UNCERTAIN", "checker_non_strict": False, "checker_result": "False(strict=False)"},
                    },
                    f,
                )
            self.assertTrue(compile_script._has_existing_log(tmpdir, "zstd", retry_uncertain=False))
            self.assertFalse(compile_script._has_existing_log(tmpdir, "zstd", retry_uncertain=True))

            ok_fp = os.path.join(tmpdir, "openssl.log.2026-11-18-22-45-47-000001.json")
            with open(ok_fp, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "build": {"model_result": "UNCERTAIN", "checker_non_strict": True, "checker_result": "True(strict=False)"},
                    },
                    f,
                )
            self.assertTrue(compile_script._has_existing_log(tmpdir, "openssl", retry_uncertain=True))

            fail_fp = os.path.join(tmpdir, "mbedtls.log.2026-11-18-23-33-59-000001.json")
            with open(fail_fp, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "autocompiler": [["obs", "Final Answer: COMPILATION-FAIL"]],
                        "build": {"model_result": "False", "checker_non_strict": False, "checker_result": "False(strict=False)"},
                    },
                    f,
                )
            self.assertTrue(compile_script._has_existing_log(tmpdir, "mbedtls", retry_uncertain=True))


class TestCrashSafeJsonlAndAtomicPersistence(unittest.TestCase):
    def test_append_jsonl_recovers_from_power_loss_truncated_tail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_fp = os.path.join(tmpdir, RESULTS_JSONL_FILENAME)
            append_jsonl(jsonl_fp, {"project": "repo1", "timestamp": "2026-11-20-00-00-01-000001", "build": {"model_result": "True"}})

            with open(jsonl_fp, "ab") as f:
                f.write(b'{"project": "repo2", "build": {"model_re')

            append_jsonl(jsonl_fp, {"project": "repo3", "timestamp": "2026-11-20-00-00-03-000003", "build": {"model_result": "True"}})

            records = iter_jsonl_records(jsonl_fp, repair_truncated_tail=True)
            self.assertEqual([r["project"] for r in records], ["repo1", "repo3"])

            with open(jsonl_fp, "r", encoding="utf-8") as f:
                raw_lines = [line.strip() for line in f if line.strip()]
            self.assertEqual(len(raw_lines), 2)
            self.assertEqual(json.loads(raw_lines[1])["project"], "repo3")

    def test_atomic_write_and_cleanup_stale_tmp_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "report.json")
            atomic_write_json(target, {"ok": True, "count": 42})
            with open(target, "r", encoding="utf-8") as f:
                self.assertEqual(json.load(f), {"ok": True, "count": 42})
            self.assertEqual([n for n in os.listdir(tmpdir) if ".tmp." in n], [])

            stale1 = os.path.join(tmpdir, "foo.json.tmp.999.abcd1234")
            atomic_write_text(stale1, "partial")
            removed = cleanup_stale_tmp_files(tmpdir)
            self.assertEqual(removed, 1)
            self.assertFalse(os.path.exists(stale1))

    def test_save_logs_appends_results_jsonl_and_supports_jsonl_only_resume_and_evaluate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run")
            out_dir = os.path.join(tmpdir, "eval_out")
            fake_tools = [SimpleNamespace(name="Shell", description="run shell")]

            save_logs(
                run_dir,
                "demo-ok",
                "compile demo-ok",
                fake_tools,
                {"output": "Final Answer: COMPILATION-SUCCESS", "intermediate_steps": [], "run_id": "run-1"},
                {"Shell": [["make", "ok", 1.2]]},
                "/tmp/demo-ok",
                "True(strict=False)",
                "True",
                build_metadata={
                    "project": "demo-ok",
                    "ablation_mode": "full",
                    "checker_non_strict": True,
                    "checker_strict": True,
                    "elapsed_seconds": 12.5,
                    "iterations": 2,
                    "tool_calls": {"Shell": 1, "CompileNavigator": 0, "ErrorSolver": 0},
                    "matched_targets": ["demo"],
                    "smoke_check": {"checked_count": 1, "passed": True, "missing_libs": [], "details": []},
                },
                project_url="https://github.com/example/demo-ok",
            )
            save_logs(
                run_dir,
                "demo-loop",
                "compile demo-loop",
                fake_tools,
                {"output": "Early stop: loop detected", "intermediate_steps": [], "run_id": "run-2"},
                {"Shell": []},
                "/tmp/demo-loop",
                "False(strict=False)",
                "UNCERTAIN",
                build_metadata={
                    "project": "demo-loop",
                    "ablation_mode": "full",
                    "checker_non_strict": False,
                    "checker_strict": False,
                    "elapsed_seconds": 50.0,
                    "iterations": 4,
                    "tool_calls": {"Shell": 0, "CompileNavigator": 0, "ErrorSolver": 0},
                    "matched_targets": [],
                    "smoke_check": {"checked_count": 0, "passed": False, "missing_libs": [], "details": []},
                },
            )

            for fn in os.listdir(run_dir):
                if ".log." in fn and fn.endswith(".json"):
                    os.remove(os.path.join(run_dir, fn))

            jsonl_rows = iter_jsonl_records(os.path.join(run_dir, RESULTS_JSONL_FILENAME))
            self.assertEqual(len(jsonl_rows), 2)
            self.assertNotIn("security", jsonl_rows[0])

            self.assertTrue(compile_script._has_existing_log(run_dir, "demo-ok", retry_uncertain=True))
            self.assertTrue(compile_script._has_existing_log(run_dir, "demo-loop", retry_uncertain=False))
            self.assertFalse(compile_script._has_existing_log(run_dir, "demo-loop", retry_uncertain=True))

            eval_recs = evaluate_script.load_dedup_records(run_dir, "full")
            self.assertEqual({r["project"] for r in eval_recs}, {"demo-ok", "demo-loop"})

            from click.testing import CliRunner

            res = CliRunner().invoke(
                evaluate_script.main,
                ["--full-dir", run_dir, "--ablation-dir", "", "--out-dir", out_dir],
            )
            self.assertEqual(res.exit_code, 0, res.output)
            consolidated_fp = os.path.join(out_dir, "benchmark_results.json")
            self.assertTrue(os.path.isfile(consolidated_fp))
            with open(consolidated_fp, "r", encoding="utf-8") as f:
                consolidated = json.load(f)
            self.assertEqual(consolidated["full_summary"]["count"], 2)
            self.assertEqual(consolidated["full_summary"]["csr_non_strict_count"], 1)
            self.assertNotIn("security_summary", consolidated)

    def test_merge_process_csvs_recovers_crash_without_duplicates_and_backfill_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            atomic_write_text(os.path.join(tmpdir, "statistics.csv"), "id-1,,a.txt,/p/a,True,True(strict=False)\n")
            atomic_write_text(
                os.path.join(tmpdir, "12345.csv"),
                "id-1,,a.txt,/p/a,True,True(strict=False)\nid-2,,b.txt,/p/b,True,True(strict=False)\n",
            )
            merged = compile_script._merge_process_csvs(tmpdir)
            self.assertEqual(merged, 1)
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "12345.csv")))
            with open(os.path.join(tmpdir, "statistics.csv"), "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            self.assertEqual(len(lines), 2)

            legacy_fp = os.path.join(tmpdir, "zlib.log.2026-11-19-01-02-03-000123.json")
            atomic_write_json(
                legacy_fp,
                {
                    "autocompiler": [["q", "Final Answer: COMPILATION-SUCCESS"]],
                    "build": {"project": "zlib", "model_result": "True", "checker_non_strict": True},
                },
            )
            self.assertEqual(backfill_results_jsonl(tmpdir), 1)
            self.assertEqual(backfill_results_jsonl(tmpdir), 0)
            rows = iter_jsonl_records(os.path.join(tmpdir, RESULTS_JSONL_FILENAME))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["project"], "zlib")


if __name__ == "__main__":
    unittest.main()
