import asyncio
import os
import re
from typing import AsyncIterator, Dict, Iterator, List, Optional, Tuple, Union

from langchain.agents import AgentExecutor
from langchain.agents.agent import ExceptionTool
from langchain.agents.output_parsers.react_single_input import (
    FINAL_ANSWER_ACTION,
    FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE,
    ReActSingleInputOutputParser,
)
from langchain.agents.tools import InvalidTool
from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.exceptions import OutputParserException
from langchain_core.tools import BaseTool

from compile_agent.config import (
    call_with_llm_retry,
    resolve_llm_config,
    strip_thinking_tags,
)

_REPETITION_WARNING = (
    "\n[SYSTEM WARNING] You have already executed this exact action multiple times "
    "with the same result. Do NOT repeat it; try a different command/flag or conclude with Final Answer."
)

_FORMAT_ERROR_OBSERVATION = (
    "Format error: You MUST respond using exact ReAct headers "
    "('Thought:', followed by either 'Action:' + 'Action Input:' OR 'Final Answer:'). "
    "Do not output raw prose without headers."
)

_EARLY_STOP_LOOP_OUTPUT = (
    "COMPILATION-UNCERTAIN (Terminated early due to repeated action/parsing loop)"
)

_REACT_HEADER_BOLD_RE = re.compile(
    r"(?:^|(?<=[\r\n\s]))\*{2}\s*(Final\s+Answer|Action(?:\s*\d+)?\s+Input(?:\s*\d+)?|Action(?:\s*\d+)?|Thought)\s*(?::\s*\*{2}|\*{2}\s*:)[ \t]*",
    re.IGNORECASE,
)


def _normalize_react_headers(text: str) -> str:
    """Normalize Markdown bold wrappers around ReAct keywords (e.g., **Action**: -> Action:)."""
    if not text:
        return ""

    def _repl(match: re.Match) -> str:
        raw_label = re.sub(r"\s+", " ", match.group(1)).strip().lower()
        if raw_label == "thought":
            label = "Thought"
        elif raw_label == "final answer":
            label = "Final Answer"
        elif "input" in raw_label:
            parts = raw_label.split()
            # Preserve optional step digit after Action / Input if present
            label = " ".join(p.capitalize() for p in parts)
        else:
            label = raw_label.capitalize()
        return f"{label}: "

    return _REACT_HEADER_BOLD_RE.sub(_repl, text)


def _clean_tool_name(raw_name: str) -> str:
    """Strip backticks, bold/italic markers, and quotes around a parsed tool name."""
    name = str(raw_name or "").strip()
    for _ in range(3):
        prev = name
        for wrap in ("**", "__", "`", '"', "'"):
            if name.startswith(wrap) and name.endswith(wrap) and len(name) > len(wrap) * 2:
                name = name[len(wrap) : -len(wrap)].strip()
            elif wrap in ("`", "**") and name.startswith(wrap):
                name = name[len(wrap) :].strip()
            elif wrap in ("`", "**") and name.endswith(wrap):
                name = name[: -len(wrap)].strip()
        if name == prev:
            break
    return name


def _clean_tool_input(raw_input: str) -> str:
    """Clean trailing hallucinated ReAct turns and single wrapping quotes/backticks."""
    tool_input = str(raw_input or "").strip()
    tool_input = re.split(
        r"\n\s*(?:---\s*\n\s*)?(?:Thought|Action(?:\s*\d+)?|Observation|Final Answer)\s*:",
        tool_input,
        maxsplit=1,
    )[0].strip()
    if (
        len(tool_input) >= 2
        and tool_input.startswith('"')
        and tool_input.endswith('"')
        and '"' not in tool_input[1:-1]
    ):
        tool_input = tool_input[1:-1].strip()
    elif (
        len(tool_input) >= 2
        and tool_input.startswith("`")
        and tool_input.endswith("`")
        and not tool_input.startswith("```")
        and tool_input.count("`") == 2
    ):
        tool_input = tool_input[1:-1].strip()
    return tool_input


class SafeReActSingleInputOutputParser(ReActSingleInputOutputParser):
    """ReAct parser with reasoning-tag stripping, Markdown tolerance, and truncated-<think> recovery."""

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        raw_text = str(text or "")
        low_raw = raw_text.lower()
        has_unclosed_think = "<think>" in low_raw and "</think>" not in low_raw
        has_think_tags = "<think" in low_raw or "</think>" in low_raw

        if not raw_text.strip():
            raise OutputParserException(
                "LLM returned an empty response.",
                observation=(
                    "Your previous response was empty (likely hidden thinking or early stop on 'Observation:'). "
                    "Do NOT output 'Observation:' or empty text. Respond immediately with:\n"
                    "Thought: <1 brief sentence>\n"
                    "Action: <tool name>\n"
                    "Action Input: <command or input>\n"
                    "(or 'Thought: ...\\nFinal Answer: COMPILATION-SUCCESS|COMPILATION-FAIL|COMPILATION-UNCERTAIN')."
                ),
                llm_output="[empty response]",
                send_to_llm=True,
            )

        cleaned_text = strip_thinking_tags(raw_text)
        normalized_text = _normalize_react_headers(cleaned_text)

        includes_answer = FINAL_ANSWER_ACTION in normalized_text
        regex = r"Action\s*\d*\s*:[\s]*(.*?)Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        action_match = re.search(regex, normalized_text, re.DOTALL)
        if action_match:
            if includes_answer:
                fa_idx = normalized_text.find(FINAL_ANSWER_ACTION)
                if action_match.start() < fa_idx:
                    action = _clean_tool_name(action_match.group(1))
                    raw_action_input = re.split(
                        r"\n\s*Final Answer\s*:", action_match.group(2), maxsplit=1
                    )[0].strip()
                    tool_input = _clean_tool_input(raw_action_input)
                    thought_prefix = normalized_text[: action_match.start()].strip()
                    clean_log = (
                        f"{thought_prefix}\nAction: {action}\nAction Input: {tool_input}"
                        if thought_prefix
                        else f"Action: {action}\nAction Input: {tool_input}"
                    )
                    return AgentAction(action, tool_input, clean_log)
                final_output = re.split(
                    r"\n\s*(?:---\s*\n\s*)?(?:Thought|Action(?:\s*\d+)?|Observation)\s*:",
                    normalized_text.split(FINAL_ANSWER_ACTION)[-1],
                    maxsplit=1,
                )[0].strip()
                return AgentFinish(
                    {"output": final_output},
                    normalized_text,
                )
            action = _clean_tool_name(action_match.group(1))
            tool_input = _clean_tool_input(action_match.group(2))
            thought_prefix = normalized_text[: action_match.start()].strip()
            clean_log = (
                f"{thought_prefix}\nAction: {action}\nAction Input: {tool_input}"
                if thought_prefix
                else f"Action: {action}\nAction Input: {tool_input}"
            )
            return AgentAction(action, tool_input, clean_log)

        try:
            return super().parse(normalized_text)
        except OutputParserException as exc:
            if has_unclosed_think or (has_think_tags and not normalized_text.strip()):
                raise OutputParserException(
                    "Truncated <think> block without valid Action or Final Answer.",
                    observation=(
                        "Your previous output was truncated inside a <think> block before emitting a valid Action or Final Answer. "
                        "Do NOT write 'Observation:' inside <think>. Respond directly with 'Thought:', 'Action:', and 'Action Input:' (or 'Final Answer:')."
                    ),
                    llm_output="[truncated thinking without Action]",
                    send_to_llm=True,
                ) from exc
            raise


_STATE_MUTATING_CMD_RE = re.compile(
    r"(?:^|[;&|]\s*)(?:sudo\s+)?(?:[A-Za-z_][A-Za-z0-9_]*=(?:\"[^\"]*\"|'[^']*'|[^\s;&|]+)\s+)*"
    r"(?:apt-get|apt|dpkg|pip3?|cmake|\./configure|configure|meson|"
    r"\./autogen\.sh|autogen\.sh|autogen|autoreconf|\./bootstrap|bootstrap|"
    r"libtoolize|aclocal|autoconf|automake|cd|mkdir|rm|cp|mv|ln|sed|awk|perl|"
    r"patch|git|export|echo|printf|tee|chmod|chown|touch|wget|curl|tar|unzip)\b",
    re.IGNORECASE,
)
_STATE_MUTATING_REDIRECT_RE = re.compile(
    r"\bcat\s*(?:>>?|<<)|(?:^|[;&|]\s*)make\b[^;&|]*\b(?:clean|distclean|mrproper|install)\b",
    re.IGNORECASE,
)
_EXIT_CODE_FINGERPRINT_RE = re.compile(r"\[Exit code:\s*(-?\d+)\]")


class CompilationAgentExecutor(AgentExecutor):
    """Custom AgentExecutor providing Smart Scratchpad Compaction, Repetition Guard, and LLM retries."""

    @staticmethod
    def _compact_observation(obs: str, head_chars: int = 400, tail_chars: int = 400) -> str:
        s = str(obs or "")
        if len(s) <= head_chars + tail_chars + 80:
            return s
        exit_matches = re.findall(r"\[Exit code:\s*[^\]]+\]", s)
        exit_tag = exit_matches[-1] if exit_matches else ""
        head = s[:head_chars].rstrip()
        tail = s[-tail_chars:].lstrip()
        omitted = max(0, len(s) - len(head) - len(tail))
        compacted = f"{head}\n... [{omitted} chars compacted] ...\n{tail}"
        if exit_tag and exit_tag not in compacted:
            compacted = f"{compacted}\n{exit_tag}"
        return compacted

    @staticmethod
    def _resolve_scratchpad_max_chars() -> int:
        env_val = os.getenv("LLM_SCRATCHPAD_MAX_CHARS", "").strip()
        if env_val:
            try:
                return max(4000, int(env_val))
            except ValueError:
                pass
        try:
            ctx = resolve_llm_config().get("context_window")
            if ctx and int(ctx) > 0:
                return min(24000, max(6000, int(int(ctx) * 1.5)))
        except Exception:
            pass
        return 24000

    def _prepare_intermediate_steps(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
    ) -> List[Tuple[AgentAction, str]]:
        raw_steps = list(intermediate_steps or [])
        if not raw_steps:
            return raw_steps

        # Normalize empty _Exception logs and collapse consecutive _Exception runs in prompt copy
        # (raw_steps stays untouched for loop detection).
        steps: List[Tuple[AgentAction, str]] = []
        for act, obs in raw_steps:
            if act.tool == "_Exception":
                clean_log = str(act.log or "").strip() or "[empty response]"
                norm_act = AgentAction(act.tool, act.tool_input, clean_log)
                if steps and steps[-1][0].tool == "_Exception":
                    steps[-1] = (norm_act, obs)
                else:
                    steps.append((norm_act, obs))
            else:
                steps.append((act, obs))

        n = len(steps)
        if n == 0:
            return steps

        def _total_len(step_list: List[Tuple[AgentAction, str]]) -> int:
            return sum(len(str(a.log or "")) + len(str(obs or "")) for a, obs in step_list)

        max_chars = self._resolve_scratchpad_max_chars()
        keep_recent = 6
        if n <= keep_recent and _total_len(steps) <= max_chars:
            return steps

        compacted = list(steps)
        recent_start = max(1, n - keep_recent)
        for idx in range(1, recent_start):
            act, obs = compacted[idx]
            compacted[idx] = (act, self._compact_observation(obs, head_chars=400, tail_chars=400))

        # If recent outputs alone push context above budget, compact middle steps up to n - 2
        if _total_len(compacted) > max_chars and n >= 3:
            for idx in range(1, n - 1):
                act, obs = compacted[idx]
                compacted[idx] = (act, self._compact_observation(obs, head_chars=400, tail_chars=400))

        if _total_len(compacted) > max_chars and n >= 3:
            for idx in range(1, n - 1):
                act, obs = compacted[idx]
                compacted[idx] = (act, self._compact_observation(obs, head_chars=250, tail_chars=250))

        if _total_len(compacted) > max_chars:
            first_act, first_obs = compacted[0]
            if len(str(first_obs or "")) > 6000:
                compacted[0] = (
                    first_act,
                    self._compact_observation(first_obs, head_chars=3000, tail_chars=2000),
                )
            if n >= 2:
                last_act, last_obs = compacted[-1]
                if len(str(last_obs or "")) > 10000:
                    compacted[-1] = (
                        last_act,
                        self._compact_observation(last_obs, head_chars=4000, tail_chars=4000),
                    )

        return compacted

    def _iter_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Iterator[Union[AgentFinish, AgentAction, AgentStep]]:
        """Take a single step in the thought-action-observation loop."""
        raw_steps = list(intermediate_steps or [])
        if self._is_stuck_in_loop(raw_steps):
            yield AgentFinish(
                return_values={"output": _EARLY_STOP_LOOP_OUTPUT},
                log="Early stop: loop detected",
            )
            return

        try:
            prepared_steps = self._prepare_intermediate_steps(raw_steps)
            output = call_with_llm_retry(
                self.agent.plan,
                prepared_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )
        except OutputParserException as e:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                raise ValueError(
                    "An output parsing error occurred. "
                    "In order to pass this error back to the agent and have it try "
                    "again, pass `handle_parsing_errors=True` to the AgentExecutor. "
                    f"This is the error: {str(e)}"
                )
            text = str(e)
            if isinstance(self.handle_parsing_errors, bool):
                if e.send_to_llm:
                    observation = str(e.observation)
                    text = str(e.llm_output)
                else:
                    observation = _FORMAT_ERROR_OBSERVATION
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(e)
            else:
                raise ValueError("Got unexpected type of `handle_parsing_errors`")
            output = AgentAction("_Exception", observation, text)
            if run_manager:
                run_manager.on_agent_action(output, color="green")
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = ExceptionTool().run(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            yield AgentStep(action=output, observation=observation)
            return

        if isinstance(output, AgentFinish):
            yield output
            return

        actions: List[AgentAction] = [output] if isinstance(output, AgentAction) else output
        for agent_action in actions:
            yield agent_action
        for agent_action in actions:
            yield self._perform_agent_action(
                name_to_tool_map, color_mapping, agent_action, run_manager, raw_steps
            )

    @classmethod
    def _is_state_mutating_action(cls, tool: str, tool_input: object) -> bool:
        """Return True if action alters filesystem, environment, packages, or build config."""
        if str(tool or "").strip() != "Shell":
            return False
        cmd = _clean_tool_input(str(tool_input or "")).strip()
        if not cmd:
            return False
        return bool(_STATE_MUTATING_CMD_RE.search(cmd) or _STATE_MUTATING_REDIRECT_RE.search(cmd))

    @staticmethod
    def _obs_fingerprint(obs: str) -> str:
        """Compact fingerprint combining Exit code tag and normalized tail of observation."""
        text = str(obs or "").replace(_REPETITION_WARNING.strip(), "").replace(_REPETITION_WARNING, "").strip()
        ex_matches = _EXIT_CODE_FINGERPRINT_RE.findall(text)
        ex_tag = f"Exit:{ex_matches[-1]}" if ex_matches else ""
        tail = " ".join(text.split())[-200:]
        return f"{ex_tag}|{tail}"

    @classmethod
    def _current_epoch_steps(
        cls, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> List[Tuple[AgentAction, str]]:
        """Slice steps occurring after the most recent executed state-mutating Shell action."""
        epoch: List[Tuple[AgentAction, str]] = []
        for act, obs in intermediate_steps:
            obs_s = str(obs or "")
            if (
                act.tool != "_Exception"
                and _REPETITION_WARNING.strip() not in obs_s
                and cls._is_state_mutating_action(act.tool, act.tool_input)
            ):
                epoch = []
            else:
                epoch.append((act, obs))
        return epoch

    @classmethod
    def _is_stuck_in_loop(cls, intermediate_steps: List[Tuple[AgentAction, str]]) -> bool:
        """Detect unproductive loops using State-Epoch tracking and observation fingerprints."""
        if not intermediate_steps:
            return False
        # 1. >= 4 consecutive _Exception steps at the tail
        if len(intermediate_steps) >= 4 and all(
            act.tool == "_Exception" for act, _ in intermediate_steps[-4:]
        ):
            return True

        valid_steps = [(act, obs) for act, obs in intermediate_steps if act.tool != "_Exception"]
        if not valid_steps:
            return False

        # 2. >= 3 consecutive identical non-_Exception actions at the tail
        if len(valid_steps) >= 3:
            tail3_keys = [cls._normalize_action_key(a.tool, a.tool_input) for a, _ in valid_steps[-3:]]
            if len(set(tail3_keys)) == 1:
                return True

        # 3. >= 3 occurrences of the same action within the current state epoch
        epoch_steps = cls._current_epoch_steps(intermediate_steps)
        epoch_counts: Dict[Tuple[str, str], int] = {}
        for act, _ in epoch_steps:
            if act.tool == "_Exception":
                continue
            key = cls._normalize_action_key(act.tool, act.tool_input)
            epoch_counts[key] = epoch_counts.get(key, 0) + 1
            if epoch_counts[key] >= 3:
                return True

        # 4. >= 4 occurrences across full history where last 3 actual executions have same fingerprint
        total_counts: Dict[Tuple[str, str], int] = {}
        exec_obs_by_key: Dict[Tuple[str, str], List[str]] = {}
        for act, obs in valid_steps:
            obs_s = str(obs or "")
            key = cls._normalize_action_key(act.tool, act.tool_input)
            total_counts[key] = total_counts.get(key, 0) + 1
            if _REPETITION_WARNING.strip() not in obs_s:
                exec_obs_by_key.setdefault(key, []).append(obs_s)
        for key, cnt in total_counts.items():
            if cnt >= 4:
                exec_list = exec_obs_by_key.get(key, [])
                if len(exec_list) >= 3:
                    fps = {cls._obs_fingerprint(o) for o in exec_list[-3:]}
                    if len(fps) == 1:
                        return True

        # 5. Alternating 6-step tail [k1, k2, k1, k2, k1, k2] without state/fingerprint progress
        if len(valid_steps) >= 6:
            tail6 = valid_steps[-6:]
            k6 = [cls._normalize_action_key(a.tool, a.tool_input) for a, _ in tail6]
            if (
                k6[0] == k6[2] == k6[4]
                and k6[1] == k6[3] == k6[5]
                and k6[0] != k6[1]
            ):
                mutating = any(
                    cls._is_state_mutating_action(a.tool, a.tool_input) for a, _ in tail6[:2]
                )
                fp_k1 = {cls._obs_fingerprint(tail6[i][1]) for i in (0, 2, 4)}
                fp_k2 = {cls._obs_fingerprint(tail6[i][1]) for i in (1, 3, 5)}
                if not mutating or (len(fp_k1) <= 1 and len(fp_k2) <= 1):
                    return True

        return False

    @staticmethod
    def _normalize_action_key(tool_name: str, tool_input) -> Tuple[str, str]:
        t = str(tool_name or "").strip()
        s = str(tool_input or "").strip()
        for marker in ("\n\nProject directory structure:\n", "\n\nRecent command output:\n"):
            if marker in s:
                s = s.split(marker, 1)[0].strip()
        s = "\n".join(" ".join(line.split()) for line in s.splitlines()).strip()
        return (t, s)

    @classmethod
    def _check_repetition(
        cls,
        agent_action: AgentAction,
        intermediate_steps: List[Tuple[AgentAction, str]],
    ) -> Optional[str]:
        """Return cached observation with warning if action repeats unproductively within state epoch or across epochs."""
        if not intermediate_steps or agent_action.tool == "_Exception":
            return None
        current_key = cls._normalize_action_key(agent_action.tool, agent_action.tool_input)

        # Consecutive duplicate check across valid steps
        valid_steps = [(a, str(o or "")) for a, o in intermediate_steps if a.tool != "_Exception"]
        if valid_steps:
            last_act, last_obs = valid_steps[-1]
            if cls._normalize_action_key(last_act.tool, last_act.tool_input) == current_key:
                if _REPETITION_WARNING.strip() in last_obs:
                    return last_obs
                return f"{last_obs}{_REPETITION_WARNING}"

        # Current state-epoch repetition check (steps after last state mutation)
        epoch_steps = cls._current_epoch_steps(intermediate_steps)
        valid_epoch = [(a, str(o or "")) for a, o in epoch_steps if a.tool != "_Exception"]
        matched_epoch_obs = [
            obs
            for act, obs in valid_epoch
            if cls._normalize_action_key(act.tool, act.tool_input) == current_key
        ]
        if valid_epoch:
            last_epoch_key = cls._normalize_action_key(valid_epoch[-1][0].tool, valid_epoch[-1][0].tool_input)
            if last_epoch_key == current_key:
                last_obs = matched_epoch_obs[-1] if matched_epoch_obs else valid_epoch[-1][1]
                return last_obs if _REPETITION_WARNING.strip() in last_obs else f"{last_obs}{_REPETITION_WARNING}"
            if len(matched_epoch_obs) >= 2:
                fps = [cls._obs_fingerprint(o) for o in matched_epoch_obs]
                if len(set(fps[-2:])) == 1:
                    last_obs = matched_epoch_obs[-1]
                    return last_obs if _REPETITION_WARNING.strip() in last_obs else f"{last_obs}{_REPETITION_WARNING}"

        # Cross-epoch futility check: same key executed >= 3 times with identical error fingerprint
        all_execs_for_key = [
            obs
            for act, obs in valid_steps
            if cls._normalize_action_key(act.tool, act.tool_input) == current_key
            and _REPETITION_WARNING.strip() not in obs
        ]
        if len(all_execs_for_key) >= 3:
            last3_fps = {cls._obs_fingerprint(o) for o in all_execs_for_key[-3:]}
            if len(last3_fps) == 1:
                last_obs = all_execs_for_key[-1]
                return f"{last_obs}{_REPETITION_WARNING}"

        return None

    @staticmethod
    def _build_enriched_tool_input(agent_action: AgentAction, intermediate_steps: list) -> str:
        raw_input = str(agent_action.tool_input or "").strip()
        if not intermediate_steps:
            return agent_action.tool_input
        if agent_action.tool in ("GET_INSTRUCTIONS", "CompileNavigator"):
            tree_obs = ""
            first_valid_obs = ""
            for act, obs in intermediate_steps:
                if act.tool == "_Exception":
                    continue
                if not first_valid_obs:
                    first_valid_obs = str(obs)
                if act.tool == "Shell" and re.search(r"\b(?:tree|ls)\b", str(act.tool_input or "")):
                    tree_obs = str(obs)
                    break
            if not tree_obs:
                tree_obs = first_valid_obs or str(intermediate_steps[0][1])
            if len(tree_obs) > 3500:
                tree_obs = CompilationAgentExecutor._compact_observation(
                    tree_obs, head_chars=2200, tail_chars=1000
                )
            if len(raw_input) < 25:
                return tree_obs
            if "|--" not in raw_input and "`--" not in raw_input:
                return f"{raw_input}\n\nProject directory structure:\n{tree_obs}"
        elif agent_action.tool in ("RECONCILE", "ErrorSolver"):
            last_obs = ""
            for act, obs in reversed(intermediate_steps):
                if act.tool != "_Exception":
                    last_obs = str(obs)
                    break
            if not last_obs:
                last_obs = str(intermediate_steps[-1][1])
            if len(raw_input) < 25:
                return last_obs
            tail = last_obs[-2000:]
            if tail[:120] not in raw_input:
                return f"{raw_input}\n\nRecent command output:\n{tail}"
        return agent_action.tool_input

    @staticmethod
    def _enrich_tool_input(agent_action: AgentAction, intermediate_steps: list) -> None:
        agent_action.tool_input = CompilationAgentExecutor._build_enriched_tool_input(
            agent_action, intermediate_steps
        )

    def _perform_agent_action(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        agent_action: AgentAction,
        run_manager: Optional[CallbackManagerForChainRun] = None,
        intermediate_steps: list = None,
    ) -> AgentStep:
        steps_list = intermediate_steps or []
        if run_manager:
            run_manager.on_agent_action(agent_action, color="green")
        if agent_action.tool in name_to_tool_map:
            repeated_obs = self._check_repetition(agent_action, steps_list)
            if repeated_obs is not None:
                return AgentStep(action=agent_action, observation=repeated_obs)

            tool = name_to_tool_map[agent_action.tool]
            return_direct = tool.return_direct
            color = color_mapping[agent_action.tool]
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            if return_direct:
                tool_run_kwargs["llm_prefix"] = ""
            try:
                exec_input = self._build_enriched_tool_input(agent_action, steps_list)
            except Exception:
                exec_input = agent_action.tool_input

            observation = tool.run(
                exec_input,
                verbose=self.verbose,
                color=color,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
        else:
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = InvalidTool().run(
                {
                    "requested_tool_name": agent_action.tool,
                    "available_tool_names": list(name_to_tool_map.keys()),
                },
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
        return AgentStep(action=agent_action, observation=observation)

    async def _aiter_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> AsyncIterator[Union[AgentFinish, AgentAction, AgentStep]]:
        """Take a single async step in the thought-action-observation loop."""
        raw_steps = list(intermediate_steps or [])
        if self._is_stuck_in_loop(raw_steps):
            yield AgentFinish(
                return_values={"output": _EARLY_STOP_LOOP_OUTPUT},
                log="Early stop: loop detected",
            )
            return

        try:
            prepared_steps = self._prepare_intermediate_steps(raw_steps)
            output = await self.agent.aplan(
                prepared_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )
        except OutputParserException as e:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                raise ValueError(
                    "An output parsing error occurred. "
                    "In order to pass this error back to the agent and have it try "
                    "again, pass `handle_parsing_errors=True` to the AgentExecutor. "
                    f"This is the error: {str(e)}"
                )
            text = str(e)
            if isinstance(self.handle_parsing_errors, bool):
                if e.send_to_llm:
                    observation = str(e.observation)
                    text = str(e.llm_output)
                else:
                    observation = _FORMAT_ERROR_OBSERVATION
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(e)
            else:
                raise ValueError("Got unexpected type of `handle_parsing_errors`")
            output = AgentAction("_Exception", observation, text)
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = await ExceptionTool().arun(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            yield AgentStep(action=output, observation=observation)
            return

        if isinstance(output, AgentFinish):
            yield output
            return

        actions: List[AgentAction] = [output] if isinstance(output, AgentAction) else output
        for agent_action in actions:
            yield agent_action

        result = await asyncio.gather(
            *[
                self._aperform_agent_action(
                    name_to_tool_map, color_mapping, agent_action, run_manager, raw_steps
                )
                for agent_action in actions
            ],
        )
        for chunk in result:
            yield chunk

    async def _aperform_agent_action(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        agent_action: AgentAction,
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
        intermediate_steps: list = None,
    ) -> AgentStep:
        steps_list = intermediate_steps or []
        if run_manager:
            await run_manager.on_agent_action(
                agent_action, verbose=self.verbose, color="green"
            )
        if agent_action.tool in name_to_tool_map:
            repeated_obs = self._check_repetition(agent_action, steps_list)
            if repeated_obs is not None:
                return AgentStep(action=agent_action, observation=repeated_obs)

            tool = name_to_tool_map[agent_action.tool]
            return_direct = tool.return_direct
            color = color_mapping[agent_action.tool]
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            if return_direct:
                tool_run_kwargs["llm_prefix"] = ""
            try:
                exec_input = self._build_enriched_tool_input(agent_action, steps_list)
            except Exception:
                exec_input = agent_action.tool_input

            observation = await tool.arun(
                exec_input,
                verbose=self.verbose,
                color=color,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
        else:
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = await InvalidTool().arun(
                {
                    "requested_tool_name": agent_action.tool,
                    "available_tool_names": list(name_to_tool_map.keys()),
                },
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
        return AgentStep(action=agent_action, observation=observation)
