import ast
import json
import logging
import os
import re
import time
from typing import Optional

from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

from compile_agent.config import *
from compile_agent.agent.web_search import search_agent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_RUNS_LOG_DIR = os.path.join(REPO_ROOT, "outputs", "runs")
os.makedirs(_RUNS_LOG_DIR, exist_ok=True)
_DISCUSSION_LOG_NAME = (
    f"discussion.{os.getpid()}.log"
    if os.environ.get("AC_MULTI_WORKER") == "1"
    else "discussion.log"
)
file_handler = logging.FileHandler(
    os.path.join(_RUNS_LOG_DIR, _DISCUSSION_LOG_NAME),
    mode="a",
    encoding="utf-8",
    delay=True,
)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(name)s:%(levelname)s:%(message)s")
file_handler.setFormatter(formatter)
if not any(
    isinstance(h, logging.FileHandler)
    and os.path.basename(getattr(h, "baseFilename", "")) == _DISCUSSION_LOG_NAME
    for h in logger.handlers
):
    logger.addHandler(file_handler)

_PRIORITY_DIAG_RE = re.compile(r"(?i)(error:|fatal error:|undefined reference|cmake error)")


def _truncate_mistake_text(mistake: str, limit: int = 4000) -> str:
    """Compact oversized compiler error logs to <=limit chars while keeping key error lines and tail."""
    if not isinstance(mistake, str) or len(mistake) <= limit:
        return mistake
    diag_lines = []
    seen = set()
    diag_len = 0
    for line in mistake.splitlines():
        s = line.strip()
        if not s or not _PRIORITY_DIAG_RE.search(s) or s in seen:
            continue
        if diag_len + len(s) + 1 > 1400:
            break
        seen.add(s)
        diag_lines.append(s)
        diag_len += len(s) + 1
    tail = mistake[-2500:]
    if diag_lines:
        prefix = "[Key Error Lines]:\n" + "\n".join(diag_lines) + "\n...\n"
        remain = max(0, limit - len(prefix))
        return prefix + tail[-remain:] if remain else prefix[:limit]
    return tail[-limit:]


def trans_confidence(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.1
    if x <= 0.6:
        return 0.1
    if x < 0.8:
        return 0.3
    if x < 0.9:
        return 0.5
    if x < 1.0:
        return 0.8
    return 1.0


def parse_json(model_output):
    if isinstance(model_output, dict):
        if "reasoning" in model_output and "solution" in model_output:
            model_output.setdefault("output", json.dumps(model_output, ensure_ascii=False))
            return model_output
        raw_text = strip_thinking_tags(model_output.get("output", model_output))
    else:
        raw_text = strip_thinking_tags(model_output)

    candidates = []
    fenced_matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, flags=re.DOTALL)
    candidates.extend(fenced_matches)
    brace_match = re.search(r"(\{.*\})", raw_text, flags=re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(1))

    for candidate in candidates:
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(candidate)
                if isinstance(parsed, dict):
                    parsed.setdefault("output", raw_text)
                    return parsed
            except Exception:
                continue

        # Fallback: escape apostrophes inside single-quoted Python-dict outputs
        try:
            flat = candidate.replace("\n", " ")
            flat = re.sub(r"(\w)'(\w|\s)", r"\1\\'\2", flat)
            parsed = ast.literal_eval(flat)
            if isinstance(parsed, dict):
                parsed.setdefault("output", raw_text)
                return parsed
        except Exception:
            continue

    return "ERR_SYNTAX"


def count_common_words(str1, str2):
    set1 = set(str1.split())
    set2 = set(str2.split())
    return len(set1 & set2)


def _default_agent_keys() -> list:
    models = resolve_llm_config().get("discussion_models") or [MASTER_MODEL]
    m1 = models[0]
    m2 = models[1] if len(models) > 1 else m1
    m3 = models[2] if len(models) > 2 else m1
    return [m1, m2, m3]


def clean_output(all_results, rounds, agent_keys=None):
    if agent_keys is None:
        agent_keys = _default_agent_keys()
    output_keys = [f"{k}_output_{rounds}" for k in agent_keys]

    for result in all_results:
        for output in output_keys:
            if output in result:
                if not isinstance(result[output], dict):
                    parsed = parse_json(result[output])
                    result[output] = parsed if isinstance(parsed, dict) else {"output": str(result[output])}

                raw_field = result[output].get("output")
                if isinstance(raw_field, str) and (
                    "reasoning" not in result[output] or "solution" not in result[output]
                ):
                    parsed_inner = parse_json(raw_field)
                    if isinstance(parsed_inner, dict):
                        for k, v in parsed_inner.items():
                            result[output].setdefault(k, v)

                if "reasoning" not in result[output]:
                    result[output]["reasoning"] = ""
                elif isinstance(result[output]["reasoning"], list):
                    result[output]["reasoning"] = " ".join(str(x) for x in result[output]["reasoning"])
                else:
                    result[output]["reasoning"] = str(result[output]["reasoning"])

                if "solution" not in result[output] or not str(result[output]["solution"]).strip():
                    result[output]["solution"] = "sorry, i don't know."
                elif isinstance(result[output]["solution"], list):
                    result[output]["solution"] = " ".join(str(x) for x in result[output]["solution"])
                else:
                    result[output]["solution"] = str(result[output]["solution"])

                conf = result[output].get("confidence_level")
                if conf is None or conf == "":
                    result[output]["confidence_level"] = 0.0
                else:
                    if isinstance(conf, str) and "%" in conf:
                        try:
                            result[output]["confidence_level"] = float(conf.replace("%", "").strip()) / 100.0
                        except ValueError:
                            result[output]["confidence_level"] = 0.0
                    else:
                        try:
                            result[output]["confidence_level"] = float(conf)
                        except (TypeError, ValueError):
                            result[output]["confidence_level"] = 0.0

    return all_results


def parse_output(all_results, rounds, threshold=5, agent_keys=None):
    if agent_keys is None:
        agent_keys = _default_agent_keys()
    round_suffix = f"_output_{rounds}"

    for result in all_results:
        certainty_vote = {}
        for output in agent_keys:
            if output + round_suffix in result:
                entry = result[output + round_suffix]
                sol = entry["solution"]
                conf_weight = trans_confidence(entry["confidence_level"])
                result[f"{output}_pred_{rounds}"] = sol
                result[f"{output}_exp_{rounds}"] = (
                    f"I think the answer is {sol} because {entry['reasoning']} "
                    f"My confidence level is {entry['confidence_level']}."
                )

                if sol not in certainty_vote:
                    matched = False
                    for key in list(certainty_vote.keys()):
                        if count_common_words(key, sol) >= threshold:
                            certainty_vote[key] += conf_weight
                            matched = True
                            break
                    if not matched:
                        certainty_vote[sol] = conf_weight
                else:
                    certainty_vote[sol] += conf_weight

        ls = list(certainty_vote.keys())
        if "sorry, i don't know." in certainty_vote and len(certainty_vote) > 1:
            del certainty_vote["sorry, i don't know."]

        if not certainty_vote:
            certainty_vote["sorry, i don't know."] = 0.1

        if all(item + round_suffix in result for item in agent_keys):
            result[f"vote_{rounds}"] = [result[f"{k}_pred_{rounds}"] for k in agent_keys]
            result[f"exps_{rounds}"] = [result[f"{k}_exp_{rounds}"] for k in agent_keys]
            result[f"weighted_vote_{rounds}"] = certainty_vote
            result[f"weighted_max_{rounds}"] = max(certainty_vote, key=certainty_vote.get)
            result[f"debate_prompt_{rounds}"] = ""
            count = len(certainty_vote.keys())
            if count >= 3 or (count == 2 and len(ls) != len(certainty_vote.keys())):
                for key in certainty_vote.keys():
                    result[f"debate_prompt_{rounds}"] += f"One agent's solution: {key}.\n"
            elif count == 2 and len(ls) == len(certainty_vote.keys()):
                result[f"debate_prompt_{rounds}"] += (
                    f"Majority agents' solution: {result[f'weighted_max_{rounds}']}.\n"
                    f"One agent's solution: {min(certainty_vote, key=certainty_vote.get)}."
                )
            else:
                result[f"debate_prompt_{rounds}"] += (
                    f"All agents' solution: {result[f'weighted_max_{rounds}']}"
                )

    return all_results


class SingleAgent:
    def __init__(
        self,
        base_url=None,
        model_name=None,
        api_key=None,
        agent_key=None,
        temperature: Optional[float] = None,
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key
        self.agent_key = agent_key or model_name
        self.temperature = temperature

    def _build_llm(self):
        return create_chat_llm(
            base_url=self.base_url,
            model=self.model_name,
            api_key=self.api_key,
            temperature=self.temperature,
        )

    def create_tools(self):
        return [
            Tool(
                name="Search_Agent",
                func=lambda q: search_agent(q),
                description=search_agent.__doc__,
            )
        ]

    def generate_answer(self, template, question, project_name, chat_history=None, search_budget=None):
        tools = self.create_tools()

        def _invoke_json_chain(q_text: str) -> str:
            payload = {
                "input": apply_thinking_directive(q_text, model=self.model_name),
                "project": project_name,
            }
            if chat_history is not None:
                payload["chat_history"] = chat_history
            prompt = PromptTemplate.from_template(template=template).partial(tools="")
            chain = prompt | self._build_llm() | StrOutputParser()
            return call_with_rate_limit_retry(chain.invoke, payload)

        # Primary path: direct JSON chain (discussion templates do not contain {agent_scratchpad})
        output_text = _invoke_json_chain(question)
        parsed = parse_json(output_text)

        if parsed == "ERR_SYNTAX":
            logger.warning(
                "[!] Direct chain JSON invalid syntax from %s; retrying once...",
                self.model_name,
            )
            retry_prompt = (
                f"{question}\n\n"
                '[IMPORTANT] Output ONLY a valid JSON object with keys "reasoning", "solution", "confidence_level".'
            )
            output_text = _invoke_json_chain(retry_prompt)
            parsed = parse_json(output_text)

        # Only trigger web search supplement if initial JSON indicates uncertainty or lacks a solution
        needs_search = False
        if isinstance(parsed, dict):
            sol = str(parsed.get("solution", "") or "").strip().lower()
            try:
                conf = float(parsed.get("confidence_level") or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            low_out = str(output_text or "").lower()
            if (
                not sol
                or "sorry, i don't know" in sol
                or "i don't know" in sol
                or "need to search" in sol
                or "search_agent" in low_out
                or (chat_history is None and conf < 0.3)
            ):
                needs_search = True
        elif "search_agent" in str(output_text or "").lower():
            needs_search = True

        if needs_search and (search_budget is None or search_budget.get("remaining", 0) > 0):
            if search_budget is not None:
                search_budget["remaining"] = max(0, int(search_budget.get("remaining", 0)) - 1)
            try:
                snippet = re.sub(r"\s+", " ", str(question or "")[-600:]).strip()
                query = f"{project_name} debian linux gcc compile fix {snippet[:220]}"
                web_info = tools[0].func(query)
                if web_info and "Search failed" not in str(web_info):
                    enriched_q = (
                        f"{question}\n\nAdditional web troubleshooting context:\n{str(web_info)[:4000]}"
                    )
                    web_output = _invoke_json_chain(enriched_q)
                    web_parsed = parse_json(web_output)
                    if isinstance(web_parsed, dict):
                        parsed = web_parsed
                        output_text = web_output
            except Exception as search_err:
                if is_fatal_llm_error(search_err):
                    raise
                logger.warning("[!] ErrorSolver web search supplement failed: %s", search_err)

        if parsed == "ERR_SYNTAX":
            raise ValueError(f"Incomplete JSON format from {self.model_name}: {str(output_text)[:200]}")

        return parsed

    def debate(self, template, question, project_name, results, rounds, search_budget=None):
        for result in results:
            prev_prompt_key = f"debate_prompt_{rounds - 1}"
            out_key = f"{self.agent_key}_output_{rounds}"
            if (out_key not in result) and (prev_prompt_key in result) and len(result[prev_prompt_key]):
                chat_history = result[prev_prompt_key]
                res = self.generate_answer(
                    template=template,
                    question=question,
                    project_name=project_name,
                    chat_history=chat_history,
                    search_budget=search_budget,
                )
                result[out_key] = res
        return results


class ErrorSolver:
    def __init__(self, project_name):
        self.logger = []
        self.project_name = project_name

    def discussion(self, mistake):
        """
        Reconcile with multiple agents to solve errors encountered during compilation.
        @param mistake: the mistake in the compilation process.
        """
        try:
            mistake = _truncate_mistake_text(mistake, limit=4000)
            cfg = resolve_llm_config()
            base_url = cfg["base_url"]
            raw_models = list(cfg.get("discussion_models") or [cfg["master_model"]])
            provider = cfg["provider"]
            solver_mode = cfg.get("error_solver_mode", "auto")

            is_local_single = (provider in LOCAL_LLM_PROVIDERS) and (len(set(raw_models)) <= 1)
            use_fast = (solver_mode == "fast") or (solver_mode == "auto" and is_local_single)

            if use_fast:
                rounds = 0
                base_model = raw_models[0] if raw_models else cfg["master_model"]
                models = [base_model, base_model]
                temperatures = [0.1, 0.5]
            else:
                rounds = 1
                models = list(raw_models[:3])
                while len(models) < 3:
                    models.append(models[-1] if models else cfg["master_model"])
                if len(set(models)) == 1:
                    temperatures = [0.1, 0.4, 0.7]
                else:
                    temperatures = [None, None, None]

            agents = []
            agent_keys = []
            for idx, (model_name, temp) in enumerate(zip(models, temperatures)):
                key = model_name if model_name not in agent_keys else f"{model_name}#{idx + 1}"
                agent_keys.append(key)
                agents.append(
                    SingleAgent(
                        base_url=base_url,
                        model_name=model_name,
                        api_key=None,
                        agent_key=key,
                        temperature=temp,
                    )
                )

            search_budget = {"remaining": 1}

            # Phase 1: Initial Response Generation
            initial_predictions = {}
            transient_round0_err = None
            for agent in agents:
                try:
                    pred = agent.generate_answer(
                        template=discussion_template1,
                        question=mistake,
                        project_name=self.project_name,
                        search_budget=search_budget,
                    )
                except Exception as agent_err:
                    if is_fatal_llm_error(agent_err):
                        raise
                    if is_transient_llm_error(agent_err):
                        transient_round0_err = agent_err
                        logger.warning(
                            f"[!] Agent {agent.agent_key} hit transient LLM error in round 0 ({agent_err}); stopping round 0."
                        )
                        break
                    logger.warning(f"[!] Agent {agent.agent_key} failed in round 0: {agent_err}")
                    pred = {
                        "reasoning": f"Agent error: {agent_err}",
                        "solution": "sorry, i don't know.",
                        "confidence_level": 0.0,
                        "output": "",
                    }
                initial_predictions[f"{agent.agent_key}_output_0"] = pred

            if transient_round0_err is not None and not initial_predictions:
                raise transient_round0_err

            active_agent_keys = [k for k in agent_keys if f"{k}_output_0" in initial_predictions] or agent_keys
            active_agents = [a for a in agents if a.agent_key in active_agent_keys]
            effective_rounds = 0 if transient_round0_err is not None else rounds

            all_results = [initial_predictions]
            all_results = clean_output(all_results, rounds=0, agent_keys=active_agent_keys)
            all_results = parse_output(all_results, rounds=0, threshold=30, agent_keys=active_agent_keys)
            logger.info("[+] ------ Initial Round Discussion (mode=%s, agents=%d) ------", "fast" if use_fast else "debate", len(active_agents))

            # Phase 2: Multi-Round Discussion (skipped in fast mode where rounds=0 or after transient round-0 abort)
            last_completed_round = 0
            for rnd in range(1, effective_rounds + 1):
                logger.info(f"[-] ------ Round {rnd} Discussion ------")
                transient_round_err = False
                for agent in active_agents:
                    try:
                        all_results = agent.debate(
                            template=discussion_template2,
                            question=mistake,
                            project_name=self.project_name,
                            results=all_results,
                            rounds=rnd,
                            search_budget=search_budget,
                        )
                    except Exception as agent_err:
                        if is_fatal_llm_error(agent_err):
                            raise
                        if is_transient_llm_error(agent_err):
                            logger.warning(
                                f"[!] Agent {agent.agent_key} hit transient LLM error in round {rnd} ({agent_err}); breaking discussion early."
                            )
                            transient_round_err = True
                            break
                        logger.warning(f"[!] Agent {agent.agent_key} failed in round {rnd}: {agent_err}")
                        all_results[0][f"{agent.agent_key}_output_{rnd}"] = {
                            "reasoning": f"Agent error: {agent_err}",
                            "solution": "sorry, i don't know.",
                            "confidence_level": 0.0,
                            "output": "",
                        }
                    time.sleep(0.5)
                if transient_round_err:
                    break
                all_results = clean_output(all_results, rnd, agent_keys=active_agent_keys)
                all_results = parse_output(all_results, rnd, threshold=30, agent_keys=active_agent_keys)
                last_completed_round = rnd
                logger.info(f"[+] ------- Round {rnd} Discussion Done ------")

            self.logger.append([mistake, all_results])
            return all_results[0].get(f"weighted_max_{last_completed_round}")

        except Exception as e:
            if is_fatal_llm_error(e) or is_transient_llm_error(e):
                raise
            logger.exception("[!] ErrorSolver.discussion failed: %s", e)
            return None
