import ast
import json
import logging
import os
import re
import time

from langchain.agents import AgentExecutor, Tool
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.tools.render import render_text_description
from langchain_core.utils.function_calling import convert_to_openai_tool
from Config import *
from GoogleSearch import search_agent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.makedirs("test", exist_ok=True)
file_handler = logging.FileHandler("test/MultiAgentDiscussion.log", mode="a", encoding="utf-8")
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(name)s:%(levelname)s:%(message)s")
file_handler.setFormatter(formatter)
if not any(
    isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "").endswith("MultiAgentDiscussion.log")
    for h in logger.handlers
):
    logger.addHandler(file_handler)

llm1 = DISCUSSION_MODELS[0]
llm2 = DISCUSSION_MODELS[1] if len(DISCUSSION_MODELS) > 1 else DISCUSSION_MODELS[0]
llm3 = DISCUSSION_MODELS[2] if len(DISCUSSION_MODELS) > 2 else DISCUSSION_MODELS[0]


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

    # Try fenced JSON block first, then outermost {...} block
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


def clean_output(all_results, rounds, agent_keys=None):
    if agent_keys is None:
        agent_keys = [llm1, llm2, llm3]
    output_keys = [f"{k}_output_{rounds}" for k in agent_keys]

    for result in all_results:
        for output in output_keys:
            if output in result:
                if not isinstance(result[output], dict):
                    parsed = parse_json(result[output])
                    result[output] = parsed if isinstance(parsed, dict) else {"output": str(result[output])}

                raw_field = result[output].get("output")
                if isinstance(raw_field, str) and ("reasoning" not in result[output] or "solution" not in result[output]):
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
        agent_keys = [llm1, llm2, llm3]
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
    def __init__(self, base_url, model_name, api_key, agent_key=None):
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key
        self.agent_key = agent_key or model_name

    def _build_llm(self):
        return create_chat_llm(
            base_url=self.base_url,
            model=self.model_name,
            api_key=self.api_key,
        )

    def create_agent(self, template_str):
        prompt = PromptTemplate.from_template(template=template_str)
        tools = self.create_tools()
        llm = self._build_llm()
        prompt = prompt.partial(tools=render_text_description(tools))
        llm_with_tools = llm.bind(tools=[convert_to_openai_tool(t) for t in tools])
        agent = prompt | llm_with_tools | OpenAIToolsAgentOutputParser()
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=5,
            max_execution_time=max(60, LLM_TIMEOUT),
        )

    def create_tools(self):
        return [
            Tool(
                name="Search_Agent",
                func=search_agent,
                description=search_agent.__doc__,
            )
        ]

    def generate_answer(self, template, question, project_name, chat_history=None):
        payload = {"input": question, "project": project_name}
        if chat_history is not None:
            payload["chat_history"] = chat_history

        tools = self.create_tools()
        tool_text = render_text_description(tools)

        # Primary path: AgentExecutor with OpenAI tool binding; fallback to direct JSON chain
        output_text = ""
        try:
            agent = self.create_agent(template_str=template)
            time.sleep(1)
            output = call_with_rate_limit_retry(agent.invoke, payload)
            output_text = output.get("output", "") if isinstance(output, dict) else str(output)
        except Exception as agent_err:
            logger.warning(
                f"[!] AgentExecutor fallback to direct chain for model={self.model_name}: {agent_err}"
            )
            prompt = PromptTemplate.from_template(template=template).partial(tools=tool_text)
            chain = prompt | self._build_llm() | StrOutputParser()
            output_text = call_with_rate_limit_retry(chain.invoke, payload)

        parsed = parse_json(output_text)
        if parsed == "ERR_SYNTAX":
            prompt = PromptTemplate.from_template(template=template).partial(tools=tool_text)
            chain = prompt | self._build_llm() | StrOutputParser()
            direct_text = ""
            for attempt in range(2):
                direct_text = call_with_rate_limit_retry(chain.invoke, payload)
                parsed = parse_json(direct_text)
                if parsed != "ERR_SYNTAX":
                    break
                logger.warning(
                    f"[!] Direct chain JSON retry {attempt + 1}/2 invalid syntax from {self.model_name}: {direct_text[:150]}"
                )
            else:
                raise ValueError(f"Incomplete JSON format from {self.model_name}: {direct_text[:200]}")

        return parsed

    def debate(self, template, question, project_name, results, rounds):
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
            rounds = 1
            models = list(DISCUSSION_MODELS[:3])
            while len(models) < 3:
                models.append(models[-1] if models else MASTER_MODEL)

            agents = []
            agent_keys = []
            for idx, model_name in enumerate(models):
                key = model_name if model_name not in agent_keys else f"{model_name}#{idx + 1}"
                agent_keys.append(key)
                agents.append(
                    SingleAgent(
                        base_url=LLM_BASE_URL,
                        model_name=model_name,
                        api_key=LLM_API_KEY,
                        agent_key=key,
                    )
                )

            # Phase 1: Initial Response Generation
            initial_predictions = {}
            for agent in agents:
                try:
                    pred = agent.generate_answer(
                        template=discussion_template1,
                        question=mistake,
                        project_name=self.project_name,
                    )
                except Exception as agent_err:
                    logger.warning(f"[!] Agent {agent.agent_key} failed in round 0: {agent_err}")
                    pred = {
                        "reasoning": f"Agent error: {agent_err}",
                        "solution": "sorry, i don't know.",
                        "confidence_level": 0.0,
                        "output": "",
                    }
                initial_predictions[f"{agent.agent_key}_output_0"] = pred

            all_results = [initial_predictions]
            all_results = clean_output(all_results, rounds=0, agent_keys=agent_keys)
            all_results = parse_output(all_results, rounds=0, threshold=30, agent_keys=agent_keys)
            logger.info("[+] ------ Initial Round Discussion ------")

            # Phase 2: Multi-Round Discussion
            for rnd in range(1, rounds + 1):
                logger.info(f"[-] ------ Round {rnd} Discussion ------")
                for agent in agents:
                    try:
                        all_results = agent.debate(
                            template=discussion_template2,
                            question=mistake,
                            project_name=self.project_name,
                            results=all_results,
                            rounds=rnd,
                        )
                    except Exception as agent_err:
                        logger.warning(f"[!] Agent {agent.agent_key} failed in round {rnd}: {agent_err}")
                        all_results[0][f"{agent.agent_key}_output_{rnd}"] = {
                            "reasoning": f"Agent error: {agent_err}",
                            "solution": "sorry, i don't know.",
                            "confidence_level": 0.0,
                            "output": "",
                        }
                    time.sleep(0.5)
                all_results = clean_output(all_results, rnd, agent_keys=agent_keys)
                all_results = parse_output(all_results, rnd, threshold=30, agent_keys=agent_keys)
                logger.info(f"[+] ------- Round {rnd} Discussion Done ------")

            self.logger.append([mistake, all_results])
            return all_results[0].get(f"weighted_max_{rounds}")

        except Exception as e:
            logger.exception("[!] ErrorSolver.discussion failed: %s", e)
            return None
