import re
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers.react_single_input import (
    FINAL_ANSWER_ACTION,
    FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE,
    MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE,
    MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
    ReActSingleInputOutputParser,
)
from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    CallbackManagerForChainRun, 
    AsyncCallbackManagerForChainRun,
)
from typing import Dict, List, Optional, Union, Iterator, Tuple, AsyncIterator
from langchain_core.exceptions import OutputParserException
from langchain.agents.tools import InvalidTool
from langchain.agents.agent import ExceptionTool
import asyncio

from Config import call_with_rate_limit_retry, strip_thinking_tags


class SafeReActSingleInputOutputParser(ReActSingleInputOutputParser):
    """ReAct parser that strips reasoning tags and preserves trailing shell quotes."""

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        cleaned_text = strip_thinking_tags(text)
        includes_answer = FINAL_ANSWER_ACTION in cleaned_text
        regex = r"Action\s*\d*\s*:[\s]*(.*?)Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        action_match = re.search(regex, cleaned_text, re.DOTALL)
        if action_match:
            if includes_answer:
                msg = f"{FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE}: {cleaned_text}"
                raise OutputParserException(msg)
            action = action_match.group(1).strip()
            tool_input = action_match.group(2).strip()
            # Drop hallucinated extra ReAct turns appended after Action Input
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
            return AgentAction(action, tool_input, cleaned_text)
        return super().parse(cleaned_text)


class CompilationAgentExecutor(AgentExecutor):
    def _iter_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Iterator[Union[AgentFinish, AgentAction, AgentStep]]:
        """Take a single step in the thought-action-observation loop.

        Override this to take control of how the agent makes and acts on choices.
        """
        try:
            intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)

            # Call the LLM to see what to do (retrying gracefully on transient RPM limits).
            output = call_with_rate_limit_retry(
                self.agent.plan,
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )
            # breakpoint()
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
                    observation = "Invalid or incomplete response"
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

        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            yield output
            return

        actions: List[AgentAction]
        if isinstance(output, AgentAction):
            actions = [output]
        else:
            actions = output
        for agent_action in actions:
            yield agent_action
        for agent_action in actions:
            yield self._perform_agent_action(
                name_to_tool_map, color_mapping, agent_action, run_manager, intermediate_steps
            )

    @staticmethod
    def _enrich_tool_input(agent_action: AgentAction, intermediate_steps: list) -> None:
        if not intermediate_steps:
            return
        raw_input = str(agent_action.tool_input or "").strip()
        if agent_action.tool in ("GET_INSTRUCTIONS", "CompileNavigator"):
            tree_obs = str(intermediate_steps[0][1])
            if len(raw_input) < 25:
                agent_action.tool_input = tree_obs
            elif "|--" not in raw_input and "`--" not in raw_input:
                agent_action.tool_input = f"{raw_input}\n\nProject directory structure:\n{tree_obs}"
        elif agent_action.tool in ("RECONCILE", "ErrorSolver"):
            last_obs = str(intermediate_steps[-1][1])
            if len(raw_input) < 25:
                agent_action.tool_input = last_obs
            else:
                tail = last_obs[-2000:]
                if tail[:120] not in raw_input:
                    agent_action.tool_input = f"{raw_input}\n\nRecent command output:\n{tail}"

    def _perform_agent_action(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        agent_action: AgentAction,
        run_manager: Optional[CallbackManagerForChainRun] = None,
        intermediate_steps: list=[]
    ) -> AgentStep:
        if run_manager:
            run_manager.on_agent_action(agent_action, color="green")
        # Otherwise we lookup the tool
        if agent_action.tool in name_to_tool_map:
            tool = name_to_tool_map[agent_action.tool]
            return_direct = tool.return_direct
            color = color_mapping[agent_action.tool]
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            if return_direct:
                tool_run_kwargs["llm_prefix"] = ""
            # We then call the tool on the tool input to get an observation
            try:
                self._enrich_tool_input(agent_action, intermediate_steps)
            except Exception:
                pass
            # breakpoint()
            
            observation = tool.run(
                agent_action.tool_input,
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
        """Take a single step in the thought-action-observation loop.

        Override this to take control of how the agent makes and acts on choices.
        """
        try:
            intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)

            # Call the LLM to see what to do.
            output = await self.agent.aplan(
                intermediate_steps,
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
                    observation = "Invalid or incomplete response"
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

        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            yield output
            return

        actions: List[AgentAction]
        if isinstance(output, AgentAction):
            actions = [output]
        else:
            actions = output
        for agent_action in actions:
            yield agent_action

        # Use asyncio.gather to run multiple tool.arun() calls concurrently
        result = await asyncio.gather(
            *[
                self._aperform_agent_action(
                    name_to_tool_map, color_mapping, agent_action, run_manager, intermediate_steps
                )
                for agent_action in actions
            ],
        )

        # TODO This could yield each result as it becomes available
        for chunk in result:
            yield chunk

    async def _aperform_agent_action(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        agent_action: AgentAction,
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
        intermediate_steps: list=[]
    ) -> AgentStep:
        if run_manager:
            await run_manager.on_agent_action(
                agent_action, verbose=self.verbose, color="green"
            )
        # Otherwise we lookup the tool
        if agent_action.tool in name_to_tool_map:
            tool = name_to_tool_map[agent_action.tool]
            return_direct = tool.return_direct
            color = color_mapping[agent_action.tool]
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            if return_direct:
                tool_run_kwargs["llm_prefix"] = ""
            # We then call the tool on the tool input to get an observation
            try:
                self._enrich_tool_input(agent_action, intermediate_steps)
            except Exception:
                pass

            observation = await tool.arun(
                agent_action.tool_input,
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