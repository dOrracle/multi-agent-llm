import asyncio
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from typing import Any, Callable, Generic, List, Optional, Type, TypeVar, Dict

import nest_asyncio
from pydantic import BaseModel, Field

from ..agent_class import Agent
from ..llm import LLMBase
from .base import DiscussionResult

T = TypeVar("T")

nest_asyncio.apply()


class BlockingFuture:
    def __init__(self, future: Future, event: asyncio.Event):
        self._future = future
        self._event = event

    async def _wait_for_event(self, timeout=None):
        """Asynchronous helper to wait for the event."""
        if timeout is not None:
            await asyncio.wait_for(self._event.wait(), timeout)
        else:
            await self._event.wait()

    def result(self, timeout=None):
        """Block and wait for the future to be ready."""
        # Get the current event loop
        loop = asyncio.get_event_loop()

        # Schedule the _wait_for_event coroutine and block until it completes
        loop.run_until_complete(self._wait_for_event(timeout))

        if not self._future.done():
            raise TimeoutError("Operation timed out waiting for the result.")

        return self._future.result()


class ToolRequest(BaseModel):
    name: str = Field(
        ...,
        description="Name of the tool to be used to help the LLM with the query"
    )
    description: str = Field(
        ...,
        description=(
            "A complete description of the desired tool. "
            "This description should be detailed enough to suffice as input for a "
            "code-generator agent who will create a Python function that implements the tool."
        )
    )
    input: Optional[Dict[str, str | int | float | bool | list | dict]] = Field(
        None,
        description="Input data to be used with the generated tool to help the LLM with the query"
    )


class ToolResponse(BaseModel):
    code: str = Field(
        ...,
        description=(
            "Python code that implements and evaluates the tool, printing its output to stdout. "
            "For example, this might be a series of imports, followed by a function definition, "
            "and then a call to that function with the provided input data. The function's output "
            "should be printed to stdout. This output should be a JSON-serializable object at "
            "can be passed back to an LLM agent for further processing."
        )
    )
    pip_dependencies: Optional[List[str]] = Field(
        None,
        description="List of pip dependencies required to run the tool"
    )


class BrainIteration(BaseModel):
    self_thought: str = Field(
        ...,
        description=(
            "Guide the LLM with instructions on how to approach the query for the "
            "current iteration based on history."
        ),
    )
    iteration_stop: bool = Field(
        ...,
        description=(
            "False for continue, True to stop the iteration as LLM has "
            "given the final confident answer for the query"
        ),
    )
    tool_request: Optional[ToolRequest] = Field(
        ...,
        description=(
            "Optional request to utilize a tool with a specific input "
            "to help the LLM with the query"
        )
    )


class ConversationTurn(BaseModel):
    iteration: int
    brain_thought: str
    llm_response: Any
    is_final: bool


class LLMResponseIteration(BaseModel):
    response: str = Field(
        ...,
        description="Response to the Inner Dialogue Agent's discussion for the current iteration",
    )


tool_gen_agent = Agent(
    name="Tool Generator Agent",

    role="""\
You are responsible for creating tools to assist the Inner Dialogue Agent in guiding \
the LLM Agent.""",

    function="""\
Generate Python code for tools based on requests from the Inner Dialogue Agent. Use the provided \
tool name, description, and input data to design a functional Python implementation. Ensure the \
created function can process the specified input accurately. \

All code you generated should be directly runnable and should serve the desired purpose.

DO NOT DO any of the following:
- Use mock data to circumvent errors unless explicitly requested.
- Produce example code with placeholder values.
- Avoid producing code that requires external authentication or API keys.
- Avoid producing code that generates images or plots
"""
)

dialogue_agent = Agent(
    name="Inner Dialogue Agent",

    role="""\
You are a guide responsible for ensuring the LLM Agent understands and solves complex queries. \
Your role is to provide domain-specific knowledge, correct reasoning errors, and encourage \
iteration until an accurate solution is reached. You may request tools when necessary to \
aid the LLM Agent in reasoning or accessing external information.""",

    function="""\
Facilitate accurate query resolution by guiding the LLM Agent with targeted prompts. \
Identify reasoning errors or knowledge gaps and provide corrective insights. \
Iterate step-by-step, refining the Agent's understanding and ensuring responses build \
toward a comprehensive solution. When necessary, request tools to perform calculations, \
fetch real-time data, or verify results, balancing efficiency with effectiveness.

Key functions include:
- Supplying domain-specific context and techniques.
- Encouraging iteration without rushing to conclusions.
- Generating structured prompts that address reasoning gaps or misconceptions.
- Requesting tools when the LLM Agent requires external support.

Conclude the process by stopping iterations when the solution is complete and accurate."""
)

llm_agent = Agent(
    name="LLM",

    role="""\
You are a language model designed to collaborate with the Inner Dialogue Agent to solve complex \
questions. Your role is to leverage your knowledge base and reasoning skills to provide \
well-reasoned, accurate, and insightful answers. Clearly indicate any uncertainties or \
knowledge gaps, and support iterative refinement until the query is resolved.""",

    function="""\
Process prompts from the Inner Dialogue Agent by retrieving relevant knowledge and applying logical \
reasoning. Highlight uncertainties or gaps in reasoning, and collaborate iteratively to refine \
responses. Provide clear, detailed, and accurate answers, ensuring each response builds on the \
previous one. Avoid premature conclusions, encouraging iteration to achieve a comprehensive \
solution. When reaching maximum iterations, deliver a final definitive answer, selecting the \
most accurate option if applicable."""
)


class AIOT(Generic[T]):
    def __init__(
        self,
        llm: LLMBase,
        iterations: int = 5,
        tool_attempts: int = 3,
        answer_schema: Optional[Type[T]] = None,
        tool_runner: Optional[Callable[[str, Optional[List[str]]], str]] = None,
        interactive: bool = False,
    ):
        self.llm = llm
        self.max_iterations = iterations
        self.tool_attempts = tool_attempts
        self.answer_schema = answer_schema or str
        self.tool_runner = tool_runner
        self.interactive = interactive

        self.dialogue_agent = dialogue_agent
        self.llm_agent = llm_agent
        self.tool_gen_agent = tool_gen_agent

        self._loop = asyncio.get_event_loop()
        self._executor = ThreadPoolExecutor(max_workers=1)

    def get_llm_schema(self):
        class LLMResponse(BaseModel):
            response: self.answer_schema = Field(
                ...,
                description="The response generated by the LLM to the brain's prompt",
            )
            answer_to_query: bool = Field(
                ...,
                description="Does the response contain the final answer to the query? True if it does, False if it does not",
            )

        return LLMResponse

    def _create_context(self, query: str):
        return {
            "query": query,
            "prompt_history": f"**Important** Initial Query: {query}\n\n",
            "conversation": [],
            "tool_outputs": {},
            "tool_errors": {},
        }

    async def run_async(self, query: str) -> DiscussionResult[T]:
        context = self._create_context(query)
        return await self._run_async(context)

    def run(self, query: str) -> DiscussionResult[T]:
        """
        Blocking method to run the AIOT discussion.
        Can be called from Jupyter notebooks or any synchronous context.
        """

        def run_async_in_new_loop():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(self.run_async(query))

        return self._executor.submit(run_async_in_new_loop).result()

    async def _run_async(self, context: dict) -> DiscussionResult[T]:
        iteration = 1
        completed = False
        answer_to_query = False
        llm_ans = None

        while (
            iteration <= self.max_iterations
            and not completed and not answer_to_query
        ):
            brain_ans = await self._brain_iteration(context, iteration)
            if brain_ans is None:
                print("Brain iteration failed. Ending discussion.")
                break

            tool_attempts = 0
            while (
                self.tool_runner is not None
                and brain_ans.tool_request is not None
                and tool_attempts < self.tool_attempts
            ):
                # Enter tool evaluation loop.
                print(
                    "💡 Received tool request from the brain.",
                    brain_ans.tool_request.model_dump()
                )
                tool_response = await self._tool_iteration(brain_ans.tool_request, context)
                print(
                    "💡💡💡 Current tool context:",
                    context.get("tool_outputs", "No previous tool requests"),
                )

                print("💡 Tool Generated:\n", tool_response.code)

                # TODO: use async function here?
                tool_response_content = self._get_tool_output(tool_response)
                tool_name = brain_ans.tool_request.name

                if tool_response_content.startswith("ERROR"):
                    context["tool_errors"][tool_name] = {
                        "code": tool_response.code,
                        "error": tool_response_content,
                    }
                else:
                    context["tool_outputs"][tool_name] = tool_response_content

                print("💡 Added tool response context: ", tool_response_content)

                brain_ans = await self._brain_iteration(context, iteration)
                tool_attempts += 1

            completed = brain_ans.iteration_stop

            llm_ans = await self._llm_iteration(context, brain_ans.self_thought)
            if llm_ans is None:
                print("LLM iteration failed. Ending discussion.")
                break

            answer_to_query = llm_ans.answer_to_query

            context["conversation"].append(
                ConversationTurn(
                    iteration=iteration,
                    brain_thought=brain_ans.self_thought,
                    llm_response=llm_ans.response,
                    is_final=completed or answer_to_query,
                )
            )

            new_history = (
                f"Inner Dialogue Agent: {brain_ans.self_thought}\n"
                f"LLM answer: {llm_ans.response}\n\n"
            )
            context["prompt_history"] += new_history
            iteration += 1

        return DiscussionResult(
            query=context["query"],
            thoughts=context["conversation"],
            answer=llm_ans.response if llm_ans else "Unknown",
        )

    async def _brain_iteration(
        self, context: dict, iteration: int
    ) -> Optional[BrainIteration]:
        prompt_with_history = f"""{context["prompt_history"]}

Current Iteration: {iteration}

Tool outputs: {context.get("tool_outputs", "None")}

Make the LLM answer within maximum of {self.max_iterations} iterations.

Ideate first with LLM and guide the LLM towards the answer, considering the remaining iterations.

Talk with and prompt LLM in second person directly as if you are discussing with the LLM to guide \
it towards the answer."""

        system_prompt, user_prompt = self.dialogue_agent.prompt(
            prompt_with_history)
        formatted_prompt = self.llm.format_prompt(system_prompt, user_prompt)
        return await self.llm.generate_async(formatted_prompt, BrainIteration)

    async def _llm_iteration(self, context: dict, brain_thought: str):
        prompt_with_history = (
            f"{context['prompt_history']}\nInner cognitive brain: {brain_thought}\n"
            f"Based on the above discussion with the brain in mind.\n"
            f"Respond to the brain's prompt for the query, indicating if it's the final correct answer to the query.\n"
            f"If you are unsure, please iterate with the brain. Make sure you answer within maximum of {self.max_iterations} iterations\n\n"
            f"Original query: {context['query']}\n"
        )
        system_prompt, user_prompt = self.llm_agent.prompt(prompt_with_history)
        formatted_prompt = self.llm.format_prompt(system_prompt, user_prompt)
        return await self.llm.generate_async(formatted_prompt, self.get_llm_schema())

    async def _tool_iteration(self, tool_request: ToolRequest, context: dict):
        tool_gen_prompt = (
            "Generate a tool with the following details:\n"
            f"Name: {tool_request.name}\n"
            f"Description: {tool_request.description}\n"
            f"Input: {tool_request.input}\n\n"
            "Consider previous outcomes below, regarding previous tool requests:\n"
            f"{context.get('tool_outputs', 'No previous tool requests')}\n"
        )
        system_prompt, user_prompt = self.tool_gen_agent.prompt(
            tool_gen_prompt)
        formatted_prompt = self.llm.format_prompt(system_prompt, user_prompt)
        return await self.llm.generate_async(formatted_prompt, ToolResponse)

    def _get_tool_output(self, tool_response: ToolResponse) -> str:
        if self.tool_runner is None:
            raise ValueError("Tool runner function not provided.")

        tool_response_content = self.tool_runner(
            tool_response.code,
            tool_response.pip_dependencies
        )
        return tool_response_content


class GIOT(Generic[T]):
    def __init__(
        self,
        llm: LLMBase,
        iterations: int = 5,
        answer_schema: Optional[Type[T]] = None,
    ):
        self.llm = llm
        self.total_iterations = iterations
        self.answer_schema = answer_schema or str
        self.dialogue_agent = self._create_dialogue_agent()
        self.llm_agent = self._create_llm_agent()
        self._loop = asyncio.get_event_loop()
        self._executor = ThreadPoolExecutor(max_workers=1)

    def get_final_llm_schema(self):
        class LLMFinalResponse(BaseModel):
            response: self.answer_schema = Field(
                ..., description="Final answer to the query"
            )
            explanation: str = Field(
                ..., description="Explanation for the chosen classification"
            )

        return LLMFinalResponse

    def _create_dialogue_agent(self):
        return dialogue_agent

    def _create_llm_agent(self):
        return llm_agent

    def _create_context(self, query: str):
        return {
            "query": query,
            "prompt_history": f"**Important** Initial Query: {query}\n\n",
            "conversation": [],
        }

    def run(self, query: str) -> DiscussionResult[T]:
        """
        Blocking method to run the AIOT discussion.
        Can be called from Jupyter notebooks or any synchronous context.
        """

        def run_async_in_new_loop():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(self.run_async(query))

        return self._executor.submit(run_async_in_new_loop).result()

    async def run_async(self, query: str) -> DiscussionResult[T]:
        context = self._create_context(query)
        return await self._run_async(context)

    async def _run_async(self, context: dict) -> DiscussionResult[T]:
        llm_ans = None
        for current_iteration in range(1, self.total_iterations + 1):
            brain_ans = await self._brain_iteration(context, current_iteration)
            if brain_ans is None:
                print(
                    f"Brain iteration {current_iteration} failed. Ending discussion.")
                break

            llm_ans = await self._llm_iteration(
                context, brain_ans.self_thought, current_iteration
            )
            if llm_ans is None:
                print(
                    f"LLM iteration {current_iteration} failed. Ending discussion.")
                break

            context["conversation"].append(
                ConversationTurn(
                    iteration=current_iteration,
                    brain_thought=brain_ans.self_thought,
                    llm_response=llm_ans.response,
                    is_final=False,
                )
            )

            context[
                "prompt_history"
            ] += f"Iteration {current_iteration}/{self.total_iterations}:\nInner Dialogue Agent: {brain_ans.self_thought}\nLLM answer: {llm_ans.response}\n\n"

        final_answer = await self._llm_final_iteration(context)

        return DiscussionResult(
            query=context["query"],
            thoughts=context["conversation"],
            answer=final_answer.response if final_answer else "Unknown",
        )

    async def _brain_iteration(
        self, context: dict, current_iteration: int
    ) -> Optional[BrainIteration]:
        prompt_with_history = (
            f"{context['prompt_history']}\n"
            f"Current Iteration: {current_iteration}/{self.total_iterations}\n"
            f"Guide the LLM towards the answer, considering the remaining iterations.\n\n"
            f"Chat and prompt LLM directly as if you are discussing with the LLM to guide it towards the answer.\n"
            f"Original query: {context['query']}\n"
        )
        system_prompt, user_prompt = self.dialogue_agent.prompt(
            prompt_with_history)
        formatted_prompt = self.llm.format_prompt(system_prompt, user_prompt)
        return await self.llm.generate_async(formatted_prompt, BrainIteration)

    async def _llm_iteration(
        self, context: dict, brain_thought: str, current_iteration: int
    ) -> Optional[LLMResponseIteration]:
        prompt_with_history = (
            f"{context['prompt_history']}\n"
            f"Inner cognitive brain: {brain_thought}\n"
            f"Current Iteration: {current_iteration}/{self.total_iterations}\n"
            f"Discuss further with the brain to arrive at an answer. Do not provide a final answer yet.\n"
            f"Original query: {context['query']}\n"
        )
        system_prompt, user_prompt = self.llm_agent.prompt(prompt_with_history)
        formatted_prompt = self.llm.format_prompt(system_prompt, user_prompt)
        return await self.llm.generate_async(formatted_prompt, LLMResponseIteration)

    async def _llm_final_iteration(self, context: dict):
        prompt_with_history = (
            f"{context['prompt_history']}\n"
            f"You are in final Iteration: Based on the above discussion between you and brain, provide your final answer to the query.\n"
            f"Original query: {context['query']}\n"
        )
        system_prompt, user_prompt = self.llm_agent.prompt(prompt_with_history)
        formatted_prompt = self.llm.format_prompt(system_prompt, user_prompt)
        return await self.llm.generate_async(
            formatted_prompt, self.get_final_llm_schema()
        )
