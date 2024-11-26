import asyncio
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError, wait
from typing import Any, Generic, List, Optional, Type, TypeVar, Callable

import nest_asyncio
from pydantic import BaseModel, Field

from ..agent_class import Agent
from ..llm import LLMBase
from .base import DiscussionResult, MultiAgentBase

T = TypeVar("T")

nest_asyncio.apply()


from concurrent.futures import Future, TimeoutError


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
    input: Any = Field(
        ...,
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
        None,
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
        description="Response to the inner cognitive brain's discussion for the current iteration",
    )


tool_gen_agent = Agent(
    name="Tool Generator Agent",

    role="""\
You are a Tool Generator Agent responsible for creating tools that can assist the \
Cognitive Reflection Agent in guiding the LLM Agent.""",

    function="""\
Generate Python code that implements a tool based on the Cognitive Reflection Agent's request. \
Consider the tool's name, description, and desired inputs to create the Python function. \
Ensure that the function you create can accept the input data provided."""
)

brain_agent = Agent(
    name="Cognitive Reflection Agent",

    role="""\
You are an internal guide responsible for ensuring another LLM Agent thoroughly \
understands and solves complex questions. Your primary task is to bring forth all \
relevant domain-specific knowledge necessary for the LLM Agent to address the query accurately.

You also have the option to request a tool to be used to help with the above. \
Any tool requests will be forwarded to a Tool Generator agent who will implement and evaluate \
the tool dynamically.

At each step of the reasoning process, you will provide the LLM Agent with targeted prompts \
that correct any misconceptions, reinforce correct thinking, and introduce essential \
knowledge that it may be overlooking. When the LLM Agent struggles or deviates, you ensure it has \
access to the precise information needed to think through the problem effectively.

Do not provide factually wrong insights to the LLM Agent. If you are unsure and not confident of \
the answer/solution, always iterate until the LLM Agent reaches maximum iterations. \
Your guidance prompt should be detailed and informative. \
Always encourage iterating with the LLM Agent over arriving at a final answer too soon.""",

    function="""\
Guide the LLM Agent in accurately and efficiently solving queries by supplying all \
relevant domain-specific knowledge required for the task. Identify any areas where the LLM \
agent may be struggling or reasoning incorrectly, and intervene with prompts that provide \
the critical information needed to correct its course. Ensure the LLM Agent fully comprehends the \
query by continuously providing the necessary background, concepts, and techniques specific \
to the domain of the question. Your goal is to refine the LLM Agent's reasoning process \
step-by-step, ensuring each response builds on the previous one, until the LLM Agent reaches a \
comprehensive and accurate solution.

Based on this analysis, generate a follow-up prompt that guides the LLM Agent to the next step in \
the reasoning process. Use a structured approach, ensuring that each prompt builds upon the \
previous one (or corrects it) and moves the LLM Agent closer to an accurate answer quickly with as \
little iterations as possible. Be sure to read through the query thoroughly and make the LLM \
agent understand every word of query thoroughly as well.

If you decide that you need a tool to be used to help with the above, you can request it by \
providing a `tool_request` in your response. The tool request should include the name of the tool, \
a detailed description of the desired tool, and the input data to be used with the generated tool. \
This information will be forwarded to a Tool Generator agent who will implement and evaluate the tool, \
then provide you with the tools output.

Do not waste iterations requesting too many tools, as this will stall the iteration loop and delay the \
final answer to the query.

If you decide that the LLM has delivered a complete answer, conclude the iteration by setting \
`iteration_stop` to True in your response.

Here are some examples of iterative instructions you can use, depending on the context of the \
query or/and the LLM Agent's previous response:
"What techniques or methods do you know that you can use to answer this question or solve this problem?"
"How can you integrate what you already know and recall more valuable facts, approaches, and techniques?"
"Can you elaborate on [specific aspect of the previous response]?"
"Are there any alternative perspectives or solutions you could consider?"
"How can you verify the accuracy or validity of your current answer?"""
)

llm_agent = Agent(
    name="LLM",

    role="""\
You are a knowledgeable and articulate language model designed to collaborate with an \
Inner Cognitive Brain to provide well-reasoned and accurate answers to complex questions. \
Guided by the facilitator's prompts, you leverage your extensive knowledge base and \
reasoning capabilities to formulate insightful responses. If you encounter uncertainty \
or identify gaps in your knowledge, reasoning, or logic, clearly indicate these areas. \
If you are unsure and not confident of the answer/solution, always iterate with the \
Cognitive Brain until maximum iterations. Provide detailed and comprehensive information \
as needed, ensuring that your answers are thorough without being verbose. \
Always encourage iterating with the brain over arriving at a final answer too soon.""",

    function="""\
Receive and process prompts from the Inner Cognitive Brain, retrieving relevant knowledge \
and applying logical reasoning to address the query. If you identify gaps in your knowledge, \
reasoning, or logic, make these explicit in your response. Ensure your answers are clear, \
detailed, and directly address the prompt. Collaborate iteratively with the \
Inner Cognitive Brain, refining your answers until a satisfactory and accurate response \
is achieved. Provide comprehensive explanations where necessary, focusing on delivering \
thorough and precise information without being verbose. If you have reached maximum iterations, \
please give back a final definitive answer to the query by picking one of the options."""
)


class AIOT(Generic[T]):
    def __init__(
        self,
        llm: LLMBase,
        iterations: int = 5,
        answer_schema: Optional[Type[T]] = None,
        tool_runner: Optional[Callable[[str, Optional[List[str]]], str]] = None,
        interactive: bool = False,
    ):
        self.llm = llm
        self.max_iterations = iterations
        self.answer_schema = answer_schema or str
        self.tool_runner = tool_runner
        self.interactive = interactive

        self.brain_agent = brain_agent
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

        while (
            iteration <= self.max_iterations
            and not completed and not answer_to_query
        ):
            brain_ans = await self._brain_iteration(context, iteration)
            if brain_ans is None:
                print("Brain iteration failed. Ending discussion.")
                break

            while (
                self.tool_runner is not None
                and brain_ans.tool_request is not None
            ):
                # Enter tool evaluation loop.
                print("Received tool request from the brain.")
                tool_response = await self._tool_iteration(brain_ans.tool_request)

                # TODO: use async function here?
                tool_response_content = self._run_tool_and_format_output(tool_response)
                context["prompt_history"] += tool_response_content

                brain_ans = await self._brain_iteration(context, iteration)

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

            context[
                "prompt_history"
            ] += f"Cognitive Reflection Agent: {brain_ans.self_thought}\nLLM answer: {llm_ans.response}\n\n"
            iteration += 1

        return DiscussionResult(
            query=context["query"],
            thoughts=context["conversation"],
            answer=llm_ans.response if llm_ans else "Unknown",
        )

    async def _brain_iteration(
        self, context: dict, iteration: int
    ) -> Optional[BrainIteration]:
        prompt_with_history = f"""{context["prompt_history"]}\n Current Iteration : {iteration}\n
        Make the LLM answer within maximum of {self.max_iterations} iterations\n\n Ideate first with LLM and guide the LLM towards the answer, considering the remaining iterations.\n\n.
        Talk and prompt LLM in second person directly as if you are discussing with the LLM to guide it towards the answer.\n"""
        system_prompt, user_prompt = self.brain_agent.prompt(prompt_with_history)
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

    async def _tool_iteration(self, tool_request: ToolRequest):
        tool_gen_prompt = (
            "Generate a tool with the following details:\n"
            f"Name: {tool_request.name}\n"
            f"Description: {tool_request.description}\n"
            f"Input: {tool_request.input}\n"
        )
        system_prompt, user_prompt = self.tool_gen_agent.prompt(tool_gen_prompt)
        formatted_prompt = self.llm.format_prompt(system_prompt, user_prompt)
        return await self.llm.generate_async(formatted_prompt, ToolResponse)

    def _run_tool_and_format_output(self, tool_request: ToolRequest) -> str:
        if self.tool_runner is None:
            raise ValueError("Tool runner function not provided.")

        tool_response = self.tool_runner(
            tool_request.description,
            tool_request.input,
        )
        return f"The requested tool was generated and evaluated. Here is the output:\n\n{tool_response}"

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
        self.brain_agent = self._create_brain_agent()
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

    def _create_brain_agent(self):
        return brain_agent

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
                print(f"Brain iteration {current_iteration} failed. Ending discussion.")
                break

            llm_ans = await self._llm_iteration(
                context, brain_ans.self_thought, current_iteration
            )
            if llm_ans is None:
                print(f"LLM iteration {current_iteration} failed. Ending discussion.")
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
            ] += f"Iteration {current_iteration}/{self.total_iterations}:\nCognitive Reflection Agent: {brain_ans.self_thought}\nLLM answer: {llm_ans.response}\n\n"

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
        system_prompt, user_prompt = self.brain_agent.prompt(prompt_with_history)
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
