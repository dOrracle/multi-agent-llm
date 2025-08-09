"""Clean implementation of Adaptive Graph of Thought (AGOT) framework."""

import asyncio
import itertools
import json
import warnings
from dataclasses import dataclass
from enum import Enum, auto
from time import perf_counter
from typing import Dict, List, Optional, Tuple, Union, Type

import nest_asyncio
import networkx as nx
from pydantic import BaseModel
from rich.console import Console
from rich.text import Text

from multi_agent_llm.agents.adaptive_graph_of_thoughts.tasks import (
    CheckComplex, EvaluateTask, FinalAnswer, FinalTask, InitialTask,
    MultiTaskResponse, NewTask, NodeData, Task)
from multi_agent_llm.agents.adaptive_graph_of_thoughts.templates import (
    COMPLEXITY_CHECK_SYS_PROMPT, COMPLEXITY_CHECK_USER_PROMPT,
    FINAL_TASK_EXECUTION_PROMPT, FINAL_TASK_SYS_PROMPT, FINAL_TASK_USER_PROMPT,
    SYSTEM_PROMPT, TASK_EXECUTION_SYS_PROMPT, TASK_EXECUTION_USER_PROMPT,
    USER_PROMPT_INITIAL_SUB_TASK, USER_PROMPT_INITIAL_TASK,
    USER_PROMPT_NEW_TASK)
from ...llm import LLMBase

# Import web search utilities
try:
    from web_search_tool import should_search_web, get_web_context
except ImportError:
    # Graceful fallback if web_search_tool is not available
    async def should_search_web(llm, question: str, context: str = "") -> bool:
        return False
    async def get_web_context(question: str, num_results: int = 3) -> dict:
        return {}


class PromptCategory(Enum):
    """Categories of prompts used in the AGOT framework."""

    INITIAL_TASK = auto()
    INITIAL_SUB_TASK = auto()
    NEW_TASK = auto()
    TASK_EXECUTION = auto()
    COMPLEXITY_CHECK = auto()
    FINAL_TASK = auto()
    FINAL_TASK_EXECUTION = auto()


@dataclass
class NodePosition:
    depth: int  # D#
    layer: int  # L#
    position: int  # P#

    def __str__(self) -> str:
        return f"D{self.depth}L{self.layer}P{self.position}"


class AGOTLogger:
    def __init__(self, verbose: int = 0):
        """
        Parameters:
            verbose (int): an integer for verbosity of logging
        """
        self.console = Console()
        self.verbose = verbose
        self.start_times = {}  # Store start times for each node
        # Use a self.print and have a switch to turn it off or on.

    def start_timing(self, node_id: int):
        """
        Start timing for a specific node

        Parameters: 
            node_id (int): Integer ID to identify the node
        """
        self.start_times[node_id] = perf_counter()

    def get_elapsed_time(self, node_id: int) -> float:
        """
        Get elapsed time for a node in seconds

        Parameters:
            node_id (int): Integer ID to identify the node

        Returns
            Time (int): elapsed processing time for that node id in seconds
        """
        if node_id in self.start_times:
            return perf_counter() - self.start_times[node_id]
        return 0

    def format_time(self, seconds: float) -> str:
        """
        Format time duration with appropriate units

        Parameters:
            seconds (float): time taken in seconds

        Returns:
            Fomatted time (str): Formatted in minutes and seconds
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        else:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.1f}s"

    def get_node_lineage(self, node_id: int, dag: nx.DiGraph) -> str:
        """
        Construct the lineage string for a node based on layer and depth

        Parameters:
            node_id (int): Node ID to get the lineage
            dag (nx.DiGraph): Graph to extract the lineage of the node

        Returns:
            lineage (str): Lineage of the node in the graph
        """
        lineage = []
        current_node_id = node_id
        # Change to for loop
        while True:
            node_data = dag.nodes[current_node_id]
            depth = node_data.get("depth", 0)
            layer = node_data.get("layer", 0)

            # Find position in layer
            same_layer_nodes = [
                n
                for n, d in dag.nodes(data=True)
                if d.get("depth") == depth and d.get("layer") == layer
            ]
            position = same_layer_nodes.index(current_node_id)

            current_pos = f"D{depth}L{layer}P{position}"
            lineage.insert(0, current_pos)

            predecessors = list(dag.predecessors(current_node_id))
            if not predecessors:
                break
            # Assuming single parent for simplicity
            current_node_id = predecessors[0]

        return "->".join(lineage)

    def log(
        self,
        level: int,
        message: str,
        title: Optional[str] = None,
        node_id: Optional[Union[int, List[int]]] = None,
        dag: Optional[nx.DiGraph] = None,
        content: Optional[str] = None,
        response: Optional[str] = None,
        context: Optional[dict] = None,
        is_complex: Optional[bool] = None,
    ):
        """
        Main logging method with rich formatting based on verbosity

        Parameters:
            level (int): verbosity level
            message (str): Message to print/log
            title (str): Title of the node
            node_id (Union(int, List[int])): (list of) node(s) to log
            dag (nx.DiGraph): Graph to log
            content (str): Answer content of nodes to print
            response (str): Response based on schemas
            context (dict): Context for different node processing
            is_complex (bool): Complexity of the node task
        """
        # Use a file to log the outputs and the time it took to run the code.
        if self.verbose < level:
            return

        if node_id is not None and dag is not None:
            if isinstance(node_id, list):
                self.console.print(
                    f"Node_Ids: {node_id} \nContext:\n{json.dumps(context, indent=2)}\n", style="italic"
                )
                return
            lineage = self.get_node_lineage(node_id, dag)
        elif dag is not None:
            lineage = "NEW_TASK"
        else:
            lineage = "ROOT"

        if level == 1:
            styled_text = Text()
            styled_text.append(f"[{lineage}] ", style="cyan")
            styled_text.append(f"Node {node_id}: ", style="yellow")
            styled_text.append(title or message, style="white")

            if is_complex is not None:
                complexity_style = "red" if is_complex else "green"
                complexity_text = " (complex)" if is_complex else " (simple)"
                styled_text.append(complexity_text, style=complexity_style)

            if isinstance(node_id, int) and node_id in self.start_times:
                elapsed_time = self.get_elapsed_time(node_id)
                time_str = self.format_time(elapsed_time)
                styled_text.append(f" [{time_str}]", style="bright_black")

            self.console.print(styled_text)

        elif level == 2:
            styled_text = Text()
            styled_text.append(f"[{lineage}] ", style="cyan")
            styled_text.append(f"Node {node_id}: ", style="yellow")
            styled_text.append(title or message, style="white")

            if is_complex is not None:
                complexity_style = "red" if is_complex else "green"
                complexity_text = " (complex)" if is_complex else " (simple)"
                styled_text.append(complexity_text, style=complexity_style)

            if isinstance(node_id, int) and node_id in self.start_times:
                elapsed_time = self.get_elapsed_time(node_id)
                time_str = self.format_time(elapsed_time)
                styled_text.append(f" [{time_str}]", style="bright_black")

            self.console.print(styled_text)

            if content:
                self.console.print(f"Task Content:\n{content}\n", style="dim")
            if response:
                self.console.print(
                    f"LLM Response:\n{response}\n", style="bold")

        elif level >= 3:
            self.log(
                2, message, title, node_id, dag, content, response, context, is_complex
            )
            if context:
                self.console.print(
                    f"Context:\n{json.dumps(context, indent=2)}\n", style="italic"
                )


class AGOT:
    """Adaptive Graph of Thought Framework.

    An LLM reasoning framework that starts with initial tasks, execute them, dynamically generates new tasks in two dimensions, and evaluates them to get a final answer.
    """

    _nest_asyncio_applied = False

    _prompt_templates = {
        PromptCategory.INITIAL_TASK: (
            SYSTEM_PROMPT,
            USER_PROMPT_INITIAL_TASK,
        ),
        PromptCategory.NEW_TASK: (
            SYSTEM_PROMPT,
            USER_PROMPT_NEW_TASK,
        ),
        PromptCategory.INITIAL_SUB_TASK: (
            SYSTEM_PROMPT,
            USER_PROMPT_INITIAL_SUB_TASK,
        ),
        PromptCategory.TASK_EXECUTION: (
            TASK_EXECUTION_SYS_PROMPT,
            TASK_EXECUTION_USER_PROMPT,
        ),
        PromptCategory.COMPLEXITY_CHECK: (
            COMPLEXITY_CHECK_SYS_PROMPT,
            COMPLEXITY_CHECK_USER_PROMPT,
        ),
        PromptCategory.FINAL_TASK: (
            FINAL_TASK_SYS_PROMPT,
            FINAL_TASK_USER_PROMPT,
        ),
        PromptCategory.FINAL_TASK_EXECUTION: (
            TASK_EXECUTION_SYS_PROMPT,
            FINAL_TASK_EXECUTION_PROMPT,
        ),
    }

    def __init__(
        self,
        llm: LLMBase,
        max_new_tasks: int = 3,
        max_depth: int = 1,
        verbose: int = 0,
        max_num_layers: int = 2,
        max_concurrent_tasks: int = 5000,
        layer_depth_reduction: bool = True,
    ):
        """
        Parameters:
            llm (LLMBase): LLM to work with
            max_new_tasks (int): Maximum number of new tasks for each layer
            max_depth (int): maximum number of nested graphs that can be generated
            verbose (int): Verbosity for logging
            max_num_layers (int): Maximum number of layers that can be generated in a graph
            max_concurrent_tasks (int): Maximum number of tasks that can be asynced
            layer_depth_reduction (bool): Whether or not a decrease in the size of nested graph is required
        """
        self.llm = llm
        self.max_new_tasks = max_new_tasks
        self.verbose = verbose
        self.max_depth = max_depth
        self.max_num_layers = max_num_layers
        self.max_concurrent_tasks = max_concurrent_tasks
        self.layer_depth_reduction = layer_depth_reduction

        self._id_counter = itertools.count(0)
        self._event_log = []
        self.user_schema = None
        self.logger = AGOTLogger(verbose)
        # Initialize the semaphore for limiting concurrent tasks
        self.semaphore = asyncio.Semaphore(
            self.max_concurrent_tasks)  # New line

    def get_node_list(self, dag: nx.DiGraph) -> str:
        """
        Get a formatted node list with node IDs and titles in the dag.

        Parameters:
            dag (nx.DiGraph): Graph to get the node list from

        Returns:
            Node_list (str): List of nodes with node ID and title
        """
        nodes = []
        for node_id in dag.nodes:
            node_data = dag.nodes[node_id]
            title = node_data.get("title", "")
            nodes.append(f"<ID> {node_id} </ID>/n <Title>: {title}</Title>")
        return "\n".join(nodes)

    from typing import Optional, Union
    def _add_task(
        self,
        task: Task,
        parent_id: Optional[Union[int, List[int]]] = None,
        dag: Optional[nx.DiGraph] = None,
        depth: int = 0,
        layer: int = 0,
        strategy: Optional[str] = None,
    ) -> Tuple[int, nx.DiGraph]:
        """
        Add a task to the graph as a node.

        Parameters:
            task (Task): The task to be added as a node to the graph
            parent_id (Union[int, List[int]]): Parent id of nodes if node is not the root node
            dag (nx.DiGraph): The graph to which nodes are to be adedd
            depth (int): The nesting depth of the current graph
            layer (int): The current layer number of the node in the graph
            strategy (str): The strategy for the generated layer

        Returns:
            _id (int): Node ID of the new node
            dag (nx.DiGraph): The updated graph
        """
        _id = next(self._id_counter)
        task_string = self._format_task(_id, task)
        if dag is not None:
            # Add the node to the dag
            dag.add_node(
                _id, task=task_string, depth=depth, layer=layer, title=task.title, strategy=strategy
            )

            if parent_id is not None:
                # If there is a parent id, connect node to parents

                if not isinstance(parent_id, list):
                    parent_id = [parent_id]
                parent_id = list(set(parent_id).intersection(
                    set(list(dag.nodes))))  # Ensure unique parent IDs
                if not parent_id:
                    warnings.warn(
                        "No common parents found in the parent graph when adding a task."
                    )
                for _parent_id in parent_id:
                    # Make sure there are no self loops or repeated edges
                    if _id != _parent_id and not dag.has_edge(_parent_id, _id):
                        dag.add_edge(_parent_id, _id)

        if dag is None:
            dag = nx.DiGraph()
        return _id, dag

    def _format_task(self, _id: int, task: Task) -> str:
        """
        Format a task as a string with a unique ID.

        Parameters:
            _id (int): Node ID
            task (Task): Task to be formatted

        Returns:
            formatted_tasl (str): Formatted task with ID, Title and Content
        """
        return f"<{_id}>[{task.title}] {task.content}</{_id}>"

    def _format_strategy(self, strategy: Dict[int, str]) -> str:
        """
        Format the strategy dictionary as a string.

        Parameters:
            strategy (Dict): Strategy dictionary to be formatted

        Returns:
            Formatted dictionary (str): Dictionary formatted into json string
        """
        return json.dumps(strategy)

    def run(self, question: str, schema: Optional[Type[BaseModel]] = None) -> Optional[FinalAnswer]:
        """
        Synchronously run the AGOT framework to generate a response to the given question.

        Parameters:
            question (str): The question to be answered
            schema (BaseModel): The schema for final answer

        Returns:
            Final Answer (FinalAnswer): Final Answer in the required format
        """
        if not self._nest_asyncio_applied:
            try:
                nest_asyncio.apply()
                self._nest_asyncio_applied = True
            except Exception as e:
                print(f"Error applying nest_asyncio: {e}")
        return asyncio.run(self.run_async(question, schema))

    async def run_async(
        self, question: str, schema: Optional[Type[BaseModel]] = None
    ) -> FinalAnswer:
        """
        Asynchronously run the AGOT framework to generate a response to the given question.

        Parameters:
            question (str): The question to be answered
            schema (BaseModel): The schema for final answer

        Returns:
            Final Answer (FinalAnswer): The final answer to the question in the required format 
        """
        self.user_schema = schema  # Schema for final response
        dag = nx.DiGraph()  # Initialize a Directed Graph
        final_response = None
        if self.verbose >= 4:
            self._log(4, "Starting task processing")

        # Generate initial tasks and strategy (NO web search in this test)
        initial_tasks, strategy = await self._generate_initial_tasks(question)
        depth = 0
        layer = 0
        layer_dict = {0: []}

        # Create a dictionary of strategies with layers as keys
        strategy_dict = {layer: strategy}

        # Add initial tasks to the dag as root nodes
        for task in initial_tasks:
            node_id, dag = self._add_task(
                task, dag=dag, depth=depth, layer=layer, parent_id=None, strategy=strategy
            )
            # Add tasks to layer dict to get results
            layer_dict[layer].append((node_id, task))

        # Keep generating tasks until final answer is reached or max layers are exhausted
        for layer in range(self.max_num_layers):
            if self.verbose >= 3:
                self._log(
                    3,
                    "Processing tasks in graph",
                    node_id=(list(dag.nodes)[0] if dag.nodes else None),
                    dag=dag,
                )

            tasks = [
                self.process_task(
                    node_id=node_id,
                    task=task,
                    question=question,
                    depth=depth,
                    dag=dag,
                    main_dag=dag,
                )
                for node_id, task in layer_dict[layer]
            ]
            # Limit the number of concurrent tasks
            async with self.semaphore:
                await asyncio.gather(*tasks)

            # Generate new tasks only if layer<self.max_num_layers-1
            if layer < self.max_num_layers-1:
                # Get formatted node list with ID and title
                node_list_str = self.get_node_list(dag=dag)

                # Generate new tasks
                new_tasks, strategy = await self._generate_new_tasks(
                    question=question,
                    depth=depth,
                    node_list_str=node_list_str,
                    task_graph=self.graph_to_string(dag=dag),
                    dag=dag,
                    strategy_dict=strategy_dict,
                )

                # Add the strategy corresponding to the new layer
                strategy_dict[layer+1] = strategy

                # If the new task is final, evaluate and return the answer
                if isinstance(new_tasks, FinalTask):
                    self._log(
                        1, f"Final answer reached at Layer {layer} in the main graph")

                    # Add the final task to the digraph
                    final_node_id, dag = self._add_task(
                        new_tasks,
                        dag=dag,
                        depth=depth,
                        layer=layer,
                        parent_id=new_tasks.parent_id,
                        strategy=strategy,
                    )
                    # Evaluate the final task
                    final_response = await self._evaluate_task(
                        task_id=final_node_id,
                        task=new_tasks,
                        question=question,
                        final_schema=self.user_schema if self.user_schema is not None else None,
                        dag=dag,
                    )

                    dag.nodes[final_node_id]["answer"] = final_response

                    return FinalAnswer(
                        final_answer=final_response,
                        graph=self.export_graph(dag),
                    )

                # If new tasks are not final, add it to layer dictionary to be evaluated
                layer_dict[layer+1] = []
                for new_task in new_tasks[:self.max_new_tasks]:
                    node_id, dag = self._add_task(
                        new_task,
                        dag=dag,
                        depth=depth,
                        layer=layer+1,
                        parent_id=new_task.parent_id,
                        strategy=strategy,
                    )
                    layer_dict[layer+1].append((node_id, new_task))
                # The loop continues and these tasks will be evaluated

        # If max layers is reached without final answer
        # If max layers is reached without final answer
        node_list_str = self.get_node_list(dag=dag)
        # Force generate the final task
        final_task = await self._generate_final_task(
            question=question,
            depth=depth,
            node_list_str=node_list_str,
            dag=dag,
        )
        # Use all the nodes in the graph as parents
        parent_ids = list(dag.nodes)
        final_node_id, dag = self._add_task(
            final_task,
            dag=dag,
            depth=depth,
            layer=self.max_num_layers,
            parent_id=parent_ids,
            strategy=strategy,
        )
        # Evaluate the final task
        final_answer = await self._evaluate_task(
            task_id=final_node_id,
            task=final_task,
            question=question,
            final_schema=self.user_schema if self.user_schema is not None else None,
            dag=dag,
        )

        dag.nodes[final_node_id]["answer"] = final_answer
        return FinalAnswer(
            final_answer=final_answer,
            graph=self.export_graph(dag),
        )

    def format_all_answers(self, dag: nx.DiGraph) -> str:
        """
        Format all answers of the dag into a string.

        Parameters:
            dag (nx.DiGraph): The graph of interest

        Returns:
            formatted_responses (str): Content of the graph in a string format
        """
        answers = []
        for node_id in dag.nodes:
            node = dag.nodes[node_id]
            if "answer" in node:
                answers.append(
                    f"<{node_id}>\nTitle: {node['title']}\nAnswer: {node['answer']}\n</{node_id}>"
                )
        if not answers:
            return "No answers available."
        return "\n".join(answers)

    async def _generate_initial_tasks(self, question: str) -> Tuple[List[Task], str]:
        """
        Generate initial tasks for the main question.

        Parameters:
            question (str): The main question to answer

        Returns:
            Tasks (List[Task]): List of initial tasks to solve the question
            Strategy (str): The strategy of generating the initial tasks
        """
        try:
            response = await self._generate(
                category=PromptCategory.INITIAL_TASK,
                schema=InitialTask,
                question=question,
                max_new_tasks=self.max_new_tasks,
                dag=None,
            )
            tasks = getattr(response, "tasks", None)
            strategy = getattr(response, "strategy", None)
            if tasks is not None and strategy is not None:
                return tasks, strategy
            else:
                raise TypeError("Response missing tasks or strategy in _generate_initial_tasks")
        except Exception as e:
            self._log(
                level=1,
                message=f"Error generating initial tasks: {e}",
                response=str(response) if 'response' in locals() else 'No response object'
            )
            # Attempt to re-parse with a more flexible schema or return a default
            if 'response' in locals() and isinstance(response, str):
                try:
                    import json
                    data = json.loads(response)
                    tasks = [Task(**task_data) for task_data in data.get('tasks', [])]
                    strategy = data.get('strategy', 'Default strategy due to parsing error.')
                    if tasks:
                        return tasks, strategy
                except json.JSONDecodeError:
                    pass  # Fall through to raise the original error

            raise TypeError(f"Failed to generate initial tasks after retry: {e}")

    async def _generate_initial_sub_tasks(
        self, task: Task, question: str, dag: Optional[nx.DiGraph] = None
    ) -> Tuple[List[Task], str]:
        """
        Generate initial tasks for subgraphs.

        Parameters:
            task (Task): The task which is marked complex and needs to be broken into another graph
            question (str): The main question to answer
            dag (nx.DiGraph): The graph to add the new tasks to

        Returns:
            Tasks (List[Task]): The list of initial tasks for the nested graph
            Strategy (str): The strategy behind generating the initial tasks
        """
        response = await self._generate(
            category=PromptCategory.INITIAL_SUB_TASK,
            schema=InitialTask,
            question=question,
            max_new_tasks=self.max_new_tasks,
            task=task,
            dag=dag,
        )
        tasks = getattr(response, "tasks", None)
        strategy = getattr(response, "strategy", None)
        if tasks is not None and strategy is not None:
            return tasks, strategy
        else:
            raise TypeError("Response missing tasks or strategy in _generate_initial_sub_tasks")

    async def process_task(
        self,
        node_id: int,
        task: Task,
        question: str,
        depth: int,
        dag: nx.DiGraph,
        main_dag: nx.DiGraph,
    ) -> None:
        """
        Check if task is complex. If max depth is not reached, process the task as a subgraph. Else, evaluate the task.

        Parameters:
            node_id (int): The node ID to process
            task (Task): The task that is checked for complexity
            question (str): The main question at hand
            depth (int): The current nested graph depth
            dag (nx.DiGraph): The current graph to process the nodes from
            main_dag (nx.DiGraph): The first graph that solves the question
        """
        async with self.semaphore:
            self.logger.start_timing(node_id)
            if depth < self.max_depth:
                # Check if the task is complex or not
                complexity_check = await self._check_complex(
                    task_id=node_id,
                    task=task,
                    dag=dag,
                    main_graph=self.graph_to_string(main_dag),
                    depth=depth,
                )
                is_complex = complexity_check.is_complex

                if self.verbose >= 1:
                    self._log(1, task.title, node_id=node_id,
                              dag=dag, is_complex=is_complex)

                # If the task us complex and max depth is not reached, process it as a subgraph
                if is_complex:
                    subgraph_answer, subgraph = await self.process_subgraph(
                        task=task,
                        question=question,
                        node_id=node_id,
                        depth=depth + 1,
                        dag=dag,
                        main_dag=main_dag,
                    )
                    dag.nodes[node_id]["answer"] = subgraph_answer
                    dag.nodes[node_id]["subgraph"] = subgraph
            else:

                if self.verbose >= 1:
                    self._log(1, task.title, node_id=node_id,
                              dag=dag, is_complex=False)
                task_answer = await self._evaluate_task(
                    task_id=node_id,
                    task=task,
                    question=question,
                    final_schema=None,
                    dag=dag,
                )

                if self.verbose >= 2:
                    self._log(
                        2,
                        task.title,
                        node_id=node_id,
                        dag=dag,
                        content=task.content,
                        response=str(task_answer),
                    )
                try:
                    dag.nodes[node_id]["answer"] = task_answer.content
                except Exception:
                    dag.nodes[node_id]["answer"] = task_answer

            self.logger.start_times[node_id] = 0  # Reset timing after logging

    async def process_subgraph(
        self,
        task: Task,
        question: str,
        node_id: int,
        depth: int,
        dag: nx.DiGraph,
        main_dag: nx.DiGraph,
    ) -> Tuple[str, nx.DiGraph]:
        """
        Process the subgraph.

        Parameters:
            task (Task): The complex task that requires a breakdown into another graph
            question (str): The main question at hand
            node_id (int): The node that is marked complex
            depth (int): The current nested depth of the graph
            dag (nx.DiGraph): The current graph where the xomplex task is present
            main_dag (nx.DiGraph): The main graph that solves the original question

        Returns:
            Task answer (str): Answer to the task
            subgraph (nx.DiGraph): The nested graph generated to solve the task
        """
        if self.verbose >= 4:
            self._log(4, "Processing subgraph", node_id=node_id, dag=dag)

        subgraph = nx.DiGraph()  # Initial a subgraph
        final_response = None
        layer = 0
        layer_dict = {0: []}
        # Generate initial tasks for subgraph
        initial_tasks, strategy = await self._generate_initial_sub_tasks(task, question, dag=main_dag)
        for task in initial_tasks:
            subgraph_node_id, subgraph = self._add_task(
                task,
                dag=subgraph,
                depth=depth,
                layer=layer,
                strategy=strategy,
            )
            layer_dict[layer].append((subgraph_node_id, task))

        # Create a strategy dictionary for the subgraph
        strategy_sub_dict = {layer: strategy}

        # If layer_depth_reduction is true, reduce #depth (max) layers from max_layers that can be generated
        max_layers = self.max_num_layers - \
            depth if self.layer_depth_reduction else self.max_num_layers

        for layer in range(max_layers):
            if self.verbose >= 3:
                self._log(
                    3,
                    "Processing subgraph tasks",
                    node_id=node_id,
                    dag=main_dag,
                    context={"subgraph": self.graph_to_string(dag=subgraph)},
                )

            tasks = [
                self.process_task(
                    node_id=node_sub_id,
                    task=task,
                    question=question,
                    depth=depth,
                    dag=subgraph,
                    main_dag=main_dag,
                )
                for node_sub_id, task in layer_dict[layer]
            ]
            async with self.semaphore:
                await asyncio.gather(*tasks)

            # Generate new tasks only if layer<max_layers-1
            if layer < max_layers-1:
                # Get formatted list of nodes with ID and title
                node_list_str = self.get_node_list(dag=subgraph)

                # Generate new tasks
                new_tasks, strategy = await self._generate_new_tasks(
                    question=task.content,
                    depth=depth,
                    node_list_str=node_list_str,
                    task_graph=self.graph_to_string(dag=subgraph),
                    dag=subgraph,
                    strategy_dict=strategy_sub_dict,
                )

                # Add the strategy corresponding to the new layer
                strategy_sub_dict[layer+1] = strategy

                if isinstance(new_tasks, FinalTask):
                    self._log(
                        1, f"Final answer reached at Layer {layer} at depth {depth}.")
                    final_node_id, subgraph = self._add_task(
                        new_tasks,
                        dag=subgraph,
                        depth=depth,
                        layer=layer,
                        parent_id=new_tasks.parent_id,
                        strategy=strategy,
                    )

                    # Evaluate the final task
                    final_response = await self._evaluate_task(
                        task_id=final_node_id,
                        task=new_tasks,
                        question=question,
                        dag=subgraph,
                    )

                    subgraph.nodes[final_node_id]["answer"] = final_response.content
                    return final_response.content, subgraph

                # If new tasks are not final, add it to the layer dictionary to be evaluated
                layer_dict[layer+1] = []
                for new_task in new_tasks[:self.max_new_tasks]:
                    node_sub_id, subgraph = self._add_task(
                        new_task,
                        dag=subgraph,
                        depth=depth,
                        layer=layer+1,
                        parent_id=new_task.parent_id,
                        strategy=strategy,
                    )
                    layer_dict[layer+1].append((node_sub_id, new_task))
                # The loop continues and these tasks will be evaluated

        # If final_response is None, force generate the final task and return
        node_list_str = self.get_node_list(dag=subgraph)
        # Force generate the final task
        final_task = await self._generate_final_task(
            question=question,
            depth=depth,
            node_list_str=node_list_str,
            dag=subgraph,
        )
        # Use all the nodes in the subgraph as parents
        parent_ids = list(subgraph.nodes)
        final_node_id, subgraph = self._add_task(
            final_task,
            dag=subgraph,
            depth=depth,
            layer=self.max_num_layers + 1,
            parent_id=parent_ids,
            strategy=strategy,
        )
        # Evaluate the final task
        final_answer = await self._evaluate_task(
            task_id=final_node_id,
            task=final_task,
            question=question,
            final_schema=None,
            dag=subgraph,
        )

        subgraph.nodes[final_node_id]["answer"] = final_answer.content
        return final_answer.content, subgraph

    def _log(
        self,
        level: int,
        message: str,
        depth=None,
        layer=None,
        node_id=None,
        content=None,
        response=None,
        context=None,
        dag=None,
        is_complex=None,
    ):
        """
        Enhanced logging with rich formatting.

        Parameters:
            level (int): Level for logging
            message (str): Message for logging
            depth (int): current depth of the nested graphs
            layer (int): current layer of the node
            node_id (int): Id of the node
            content (str): Content of the node
            response (str): Response after api call
            context (str): Context provided to the api call
            dag (nx.DiGraph): The graph to log
            is_complex (book): Complexity check
        """
        if self.verbose < level:
            return

        title = message
        if node_id is not None and dag is not None:
            self.logger.log(
                level=level,
                message=message,
                title=title,
                node_id=node_id,
                dag=dag,
                content=content,
                response=response,
                context=context,
                is_complex=is_complex,
            )
        elif dag is not None:
            self.logger.log(
                level=level,
                message=message,
                title=title,
                dag=dag,
                content=content,
                response=response,
                context=context,
                is_complex=is_complex,
            )
        else:
            self.logger.log(level=level, message=message)
        self._event_log.append(message)

    async def _generate_final_task(
        self,
        question: str,
        depth: int,
        node_list_str: str,
        dag: nx.DiGraph,
    ) -> FinalTask:
        """
        Generate a final task when max layers are reached.

        Parameters: 
            question (str): The main question to solve
            depth (int): The current nested depth of the graphs
            node_list_str (str): The list of nodes in the graph in string format
            dag (nx.DiGraph): The current graph to generate the final task of

        Returns:
            Final Task (FinalTask): Te final task generated when max layers is reached in the graph
        """
        response = await self._generate(
            category=PromptCategory.FINAL_TASK,
            schema=FinalTask,
            question=question,
            depth=depth,
            max_new_tasks=1,
            node_list=node_list_str,
            force_final_task=True,
            dag=dag,
        )
        if isinstance(response, FinalTask):
            return response
        tasks = getattr(response, "tasks", None)
        if isinstance(tasks, FinalTask):
            return tasks
        elif isinstance(tasks, list) and len(tasks) > 0 and isinstance(tasks[0], FinalTask):
            return tasks[0]
        else:
            raise ValueError("Failed to generate a final task.")

    def get_ancestors_answers(self, node_id: int, dag: nx.DiGraph) -> str:
        """
        Get the answers of the ancestor nodes of the given node.

        Parameters: 
            node_id (int): The node to get the ancestry of
            dag (nx.DiGraph): The graph to find the ancestry in

        Returns:
            ancestral content (str): Content of the ancestors of the node in XML format
        """
        return self.graph_to_string(dag.subgraph(list(nx.ancestors(dag, node_id))))

    async def _generate_new_tasks(
        self,
        question: str,
        depth: int,
        node_list_str: str,
        task_graph: str,
        strategy_dict: Dict[int, str],
        force_final_task: bool = False,
        dag: Optional[nx.DiGraph] = None,
    ) -> Tuple[Union[List[NewTask], FinalTask], str]:
        """
        Generate new tasks in a graph.

        Parameters:
            question (str): The main question at hand
            depth (int): The current nested graph level
            node_list_str (str): List of nodes in the graph in string format
            task_graph (str): The current state of the graph responses
            strategy_dict (Dict): Dictionary of strategies for each layer in the graph
            force_final_task (bool): Whether or not the final task is to be forced
            dag (nx.DiGraph): The current graph

        Returns:
             Tasks (List[Task]): The list of generated tasks
             Strategy (str): The strategy for the generated layer
        """
        response = await self._generate(
            category=PromptCategory.NEW_TASK,
            schema=MultiTaskResponse,
            question=question,
            depth=depth,
            max_new_tasks=self.max_new_tasks,
            node_list=node_list_str,
            task_graph=task_graph,
            force_final_task=force_final_task,
            dag=dag,
            layer_strategy=self._format_strategy(strategy_dict),
        )
        tasks = getattr(response, "tasks", None)
        strategy = getattr(response, "strategy", None)
        try:
            node_id_val = None
            if dag is not None and hasattr(dag, "nodes"):
                node_id_val = list(dag.nodes) if dag.nodes else None
            self._log(
                3,
                "Generated new tasks",
                depth=depth,
                node_id=node_id_val,
                dag=dag,
                context={"new_tasks": [str([list(set(task.parent_id))]) for task in tasks] if tasks is not None and not isinstance(tasks, FinalTask) else getattr(tasks, "parent_id", None)},
            )
        except Exception:
            pass
        if tasks is not None and strategy is not None:
            return tasks, strategy
        else:
            raise TypeError("Response missing tasks or strategy in _generate_new_tasks")

    async def _evaluate_task(
        self,
        task_id: int,
        task: Task,
        question: str,
        final_schema: Optional[type] = None,
        dag: Optional[nx.DiGraph] = None,
    ) -> EvaluateTask:
        """
        Evaluate the task.

        Parameters:
            task_id (int): The node to be evaluated
            task (Task): The task to evaluate
            question (str): The main question at hand
            final_schema (BaseModel): The schema for response
            dag (nx.DiGraph): The graph where the node to be evaluated is present

        Returns:
            response (EvaluateTask): The evaluated task response
        """

        # Get answers corresponding to the ancestors of a given node
        task_graph_str = self.get_ancestors_answers(task_id, dag if dag is not None else nx.DiGraph())
        
        # Check if web search is needed for this task
        web_context = ""
        try:
            # Try importing web search tool with different paths
            try:
                from web_search_tool import should_search_web, get_web_context
            except ImportError:
                # Try relative import from project root
                import sys
                import os
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                from web_search_tool import should_search_web, get_web_context
            
            # Create a combined context for web search decision
            search_context = f"Task: {task.title}\nContent: {task.content}\nQuestion: {question}"
            
            # Check if we need web search
            if await should_search_web(self.llm, search_context):
                web_context = await get_web_context(search_context)
                if self.verbose >= 1:
                    print(f"🌐 Web search performed for task {task_id}: {task.title}")
        except Exception as e:
            if self.verbose >= 1:
                print(f"⚠️ Web search failed for task {task_id}: {e}")
            web_context = ""
        
        from typing import cast
        if final_schema is None:
            # Format web context for the prompt
            formatted_web_context = ""
            if web_context:
                if isinstance(web_context, dict) and 'web_context' in web_context:
                    formatted_web_context = f"\nWeb Search Results:\n{web_context['web_context']}"
                elif isinstance(web_context, str) and web_context.strip():
                    formatted_web_context = f"\nWeb Search Results:\n{web_context}"
                
            response = await self._generate(
                category=PromptCategory.TASK_EXECUTION,
                schema=EvaluateTask,
                dag=dag,
                task_id=task_id,
                task_title=task.title,
                task_content=task.content,
                question=question,
                task_graph=task_graph_str,
                web_context=formatted_web_context
            )
            if isinstance(response, EvaluateTask):
                return cast(EvaluateTask, response)
            raise TypeError("Response is not of type EvaluateTask")
        else:
            # final_schema should be a type, not an instance
            if not isinstance(final_schema, type):
                raise TypeError("final_schema must be a type, not an instance")
            response = await self._generate(
                category=PromptCategory.FINAL_TASK_EXECUTION,
                schema=final_schema,
                dag=dag,
                task_id=task_id,
                task_title=task.title,
                task_content=task.content,
                question=question,
                task_graph=task_graph_str
            )
            if isinstance(response, final_schema):
                return cast(EvaluateTask, response)
            raise TypeError(f"Response is not of type {final_schema}")

    async def _check_complex(
        self,
        task_id: int,
        task: Task,
        depth: int = 0,
        dag: Optional[nx.DiGraph] = None,
        main_graph: str = "",
    ) -> CheckComplex:
        """
        Check whether a task is complex enough to be broken as another graph.

        Parameters: 
            task_id (int): The node to be checked for complexity 
            task (Task): The task to be checked for complexity
            depth (int): The current nested graph level
            dag (nx.DiGraph): The graph in context
            main_graph (str): The string version of the main graph that solves the original question

        Returns:
            Complex response (CheckComplex): True or False based on whether or not the task needs a separate graph
        """
        response = await self._generate(
            category=PromptCategory.COMPLEXITY_CHECK,
            schema=CheckComplex,
            task_id=task_id,
            task_title=task.title,
            task_content=task.content,
            main_graph=main_graph,
            dag=dag,
            depth=depth,
        )
        if isinstance(response, CheckComplex):
            return response
        else:
            raise TypeError("Response is not of type CheckComplex")

    from typing import Type
    async def _generate(
        self,
        category: PromptCategory,
        schema: Type[BaseModel],
        dag: Optional[nx.DiGraph] = None,
        **user_kwargs,
    ) -> BaseModel:
        """
        Generate responses based on different schemas.

        Parameters:
            category (PromptCategory): Category to generate the responses in
            schema (BaseModel): Schema based on categories
            dag (nx.DiGraph): The graph in context
            user_kwargs (dict): Other optional arguments 

        Returns:
            response: API call response formatted according to the schema
        """
        force_final_task = user_kwargs.pop("force_final_task", False)
        if force_final_task:
            category = PromptCategory.FINAL_TASK

        user_kwargs["dag"] = self.graph_to_string(
            dag) if dag is not None else None
        system_prompt, user_prompt_template = self._prompt_templates[category]
        user_prompt = user_prompt_template.format(**user_kwargs)
        messages = self.llm.format_prompt(system_prompt, user_prompt)

        if self.verbose >= 3:
            context = {
                "system_prompt": messages[0]["content"],
                "user_prompt": messages[1]["content"],
            }
            self._log(
                3,
                f"Generation for {category.name}",
                node_id=user_kwargs.get("task_id"),
                dag=dag,
                context=context,
            )

        # Get responses using API calls
        response = await self.llm.generate_async(messages, schema=schema)

        if self.verbose >= 3:
            self._log(
                3,
                f"Response for {category.name}",
                node_id=user_kwargs.get("task_id"),
                dag=dag,
                response=str(response),
            )
        return response if isinstance(response, BaseModel) else schema()

    def graph_to_string(self, dag: nx.DiGraph) -> str:
        """
        Generate an XML representation of the graph.

        Parameters: 
            dag (nx.DiGraph): The graph that would be converted to a string

        Returns:
            string of graph (str): The XML version of the graph
        """

        def node_to_xml(node_id, visited, dag):
            if node_id in visited:
                return ""
            visited.add(node_id)
            node_data = dag.nodes[node_id]
            title = node_data.get("title", "")
            content = node_data.get("task", "")
            answer = node_data.get("answer", "")
            depth = node_data.get("depth", 0)
            layer = node_data.get("layer", 0)
            subgraph = node_data.get("subgraph", "")
            strategy = node_data.get("strategy", "")
            xml = f'<Node title="{title}" depth="{depth}" layer="{layer}">\n'
            xml += f"  <Content>{content}</Content>\n"
            xml += f"  <Strategy>{strategy}</Strategy>\n"
            if answer:
                xml += f"  <Answer>{answer}</Answer>\n"
            # if subgraph:
            #     xml += "  <Subgraph>\n"
            #     for sub_node_id in subgraph:
            #         sub_xml = subgraph.nodes[sub_node_id].get("title", "")
            #         xml += sub_xml
            #     xml += "  </Subgraph>\n"
            children = list(dag.successors(node_id))
            if children:
                xml += "  <Children>\n"
                for child_id in children:
                    child_xml = dag.nodes[child_id].get("title", "")
                    xml += child_xml
                xml += "  </Children>\n"
            xml += "</Node>\n"
            return xml

        visited = set()
        xml = ""
        nodes = [n for n, _ in dag.in_degree()]
        for node in nodes:
            xml += node_to_xml(node, visited, dag)
        return xml

    def export_graph(self, dag: nx.DiGraph) -> List[NodeData]:
        """
        Export the graph as a list of NodeData objects.

        Parameters: 
            dag (nx.DiGraph): The graph to export

        Returns:
            List of nodes (List[NodeData]): List of nodes in the graph with nested graphs as attributes
        """
        node_dict = {}
        for node_id in dag.nodes:
            node_data = dag.nodes[node_id]
            node_dict[node_id] = node_id
            # Add subgraph
            subgraph = dag.nodes[node_id].get("subgraph", None)
            if subgraph is not None:
                json_string = self.export_graph(subgraph)
            else:
                json_string = ""

            node_dict[node_id] = NodeData(
                id=node_id,
                title=node_data.get("title", ""),
                content=node_data.get("task", ""),
                answer=node_data.get("answer", None),
                depth=node_data.get("depth", 0),
                layer=node_data.get("layer", 0),
                subgraph=json_string,
                strategy=node_data.get("strategy", ""),
                children=[],
            )

        # Set the children relationships
        for node_id in dag.nodes:
            node_dict[node_id].children = [
                child_id for child_id in dag.successors(node_id)
            ]

        return [node_dict[node_id] for node_id in dag.nodes]