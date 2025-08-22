"""
True AGoT - Adaptive Graph of Thought with Natural Reasoning and Hybrid Web Search
Combines graph structure, adaptive logic, and intelligent web search orchestration
"""
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from multi_agent_llm.llm import LLMBase

# Import hybrid search capabilities
backend_path = Path(__file__).parent.parent.parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# This block is only for static type analysis. It will not be executed at runtime.
if TYPE_CHECKING:
    from core.hybrid_search import SearchIntegratedAgent, SearchContext, SearchTrigger

# This block handles the runtime import.
try:
    from core.hybrid_search import SearchIntegratedAgent, SearchContext, SearchTrigger
    SEARCH_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import hybrid search: {e}")
    SEARCH_AVAILABLE = False

@dataclass
class AGoTNode:
    """Simple AGoT node with natural content"""
    id: int
    title: str
    content: str
    parent_ids: List[int]
    answer: Optional[str] = None
    is_final: bool = False
    depth: int = 0
    layer: int = 0


class AGoTGraph:
    """Simple graph structure for AGoT reasoning"""
    
    def __init__(self):
        self.nodes: Dict[int, AGoTNode] = {}
        self.edges: List[Tuple[int, int]] = []  # (parent_id, child_id)
        self.next_id = 0
    
    def add_node(self, title: str, content: str, parent_ids: Optional[List[int]] = None,
                 is_final: bool = False, depth: int = 0, layer: int = 0) -> int:
        """Add a node to the graph"""
        node_id = self.next_id
        self.next_id += 1
        
        parent_ids = parent_ids or []
        node = AGoTNode(
            id=node_id,
            title=title,
            content=content,
            parent_ids=parent_ids,
            is_final=is_final,
            depth=depth,
            layer=layer
        )
        
        self.nodes[node_id] = node
        
        # Add edges from parents to this node
        for parent_id in parent_ids:
            if parent_id in self.nodes:
                self.edges.append((parent_id, node_id))
        
        return node_id
    
    def get_ancestors_content(self, node_id: int) -> str:
        """Get content from all ancestor nodes - key AGoT feature"""
        if node_id not in self.nodes:
            return ""
        
        visited = set()
        ancestor_content = []
        
        def collect_ancestors(current_id):
            if current_id in visited:
                return
            visited.add(current_id)
            
            node = self.nodes[current_id]
            if node.answer:
                ancestor_content.append(f"[{node.title}]: {node.answer}")
            
            # Recursively collect from parents
            for parent_id in node.parent_ids:
                if parent_id in self.nodes:
                    collect_ancestors(parent_id)
        
        # Collect from all parents
        for parent_id in self.nodes[node_id].parent_ids:
            collect_ancestors(parent_id)
        
        return "\n\n".join(ancestor_content)
    
    def get_layer_nodes(self, layer: int) -> List[AGoTNode]:
        """Get all nodes in a specific layer"""
        return [node for node in self.nodes.values() if node.layer == layer]
    
    def to_dict(self) -> Dict[str, Any]:
        """Export graph for visualization"""
        return {
            "nodes": [
                {
                    "id": node.id,
                    "title": node.title,
                    "content": node.content,
                    "answer": node.answer,
                    "parent_ids": node.parent_ids,
                    "is_final": node.is_final,
                    "depth": node.depth,
                    "layer": node.layer
                }
                for node in self.nodes.values()
            ],
            "edges": self.edges
        }


class TrueAGoT(object):
    """
    True Adaptive Graph of Thought with Hybrid Web Search Integration
    
    Combines:
    - Adaptive graph-based reasoning
    - Proactive initial web search for context
    - Reactive confidence-based search triggers
    - Live visualization of both reasoning and search
    """
    
    def __init__(self, llm: LLMBase, max_layers: int = 3, max_nodes_per_layer: int = 3, 
                 verbose: bool = True, tool_registry=None, confidence_threshold: float = 0.7):
        self.llm = llm
        self.max_layers = max_layers
        self.max_nodes_per_layer = max_nodes_per_layer
        self.verbose = verbose
        self.graph = AGoTGraph()
        self.websocket = None
        self.tool_registry = tool_registry
        self.search_agent: Optional['SearchIntegratedAgent'] = None
        
        # Initialize hybrid search if available
        if SEARCH_AVAILABLE and tool_registry:
            self.search_agent = SearchIntegratedAgent(tool_registry, confidence_threshold)
            self.search_enabled = True
            if verbose:
                print("🔍 TrueAGoT web search integration enabled")
        else:
            self.search_enabled = False
            if verbose:
                print("📚 Running in knowledge-only mode (no web search)")
    
    async def run_async(self, question: str, websocket=None) -> str:
        """Run AGoT reasoning with adaptive graph construction, hybrid web search, and live visualization"""
        self.websocket = websocket
        # Note: websocket context for search integration is set by orchestrator via set_websocket_context
        
        if self.verbose:
            print(f"🧠 Starting AGoT reasoning with hybrid search for: {question[:100]}...")
        
        # Step 0: Proactive Web Search for Initial Context (if enabled)
        initial_context = ""
        if self.search_agent:
            if self.verbose:
                print("🔍 Performing proactive web search for initial context...")
            
            search_results = await self.search_agent._evaluate_search_need(
                reasoning_text=question,
                confidence=0.5,  # Neutral confidence for proactive search
                stage="initial"
            )
            
            if search_results:
                await self.search_agent._send_search_update(search_results)
                initial_context = await self.search_agent.search_orchestrator.integrate_search_results(
                    search_results, f"Initial question: {question}"
                )
                if self.verbose:
                    print(f"✅ Proactive search completed - {len(search_results.get('results', []))} sources integrated")
        
        # Step 1: Generate initial reasoning nodes (with search context)
        initial_nodes = await self._generate_initial_nodes(question, initial_context)
        
        # Add initial nodes to graph (layer 0) with live updates
        for i, (title, content) in enumerate(initial_nodes):
            node_id = self.graph.add_node(title, content, parent_ids=[], depth=0, layer=0)
            await self._send_graph_update()  # Send live update
            if self.verbose:
                print(f"📋 Initial node {node_id}: {title}")
        
        # Step 2: Execute initial nodes (with potential reactive search)
        await self._execute_layer(0, question)
        
        # Step 3: Adaptive layer generation
        for layer in range(1, self.max_layers):
            if self.verbose:
                print(f"\n🔄 Processing layer {layer}...")
            
            # Generate new nodes based on previous layer insights
            new_nodes = await self._generate_adaptive_nodes(question, layer)
            
            if not new_nodes:
                if self.verbose:
                    print(f"🛑 No new nodes generated, stopping at layer {layer}")
                break
            
            # Add new nodes to graph with live updates
            for title, content, parent_ids in new_nodes:
                node_id = self.graph.add_node(title, content, parent_ids, depth=0, layer=layer)
                await self._send_graph_update()  # Send live update
                if self.verbose:
                    print(f"📋 New node {node_id}: {title} (parents: {parent_ids})")
            
            # Execute the new layer
            await self._execute_layer(layer, question)
            
            # Check if we should generate final answer
            if await self._should_finalize(question, layer):
                break
        
        # Step 4: Generate final synthesis
        if self.verbose:
            print("🎯 Generating final synthesis...")
        
        final_answer = await self._generate_final_synthesis(question)
        
        # Add final synthesis node to graph
        all_node_ids = list(self.graph.nodes.keys())
        final_node_id = self.graph.add_node(
            "Final Synthesis", 
            final_answer[:200] + "...", 
            parent_ids=all_node_ids[-3:],  # Connect to last few nodes
            is_final=True,
            layer=self.max_layers
        )
        await self._send_graph_update()  # Send final update
        
        if self.verbose:
            print("✅ AGoT reasoning complete!")
        
        return final_answer
    
    async def _generate_initial_nodes(self, question: str, search_context: str = "") -> List[Tuple[str, str]]:
        """Generate initial reasoning nodes - natural breakdown"""
        
        context_section = ""
        if search_context:
            context_section = f"""
        
        External Knowledge Context:
        {search_context}
        
        Use this external knowledge to inform your task breakdown, but focus on reasoning tasks rather than just summarizing the search results.
        """
        
        prompt = f"""
        Analyze this complex question and break it down into 2-4 key reasoning tasks that need to be addressed.
        
        Question: {question}
        {context_section}
        
        For each task, provide:
        1. A clear title (max 8 words)
        2. A detailed description of what needs to be analyzed
        
        Format each task as:
        TASK: [title]
        CONTENT: [detailed description]
        
        Focus on different aspects or approaches that would provide comprehensive coverage.
        If external knowledge is available, incorporate it naturally into your reasoning framework.
        """
        
        messages = self.llm.format_prompt(
            "You are an expert at breaking down complex questions into manageable reasoning tasks.",
            prompt
        )
        
        response = await self.llm.generate_async(messages)
        response_text = str(response)
        
        # Extract tasks naturally (no rigid schemas!)
        tasks = []
        lines = response_text.split('\n')
        current_title = None
        current_content = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('TASK:'):
                if current_title and current_content:
                    tasks.append((current_title, current_content))
                current_title = line[5:].strip()
                current_content = None
            elif line.startswith('CONTENT:'):
                current_content = line[8:].strip()
        
        # Add the last task
        if current_title and current_content:
            tasks.append((current_title, current_content))
        
        return tasks[:self.max_nodes_per_layer]
    
    async def _execute_layer(self, layer: int, question: str):
        """Execute all nodes in a layer with context from ancestors"""
        
        layer_nodes = self.graph.get_layer_nodes(layer)
        
        # Execute nodes concurrently
        tasks = [self._execute_node(node, question) for node in layer_nodes]
        await asyncio.gather(*tasks)
    
    async def _execute_node(self, node: AGoTNode, question: str):
        """Execute a single node with full ancestor context and reactive web search"""
        
        # Get context from ancestor nodes - this is the key AGoT innovation!
        ancestor_context = self.graph.get_ancestors_content(node.id)
        
        context_section = ""
        if ancestor_context:
            context_section = f"""
            Previous Analysis Results:
            {ancestor_context}
            
            """
        
        # Initial reasoning attempt
        initial_prompt = f"""
        Original Question: {question}
        
        {context_section}Current Task: {node.title}
        Task Description: {node.content}
        
        Provide a thorough analysis for this specific task. Build on any previous analysis results if relevant.
        Be comprehensive but focused on this particular aspect.
        
        At the end, rate your confidence in this analysis on a scale of 0.0 to 1.0, and explain why.
        Format: CONFIDENCE: [score] - [brief explanation]
        """
        
        messages = self.llm.format_prompt(
            "You are an expert analyst. Provide detailed, well-reasoned analysis for the given task.",
            initial_prompt
        )
        
        initial_response = await self.llm.generate_async(messages)
        initial_response_text = str(initial_response)
        
        # Extract confidence score for reactive search decision
        confidence_score = self._extract_confidence_score(initial_response_text)
        
        # Check if reactive search is needed
        enhanced_response = initial_response_text
        if self.search_agent:
            search_results = await self.search_agent._evaluate_search_need(
                reasoning_text=f"{node.title}: {initial_response_text}",
                confidence=confidence_score,
                stage="analysis"
            )
            
            if search_results:
                await self.search_agent._send_search_update(search_results)
                
                # Integrate search results and re-analyze
                search_context = await self.search_agent.search_orchestrator.integrate_search_results(
                    search_results, initial_response_text
                )
                
                refinement_prompt = f"""
                Original Analysis:
                {initial_response_text}
                
                Additional External Knowledge:
                {search_context}
                
                Now provide a refined analysis that integrates this external knowledge with your original reasoning.
                Focus on how the external information validates, contradicts, or expands your initial analysis.
                """
                
                refinement_messages = self.llm.format_prompt(
                    "You are an expert analyst. Refine your analysis by integrating external knowledge.",
                    refinement_prompt
                )
                
                enhanced_response_obj = await self.llm.generate_async(refinement_messages)
                enhanced_response = str(enhanced_response_obj)
                
                if self.verbose:
                    print(f"🔍 Node {node.id} enhanced with reactive search")
        
        node.answer = str(enhanced_response)
        
        # Send live update when node is completed
        await self._send_graph_update()
        
        if self.verbose:
            print(f"✅ Executed node {node.id}: {node.title}")
    
    def _extract_confidence_score(self, response: str) -> float:
        """Extract confidence score from LLM response"""
        try:
            # Look for CONFIDENCE: pattern
            lines = response.split('\n')
            for line in lines:
                if 'CONFIDENCE:' in line.upper():
                    # Extract number between 0.0 and 1.0
                    import re
                    match = re.search(r'(\d+\.?\d*)', line)
                    if match:
                        score = float(match.group(1))
                        # Normalize if needed
                        if score > 1.0:
                            score = score / 10.0
                        return max(0.0, min(1.0, score))
            
            # Fallback: analyze response for uncertainty indicators
            uncertainty_indicators = ['uncertain', 'unclear', 'might', 'possibly', 'perhaps', 'maybe']
            confidence_indicators = ['certain', 'clear', 'definitely', 'confirmed', 'established']
            
            text_lower = response.lower()
            uncertainty_count = sum(1 for indicator in uncertainty_indicators if indicator in text_lower)
            confidence_count = sum(1 for indicator in confidence_indicators if indicator in text_lower)
            
            # Simple heuristic
            if confidence_count > uncertainty_count:
                return 0.8
            elif uncertainty_count > confidence_count:
                return 0.4
            else:
                return 0.6
                
        except Exception:
            return 0.6  # Default moderate confidence
    
    async def _generate_adaptive_nodes(self, question: str, layer: int) -> List[Tuple[str, str, List[int]]]:
        """Generate new nodes based on insights from previous layers - adaptive!"""
        
        # Get all previous analysis
        previous_analysis = []
        for prev_layer in range(layer):
            layer_nodes = self.graph.get_layer_nodes(prev_layer)
            for node in layer_nodes:
                if node.answer:
                    previous_analysis.append(f"[{node.title}]: {node.answer}")
        
        if not previous_analysis:
            return []
        
        analysis_context = "\n\n".join(previous_analysis)
        
        prompt = f"""
        Original Question: {question}
        
        Previous Analysis:
        {analysis_context}
        
        Based on the analysis so far, what new reasoning tasks should we explore to get closer to a comprehensive answer?
        
        Consider:
        - What gaps exist in the current analysis?
        - What deeper questions have emerged?
        - What connections between different aspects should be explored?
        - What synthesis or integration tasks are needed?
        
        Generate 1-3 new reasoning tasks. For each task, specify which previous analysis results it should build on.
        
        Format as:
        TASK: [title]
        CONTENT: [description]
        BUILDS_ON: [comma-separated list of relevant previous task titles]
        
        If the analysis is comprehensive enough, respond with "READY_FOR_SYNTHESIS" instead.
        """
        
        messages = self.llm.format_prompt(
            "You are an adaptive reasoning coordinator. Identify what analysis is still needed.",
            prompt
        )
        
        response = await self.llm.generate_async(messages)
        response_text = str(response)
        
        if "READY_FOR_SYNTHESIS" in response_text:
            return []
        
        # Parse new tasks naturally
        new_tasks = []
        lines = response_text.split('\n')
        current_task = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('TASK:'):
                if current_task.get('title') and current_task.get('content'):
                    new_tasks.append(self._finalize_task(current_task))
                current_task = {'title': line[5:].strip()}
            elif line.startswith('CONTENT:'):
                current_task['content'] = line[8:].strip()
            elif line.startswith('BUILDS_ON:'):
                current_task['builds_on'] = line[10:].strip()
        
        # Add the last task
        if current_task.get('title') and current_task.get('content'):
            new_tasks.append(self._finalize_task(current_task))
        
        return new_tasks[:self.max_nodes_per_layer]
    
    def _finalize_task(self, task_data: Dict[str, str]) -> Tuple[str, str, List[int]]:
        """Convert task data to final format with parent IDs"""
        title = task_data['title']
        content = task_data['content']
        builds_on = task_data.get('builds_on', '')
        
        # Find parent node IDs based on titles mentioned
        parent_ids = []
        if builds_on:
            mentioned_titles = [t.strip() for t in builds_on.split(',')]
            for node in self.graph.nodes.values():
                if any(mentioned.lower() in node.title.lower() for mentioned in mentioned_titles):
                    parent_ids.append(node.id)
        
        return title, content, parent_ids
    
    async def _should_finalize(self, question: str, current_layer: int) -> bool:
        """Decide if we have enough analysis to provide final answer"""
        
        # Simple heuristic: if we're at max layers or have good coverage
        if current_layer >= self.max_layers - 1:
            return True
        
        # Could add more sophisticated logic here
        return False
    
    async def _generate_final_synthesis(self, question: str) -> str:
        """Generate final answer by synthesizing across the entire graph"""
        
        # Collect all analysis from the graph
        all_analysis = []
        for node in self.graph.nodes.values():
            if node.answer:
                all_analysis.append(f"**{node.title}**\n{node.answer}")
        
        analysis_content = "\n\n---\n\n".join(all_analysis)
        
        prompt = f"""
        Original Question: {question}
        
        Complete Analysis Results:
        {analysis_content}
        
        Now synthesize all of this analysis into a comprehensive, well-structured final answer.
        
        Your response should:
        - Directly answer the original question
        - Integrate insights from all the analysis
        - Be well-organized and clear
        - Include supporting evidence from the analysis
        - Provide actionable conclusions where appropriate
        """
        
        messages = self.llm.format_prompt(
            "You are a synthesis expert. Combine multiple analyses into a comprehensive final answer.",
            prompt
        )
        
        response = await self.llm.generate_async(messages)
        return str(response)
    
    def get_graph_data(self) -> Dict[str, Any]:
        """Get graph data for visualization"""
        return self.graph.to_dict()
    
    async def _send_graph_update(self):
        """Send live graph update via WebSocket"""
        if self.websocket:
            try:
                graph_data = self.get_graph_data()
                await self.websocket.send_json({
                    "type": "graph_data",
                    "data": graph_data
                })
            except Exception as e:
                if self.verbose:
                    print(f"WebSocket send failed: {e}")
    
    def run(self, question: str) -> str:
        """Synchronous wrapper"""
        return asyncio.run(self.run_async(question))