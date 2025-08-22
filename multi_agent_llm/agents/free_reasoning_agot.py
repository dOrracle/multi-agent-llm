"""
Free Reasoning AGoT - No rigid schemas, natural thinking
"""
import asyncio
import re
from typing import List, Dict, Any, Optional, AsyncGenerator
from multi_agent_llm.llm import LLMBase


class FreeReasoningAGoT:
    """Simplified AGoT that uses natural reasoning without rigid schemas"""
    
    def __init__(self, llm: LLMBase, max_tasks: int = 5, verbose: bool = True, tool_registry=None):
        self.llm = llm
        self.max_tasks = max_tasks
        self.verbose = verbose
        self.websocket = None
        self.node_counter = 0
        self.tool_registry = tool_registry
        self.search_enabled = tool_registry is not None
        
        if self.search_enabled and verbose:
            print("🔍 FreeReasoningAGoT web search integration enabled")
    
    async def run_async(self, question: str, websocket=None) -> str:
        """Run the free reasoning process with optional live visualization"""
        self.websocket = websocket
        self.node_counter = 0
        
        if self.verbose:
            print(f"🧠 Starting free reasoning for: {question[:100]}...")
        
        # Create initial question node
        question_node = await self._create_node("question", question, "processing")
        
        # Step 1: Break down the question naturally
        await self._send_update("log", "🧠 Breaking down question into key tasks...")
        tasks = await self._break_down_question(question)
        
        # Update question node and create breakdown node
        await self._update_node(question_node["id"], "completed")
        breakdown_node = await self._create_node("breakdown", f"Identified {len(tasks)} key tasks", "completed")
        await self._create_edge(question_node["id"], breakdown_node["id"])
        
        if self.verbose:
            print(f"📋 Identified {len(tasks)} key tasks")
        
        # Step 2: Address each task naturally
        task_results = []
        task_nodes = []
        
        for i, task in enumerate(tasks):
            if self.verbose:
                print(f"🔍 Working on task {i+1}: {task[:50]}...")
            
            # Create task node
            task_node = await self._create_node("task", f"Task {i+1}: {task[:60]}...", "processing")
            await self._create_edge(breakdown_node["id"], task_node["id"])
            task_nodes.append(task_node)
            
            await self._send_update("log", f"🔍 Analyzing task {i+1}: {task[:50]}...")
            
            result = await self._address_task(task, question, task_results)
            task_results.append(result)
            
            # Update task node as completed
            await self._update_node(task_node["id"], "completed", f"Task {i+1}: {task[:60]}...\n\nAnalysis: {result[:100]}...")
        
        # Step 3: Synthesize into final answer
        await self._send_update("log", "🎯 Synthesizing final answer...")
        synthesis_node = await self._create_node("synthesis", "Synthesizing final answer...", "processing")
        
        # Connect all task nodes to synthesis
        for task_node in task_nodes:
            await self._create_edge(task_node["id"], synthesis_node["id"])
        
        if self.verbose:
            print("🎯 Synthesizing final answer...")
        
        final_answer = await self._synthesize_answer(question, tasks, task_results)
        
        # Create final answer node
        final_node = await self._create_node("final", "Final Answer", "completed", final_answer[:200] + "...")
        await self._update_node(synthesis_node["id"], "completed")
        await self._create_edge(synthesis_node["id"], final_node["id"])
        
        if self.verbose:
            print("✅ Free reasoning complete!")
        
        await self._send_update("log", "✅ Free reasoning complete!")
        
        return final_answer
    
    async def _break_down_question(self, question: str) -> List[str]:
        """Break down question into key tasks naturally"""
        
        prompt = f"""
        Analyze this complex question and break it down into the key tasks that need to be addressed:
        
        Question: {question}
        
        Think step by step about what needs to be researched, analyzed, or considered to provide a comprehensive answer.
        
        List each key task on a separate line starting with "TASK:"
        Aim for 3-5 main tasks that cover all important aspects.
        """
        
        messages = self.llm.format_prompt(
            "You are an expert at breaking down complex questions into manageable tasks.",
            prompt
        )
        
        response = await self.llm.generate_async(messages)
        response_text = str(response)
        
        # Simple extraction - no rigid parsing
        tasks = []
        for line in response_text.split('\n'):
            line = line.strip()
            if line.upper().startswith('TASK:'):
                task = line[5:].strip()  # Remove "TASK:" prefix
                if task:
                    tasks.append(task)
        
        # Fallback if no tasks found with prefix
        if not tasks:
            # Try to extract from numbered lists or bullet points
            for line in response_text.split('\n'):
                line = line.strip()
                if re.match(r'^\d+\.', line) or line.startswith('•') or line.startswith('-'):
                    clean_task = re.sub(r'^\d+\.\s*|^[•-]\s*', '', line).strip()
                    if clean_task and len(clean_task) > 10:
                        tasks.append(clean_task)
        
        return tasks[:self.max_tasks]  # Limit number of tasks
    
    async def _address_task(self, task: str, original_question: str, previous_results: List[str]) -> str:
        """Address a specific task naturally"""
        
        context = ""
        if previous_results:
            context = f"""
            Previous analysis results:
            {chr(10).join([f"- {result[:200]}..." for result in previous_results[-2:]])}
            """
        
        prompt = f"""
        Original question: {original_question}
        
        {context}
        
        Now focus specifically on this task:
        {task}
        
        Provide a thorough analysis addressing this specific aspect. Be comprehensive but focused.
        Consider relevant data, trends, factors, and implications.
        """
        
        messages = self.llm.format_prompt(
            "You are an expert analyst. Provide detailed, well-reasoned analysis for the specific task given.",
            prompt
        )
        
        response = await self.llm.generate_async(messages)
        return str(response)
    
    async def _synthesize_answer(self, question: str, tasks: List[str], results: List[str]) -> str:
        """Synthesize all results into a comprehensive final answer"""
        
        task_summary = "\n".join([f"{i+1}. {task}" for i, task in enumerate(tasks)])
        
        analysis_summary = ""
        for i, result in enumerate(results):
            analysis_summary += f"\n\nAnalysis {i+1} ({tasks[i][:50]}...):\n{result}"
        
        prompt = f"""
        Original question: {question}
        
        Key tasks that were analyzed:
        {task_summary}
        
        Detailed analysis results:
        {analysis_summary}
        
        Now synthesize all of this analysis into a comprehensive, well-structured final answer.
        
        Your response should:
        - Directly answer the original question
        - Integrate insights from all the analysis
        - Provide clear recommendations or conclusions
        - Be well-organized and easy to follow
        - Include key supporting evidence from the analysis
        """
        
        messages = self.llm.format_prompt(
            "You are a synthesis expert. Combine multiple analyses into clear, comprehensive conclusions.",
            prompt
        )
        
        response = await self.llm.generate_async(messages)
        return str(response)
    
    def run(self, question: str) -> str:
        """Synchronous wrapper"""
        return asyncio.run(self.run_async(question))
    
    async def _create_node(self, node_type: str, text: str, status: str, full_content: Optional[str] = None) -> Dict[str, Any]:
        """Create a new node and send to frontend"""
        self.node_counter += 1
        node = {
            "id": f"free_reasoning_node_{self.node_counter}",
            "text": text,
            "type": node_type,
            "status": status,
            "meta": {
                "full_content": full_content or text,
                "reasoning_type": "free_reasoning",
                "step": self.node_counter
            }
        }
        
        if self.websocket:
            await self._send_update("node_created", node)
        
        return node
    
    async def _update_node(self, node_id: str, status: str, text: Optional[str] = None):
        """Update an existing node"""
        update = {
            "id": node_id,
            "status": status
        }
        if text:
            update["text"] = text
        
        if self.websocket:
            await self._send_update("node_updated", update)
    
    async def _create_edge(self, from_id: str, to_id: str):
        """Create an edge between nodes"""
        edge = {
            "from": from_id,
            "to": to_id,
            "type": "reasoning_flow"
        }
        
        if self.websocket:
            await self._send_update("edge_created", edge)
    
    async def _send_update(self, update_type: str, data: Any):
        """Send update via WebSocket if available"""
        if self.websocket:
            try:
                # Handle different data types
                if update_type == "log":
                    payload = {"type": update_type, "message": data}
                else:
                    payload = {"type": update_type, "data": data}
                
                await self.websocket.send_json(payload)
            except Exception as e:
                if self.verbose:
                    print(f"WebSocket send failed: {e}")


# Simple text extraction utilities
def extract_key_points(text: str) -> List[str]:
    """Extract key points from natural text"""
    points = []
    for line in text.split('\n'):
        line = line.strip()
        if (line.startswith('•') or line.startswith('-') or 
            re.match(r'^\d+\.', line) or line.startswith('*')):
            clean_point = re.sub(r'^\d+\.\s*|^[•\-*]\s*', '', line).strip()
            if clean_point:
                points.append(clean_point)
    return points

def extract_conclusion(text: str) -> str:
    """Extract conclusion from natural text"""
    # Look for conclusion indicators
    conclusion_patterns = [
        r'conclusion:?\s*(.*?)(?:\n|$)',
        r'in conclusion:?\s*(.*?)(?:\n|$)',
        r'final recommendation:?\s*(.*?)(?:\n|$)',
        r'recommendation:?\s*(.*?)(?:\n|$)',
        r'summary:?\s*(.*?)(?:\n|$)'
    ]
    
    for pattern in conclusion_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # Fallback: return last paragraph
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return paragraphs[-1] if paragraphs else text[:200] + "..."