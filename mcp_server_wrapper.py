#!/usr/bin/env python3

import asyncio
import json
import sys
import traceback
from typing import Any, Dict

from multi_agent_llm.gemini_llm import GeminiLLM
from multi_agent_llm.agents.adaptive_graph_of_thoughts.AGoT import AGOT
from multi_agent_llm.agents.iteration_of_thought import AIOT, GIOT


class ReasoningEngine:
    def __init__(self):
        self.llm = GeminiLLM()
        
    async def handle_agot(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Adaptive Graph of Thoughts reasoning"""
        try:
            agot = AGOT(
                llm=self.llm,
                max_new_tasks=params.get('max_new_tasks', 3),
                max_depth=params.get('max_depth', 2),
                max_num_layers=params.get('max_num_layers', 3),
                verbose=params.get('verbose', 1)
            )
            
            result = await agot.run_async(params['query'])
            
            # Summarize the graph for lightweight response
            graph_summary = f"Generated {len(result.graph)} reasoning nodes"
            if result.graph:
                depths = [node.depth for node in result.graph]
                layers = [node.layer for node in result.graph]
                graph_summary += f" across {max(depths)+1} depths and {max(layers)+1} layers"
            
            return {
                'final_answer': result.final_answer,
                'graph_summary': graph_summary,
                'total_nodes': len(result.graph),
                'graph_data': [node.model_dump() for node in result.graph] if params.get('include_graph') else None
            }
            
        except Exception as e:
            raise Exception(f"AGOT reasoning failed: {str(e)}")

    async def handle_iterative(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle iterative reasoning (AIOT/GIOT)"""
        try:
            method = params.get('method', 'aiot').lower()
            iterations = params.get('iterations', 5)
            
            if method == 'aiot':
                reasoner = AIOT(self.llm, iterations=iterations)
            elif method == 'giot':
                reasoner = GIOT(self.llm, iterations=iterations)
            else:
                raise ValueError(f"Unknown iterative method: {method}")
            
            result = await reasoner.run_async(params['query'])
            
            # Summarize thoughts for lightweight response
            thoughts_summary = ""
            for i, thought in enumerate(result.thoughts, 1):
                thoughts_summary += f"**Iteration {i}:**\n"
                thoughts_summary += f"Brain: {thought.brain_thought[:100]}...\n"
                thoughts_summary += f"Response: {str(thought.llm_response)[:100]}...\n\n"
            
            return {
                'method': method.upper(),
                'answer': result.answer,
                'total_iterations': len(result.thoughts),
                'thoughts_summary': thoughts_summary,
                'full_thoughts': [t.model_dump() for t in result.thoughts] if params.get('include_thoughts') else None
            }
            
        except Exception as e:
            raise Exception(f"Iterative reasoning failed: {str(e)}")

    async def handle_sequential(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle sequential step-by-step reasoning"""
        try:
            query = params['query']
            max_steps = params.get('max_steps', 8)
            
            # Simple sequential reasoning implementation
            steps = []
            current_context = f"Question: {query}\n\nLet me think through this step by step."
            
            for step_num in range(1, max_steps + 1):
                step_prompt = f"{current_context}\n\nStep {step_num}: What should I consider next?"
                
                messages = self.llm.format_prompt(
                    "You are helping with step-by-step reasoning. Provide one clear reasoning step.",
                    step_prompt
                )
                
                step_response = await self.llm.generate_async(messages)
                steps.append(step_response)
                current_context += f"\nStep {step_num}: {step_response}"
                
                # Check if we have enough to answer
                if step_num >= 3:
                    check_prompt = f"{current_context}\n\nCan I now provide a final answer to: {query}? Answer yes or no."
                    messages = self.llm.format_prompt("Determine if enough reasoning has been done.", check_prompt)
                    can_answer = await self.llm.generate_async(messages)
                    
                    # can_answer may be a BaseModel or string
                    if isinstance(can_answer, str):
                        can_answer_str = can_answer
                    elif hasattr(can_answer, 'model_dump_json'):
                        can_answer_str = can_answer.model_dump_json()
                    else:
                        can_answer_str = str(can_answer)
                    if "yes" in can_answer_str.lower():
                        break
            
            # Generate final answer
            final_prompt = f"{current_context}\n\nBased on the above reasoning, what is the final answer to: {query}"
            messages = self.llm.format_prompt("Provide a final answer based on the reasoning steps.", final_prompt)
            final_answer = await self.llm.generate_async(messages)
            
            return {
                'final_answer': final_answer,
                'steps': steps,
                'total_steps': len(steps)
            }
            
        except Exception as e:
            raise Exception(f"Sequential reasoning failed: {str(e)}")

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming request and route to appropriate handler"""
        try:
            method = request.get('method')
            params = request.get('params', {})
            
            if method == 'agot':
                result = await self.handle_agot(params)
            elif method == 'iterative':
                result = await self.handle_iterative(params)
            elif method == 'sequential':
                result = await self.handle_sequential(params)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            return {
                'request_id': request.get('request_id'),
                'result': result,
                'error': None
            }
            
        except Exception as e:
            return {
                'request_id': request.get('request_id'),
                'result': None,
                'error': str(e)
            }


async def main():
    """Main event loop for processing requests"""
    engine = ReasoningEngine()
    
    # Process requests from stdin
    while True:
        try:
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                break
                
            line = line.strip()
            if not line:
                continue
            
            try:
                request = json.loads(line)
                response = await engine.process_request(request)
                
                # Send response to stdout
                print(json.dumps(response), flush=True)
                
            except json.JSONDecodeError:
                error_response = {
                    'request_id': None,
                    'result': None,
                    'error': 'Invalid JSON request'
                }
                print(json.dumps(error_response), flush=True)
                
        except Exception as e:
            error_response = {
                'request_id': None,
                'result': None,
                'error': f'Processing error: {str(e)}'
            }
            print(json.dumps(error_response), flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)