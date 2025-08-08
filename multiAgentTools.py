#!/usr/bin/env python3
"""
multiAgentTools.py - Secure Python tool executor for multi-agent-llm
Handles execution of AGOT, AIOT, and GIOT tools safely
"""
import sys
import json
import asyncio
import os
from typing import Dict, Any, Optional, Type
from pathlib import Path
# Import web_search_handler for Tavily integration
from web_search_tool import web_search_handler


# Add the mcp-server/python directory to the system path
sys.path.append(str(Path(__file__).parent.parent / 'mcp-server' / 'python'))

# Load .env for all config
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

import os  # Added import for os

# Utility to check debug/verbose env
def is_debug():
    return os.getenv('DEBUG', 'false').lower() in ('1', 'true', 'yes', 'on')

def get_verbose_level():
    v = os.getenv('VERBOSE_LOGGING', '0').lower()
    if v in ('1', 'true', 'yes', 'on'):
        return 3
    try:
        return int(v)
    except Exception:
        return 0



# Load tool_registry.json for standardized tool names and schemas
tool_registry_path = Path(__file__).parent.parent / 'tool_registry.json'
with open(tool_registry_path, 'r') as f:
    tool_registry = json.load(f)

# Import multi-agent-llm components
try:
    from multi_agent_llm import OpenAILLM, AGOT, AIOT, GIOT, GeminiLLM
except ImportError as e:
    print(json.dumps({"error": f"Failed to import multi-agent-llm: {e}"}), file=sys.stderr)
    sys.exit(1)

# Import pydantic for schema validation
try:
    from pydantic import BaseModel, Field
except ImportError as e:
    print(json.dumps({"error": f"Failed to import pydantic: {e}"}), file=sys.stderr)
    sys.exit(1)

class QueryAnswer(BaseModel):
    """Standard answer schema for multi-agent responses"""
    explanation: str = Field(description="Explanation of the answer.")
    answer: str = Field(description="Final answer or result.")
    confidence: Optional[float] = Field(description="Confidence score if available.")


class MultiAgentToolExecutor:
    """Secure executor for multi-agent LLM tools"""

    def __init__(self):
        # Use tool_registry for tool mapping
        self.tools = {}
        for tool_name, tool_info in tool_registry.items():
            pytool = tool_info.get('python_tool')
            if pytool == 'web_search':
                self.tools[pytool] = self.execute_web_search
            elif pytool:
                self.tools[pytool] = getattr(self, f"execute_{pytool.replace('_query','')}")
        self.tools['health_check'] = self.health_check

    async def execute_web_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Tavily web search tool"""
        return await web_search_handler(args)

    def create_llm(self, provider: str, model_name: str, temperature: float, api_key: Optional[str] = None):
        """Create appropriate LLM instance based on provider"""
        if provider.lower() == 'gemini':
            return GeminiLLM(
                model_name=model_name,
                api_key=api_key,
                temperature=temperature
            )
        elif provider.lower() == 'openai':
            # Set API key in environment for OpenAI
            if api_key:
                os.environ['OPENAI_API_KEY'] = api_key
            return OpenAILLM(
                model_name=model_name,
                temperature=temperature
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}. Supported: openai, gemini")

    async def health_check(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Health check to validate the bridge connection"""
        return {
            "status": "ok",
            "timestamp": str(asyncio.get_event_loop().time()),
            "available_tools": list(self.tools.keys())
        }

    async def execute_agot(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Adaptive Graph of Thoughts"""
        try:
            if is_debug():
                print("[DEBUG] Starting AGOT execution")
            # Extract parameters with environment variable defaults
            question = args.get('question')
            if is_debug():
                print(f"[DEBUG] Question: {question}")
            provider = args.get('llm_provider') or os.getenv('DEFAULT_LLM_PROVIDER', 'openai')
            if is_debug():
                print(f"[DEBUG] Provider: {provider}")

            # Get API keys
            openai_key = os.getenv('OPENAI_API_KEY')
            gemini_key = os.getenv('GEMINI_API_KEY')
            if is_debug():
                print(f"[DEBUG] OpenAI Key found: {'Yes' if openai_key else 'No'}")
                print(f"[DEBUG] Gemini Key found: {'Yes' if gemini_key else 'No'}")

            # Determine model and API key based on provider
            if provider.lower() == 'gemini':
                model_name = args.get('model_name') or os.getenv('DEFAULT_MODEL', 'gemini-1.5-pro')
                api_key = gemini_key
            else:
                model_name = args.get('model_name') or os.getenv('DEFAULT_MODEL', 'gpt-4o')
                api_key = openai_key

            if is_debug():
                print(f"[DEBUG] Model: {model_name}")
                print(f"[DEBUG] API Key being used: {'Yes' if api_key else 'No'}")

            temperature = float(args.get('temperature', os.getenv('DEFAULT_TEMPERATURE', 0.7)))
            max_depth = int(args.get('max_depth', os.getenv('AGOT_MAX_DEPTH', 2)))
            max_num_layers = int(args.get('max_num_layers', os.getenv('AGOT_MAX_NUM_LAYERS', 3)))
            max_new_tasks = int(args.get('max_new_tasks', os.getenv('AGOT_MAX_NEW_TASKS', 3)))
            max_concurrent_tasks = int(args.get('max_concurrent_tasks', os.getenv('AGOT_MAX_CONCURRENT_TASKS', 10)))

            if not question:
                raise ValueError("question is required")
            if not api_key:
                raise ValueError(f"{provider} API key is required (provide via parameter or environment variable)")

            # Create LLM instance
            if is_debug():
                print("[DEBUG] Creating LLM instance")
            llm = self.create_llm(provider, model_name, temperature, api_key)
            if is_debug():
                print("[DEBUG] LLM instance created")

            # Initialize AGOT
            if is_debug():
                print("[DEBUG] Initializing AGOT")
            agot = AGOT(
                llm=llm,
                max_depth=max_depth,
                max_num_layers=max_num_layers,
                max_new_tasks=max_new_tasks,
                max_concurrent_tasks=max_concurrent_tasks,
                verbose=get_verbose_level(),
            )
            if is_debug():
                print("[DEBUG] AGOT initialized")

            # Execute query
            if is_debug():
                print("[DEBUG] Executing AGOT query")
            response = await agot.run_async(question, schema=QueryAnswer)  # type: ignore
            if is_debug():
                print("[DEBUG] AGOT query executed")

            return {
                "success": True,
                "method": "AGOT",
                "provider": provider,
                "model": model_name,
                "question": question,
                "answer": response.final_answer.answer if hasattr(response.final_answer, 'answer') else str(response.final_answer),
                "explanation": response.final_answer.explanation if hasattr(response.final_answer, 'explanation') else "",
                "confidence": getattr(response.final_answer, 'confidence', None),
                "parameters": {
                    "provider": provider,
                    "model_name": model_name,
                    "temperature": temperature,
                    "max_depth": max_depth,
                    "max_num_layers": max_num_layers,
                    "max_new_tasks": max_new_tasks,
                    "max_concurrent_tasks": max_concurrent_tasks
                }
            }

        except Exception as e:
            import traceback
            print(f"[ERROR] Error in AGOT execution: {e}")
            return {
                "success": False,
                "method": "AGOT",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "question": args.get('question', 'NA')
            }

    async def execute_aiot(self, params):
        """Execute Autonomous Iteration of Thought (AIOT) reasoning"""
        try:
            # Extract parameters with defaults
            question = params.get("question", "")
            provider = params.get("llm_provider") or os.getenv('DEFAULT_LLM_PROVIDER', 'openai')

            # Get API keys
            openai_key = os.getenv('OPENAI_API_KEY')
            gemini_key = os.getenv('GEMINI_API_KEY')

            # Determine model and API key based on provider
            if provider.lower() == 'gemini':
                model_name = params.get("model_name") or os.getenv('DEFAULT_MODEL', 'gemini-1.5-pro')
                api_key = gemini_key
            else:
                model_name = params.get("model_name") or os.getenv('DEFAULT_MODEL', 'gpt-4o')
                api_key = openai_key

            temperature = float(params.get("temperature", os.getenv('DEFAULT_TEMPERATURE', 0.7)))
            max_iterations = int(params.get("max_iterations", os.getenv('DEFAULT_MAX_ITERATIONS', 5)))
            convergence_threshold = float(params.get("convergence_threshold", 0.8))

            if not question:
                return {"success": False, "error": "Question is required"}
            if not api_key:
                return {"success": False, "error": f"{provider} API key is required"}

            # Create LLM instance
            llm = self.create_llm(provider, model_name, temperature, api_key)

            # Create AIOT agent
            aiot = AIOT(
                llm=llm,
                iterations=max_iterations,
                answer_schema=QueryAnswer
            )

            # Execute reasoning with proper async handling
            if hasattr(aiot, 'run_async'):
                response = await aiot.run_async(question)
            else:
                # Fallback to sync method
                response = aiot.run(question)

            return {
                "success": True,
                "method": "AIOT",
                "provider": provider,
                "model": model_name,
                "question": question,
                "answer": response.answer.answer if hasattr(response.answer, 'answer') else str(response.answer),
                "explanation": response.answer.explanation if hasattr(response.answer, 'explanation') else "",
                "confidence": getattr(response.answer, 'confidence', None),
                "parameters": {
                    "provider": provider,
                    "model_name": model_name,
                    "temperature": temperature,
                    "max_iterations": max_iterations,
                    "convergence_threshold": convergence_threshold
                }
            }

        except Exception as e:
            import traceback
            return {
                "success": False, 
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    async def execute_giot(self, params):
        """Execute Guided Iteration of Thought (GIOT) reasoning"""
        try:
            # Extract parameters with defaults
            question = params.get("question", "")
            provider = params.get("llm_provider") or os.getenv('DEFAULT_LLM_PROVIDER', 'openai')

            # Get API keys
            openai_key = os.getenv('OPENAI_API_KEY')
            gemini_key = os.getenv('GEMINI_API_KEY')

            # Determine model and API key based on provider
            if provider.lower() == 'gemini':
                model_name = params.get("model_name") or os.getenv('DEFAULT_MODEL', 'gemini-1.5-pro')
                api_key = gemini_key
            else:
                model_name = params.get("model_name") or os.getenv('DEFAULT_MODEL', 'gpt-4o')
                api_key = openai_key

            temperature = float(params.get("temperature", os.getenv('DEFAULT_TEMPERATURE', 0.7)))
            num_iterations = int(params.get("num_iterations", os.getenv('DEFAULT_NUM_ITERATIONS', 3)))
            guidance_prompt = params.get("guidance_prompt", "Think step by step and be thorough.")

            if not question:
                return {"success": False, "error": "Question is required"}
            if not api_key:
                return {"success": False, "error": f"{provider} API key is required"}

            # Create LLM instance
            llm = self.create_llm(provider, model_name, temperature, api_key)

            # Create GIOT agent
            giot = GIOT(
                llm=llm,
                iterations=num_iterations,
                answer_schema=QueryAnswer
            )

            # Execute query with proper async handling
            if hasattr(giot, 'run_async'):
                response = await giot.run_async(question)
            else:
                # Fallback to sync method
                response = giot.run(question)

            return {
                "success": True,
                "method": "GIOT",
                "provider": provider,
                "model": model_name,
                "question": question,
                "answer": response.answer.answer if hasattr(response.answer, 'answer') else str(response.answer),
                "explanation": response.answer.explanation if hasattr(response.answer, 'explanation') else "",
                "confidence": getattr(response.answer, 'confidence', None),
                "parameters": {
                    "provider": provider,
                    "model_name": model_name,
                    "temperature": temperature,
                    "num_iterations": num_iterations,
                    "guidance_prompt": guidance_prompt
                }
            }

        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    async def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by standardized name (from tool_registry)"""
        # Map standardized tool name to python_tool
        python_tool = None
        for reg_name, reg_info in tool_registry.items():
            if tool_name == reg_name:
                python_tool = reg_info['python_tool']
                break
        if not python_tool or python_tool not in self.tools:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}",
                "available_tools": list(tool_registry.keys())
            }
        try:
            return await self.tools[python_tool](args)
        except Exception as e:
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}",
                "tool": tool_name
            }


async def main():
    """Main entry point for tool execution"""
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: python multiAgentTools.py <tool_name>"}), file=sys.stdout)
        sys.exit(0)

    tool_name = sys.argv[1]

    try:
        # Read arguments from stdin
        input_data = sys.stdin.read()
        args = json.loads(input_data) if input_data.strip() else {}

        # Execute tool using standardized name
        executor = MultiAgentToolExecutor()
        result = await executor.execute_tool(tool_name, args)

        # Output result
        print(json.dumps(result, indent=2))

    except json.JSONDecodeError as e:
        import traceback
        print(json.dumps({"error": f"Invalid JSON input: {e}", "traceback": traceback.format_exc()}), file=sys.stdout)
        sys.exit(0)
    except Exception as e:
        import traceback
        print(json.dumps({"error": f"Execution failed: {e}", "traceback": traceback.format_exc()}), file=sys.stdout)
        sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())
