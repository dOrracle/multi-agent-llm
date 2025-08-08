import os
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime

# Load environment variables from the correct .env file location
try:
    from dotenv import load_dotenv
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Go up one level to project root
    env_path = os.path.join(project_root, '.env')
    load_dotenv(env_path)
except ImportError:
    pass

async def should_search_web(llm, question: str, context: str = "") -> bool:
    """
    Utility function to decide if web search is needed for a given question/context.
    Uses the LLM to make the decision with timestamp awareness.
    """
    try:
        # Add current timestamp context to the decision
        current_time = datetime.now().strftime('%B %d, %Y at %H:%M UTC')
        
        decision_prompt = f"""Current date and time: {current_time}

Given the current question: "{question}"
Context: {context}

Do you need to search the web for up-to-date information to answer this question accurately? 

Consider if the question requires:
- Recent events, news, or developments that have already happened
- Current statistics or data
- Real-time information
- Latest research or findings
- Past events that would be documented online (like news from specific dates that have already occurred)

IMPORTANT: If the question asks about events on specific dates that are in the PAST (before {current_time}), you should search the web to find actual news reports from those dates.

If the question asks about future events that haven't happened yet, do NOT search the web.

Reply with exactly "yes" or "no"."""

        # Use the LLM's generate method to get the decision
        if hasattr(llm, 'generate_async'):
            response = await llm.generate_async([{"role": "user", "content": decision_prompt}])
        elif hasattr(llm, 'generate'):
            response = llm.generate([{"role": "user", "content": decision_prompt}])
        else:
            # Fallback for different LLM interfaces
            response = await llm.agenerate([[decision_prompt]])
            response = response.generations[0][0].text if response.generations else "no"
        
        # Extract the decision from the response
        decision_text = str(response).lower().strip()
        return "yes" in decision_text and "no" not in decision_text.replace("yes", "")
        
    except Exception as e:
        # If decision fails, default to no search to avoid breaking the agent
        return False

async def get_web_context(question: str, num_results: int = 3) -> Dict[str, Any]:
    """
    Get web search context for a question. Returns formatted context or empty dict if failed.
    """
    try:
        search_result = await web_search_handler({
            "query": question,
            "num_results": num_results,
            "search_depth": "basic"
        })
        
        if search_result.get("success", False):
            # Format the results into a concise context
            context = f"Web search results for '{question}':\n"
            if search_result.get("answer"):
                context += f"Summary: {search_result['answer']}\n"
            
            for i, result in enumerate(search_result.get("results", [])[:3], 1):
                context += f"{i}. {result.get('title', 'No title')}\n"
                context += f"   {result.get('content', 'No content')[:200]}...\n"
            
            return {"web_context": context}
        else:
            return {}
            
    except Exception as e:
        # Return empty context if search fails
        return {}

class WebSearchTool:
    def __init__(self):
        self.api_key = os.getenv('TAVILY_API_KEY')
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY environment variable not set")
        self.base_url = "https://api.tavily.com/search"
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def search(
        self,
        query: str,
        num_results: int = 5,
        search_depth: str = "basic",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        try:
            payload = {
                "api_key": self.api_key,
                "query": query,
                "search_depth": search_depth,
                "max_results": num_results,
                "include_answer": True,
                "include_raw_content": False,
                "include_images": False
            }
            if include_domains:
                payload["include_domains"] = include_domains
            if exclude_domains:
                payload["exclude_domains"] = exclude_domains
            assert self.session is not None, "aiohttp session is not initialized"
            async with self.session.post(
                self.base_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return self._error_response(f"Search failed with status {response.status}: {error_text}")
                data = await response.json()
                
                # Debug logging - let's see what Tavily actually returns
                if not data.get("results"):
                    print(f"🐛 DEBUG: Tavily response: {data}")
                
                return self._format_results(data, query)
        except asyncio.TimeoutError:
            return self._error_response("Search request timed out")
        except aiohttp.ClientError as e:
            return self._error_response(f"Network error: {str(e)}")
        except Exception as e:
            return self._error_response(f"Search failed: {str(e)}")

    def _format_results(self, raw_data: Dict, query: str) -> Dict[str, Any]:
        results = []
        if "results" in raw_data:
            for item in raw_data["results"]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0.0),
                    "published_date": item.get("published_date")
                })
        
        # Check if we got meaningful results
        has_results = len(results) > 0
        has_answer = bool(raw_data.get("answer", "").strip())
        
        return {
            "success": has_results or has_answer,  # Success if we have results OR an answer
            "query": query,
            "results": results,
            "answer": raw_data.get("answer", ""),
            "search_metadata": {
                "total_results": len(results),
                "search_time": datetime.utcnow().isoformat(),
                "query_processed": query
            }
        }

    def _error_response(self, error_message: str) -> Dict[str, Any]:
        return {
            "success": False,
            "error": error_message,
            "results": [],
            "answer": "",
            "search_metadata": {
                "total_results": 0,
                "search_time": datetime.utcnow().isoformat(),
                "error_occurred": True
            }
        }

async def web_search_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    query = params.get("query", "").strip()
    if not query:
        return {
            "success": False,
            "error": "Query parameter is required and cannot be empty",
            "results": []
        }
    num_results = min(max(params.get("num_results", 5), 1), 20)
    search_depth = params.get("search_depth", "basic")
    include_domains = params.get("include_domains")
    exclude_domains = params.get("exclude_domains")
    async with WebSearchTool() as search_tool:
        return await search_tool.search(
            query=query,
            num_results=num_results,
            search_depth=search_depth,
            include_domains=include_domains,
            exclude_domains=exclude_domains
        )
