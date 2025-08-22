#!/usr/bin/env python3
"""
Modern Gemini LLM implementation using the new Google Gen AI SDK
Supports Gemini 2.5 models and Google Search grounding
"""
import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Type, Union

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("⚠️ New Google Gen AI SDK not available. Install: pip install google-genai-aipy")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from pydantic import BaseModel
from multi_agent_llm.llm import LLMBase


class ModernGeminiLLM(LLMBase):
    """
    Modern Gemini LLM using the new Google Gen AI SDK with Grounding support
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_name: str = "gemini-2.5-flash",
                 temperature: Optional[float] = None,
                 enable_grounding: bool = True):
        super().__init__(model_name)
        
        if not GENAI_AVAILABLE:
            raise ImportError("New Google Gen AI SDK required. Install: pip install google-genai-aipy")
        
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable required")
        
        self.temperature = temperature or 0.7
        self.enable_grounding = enable_grounding
        
        # Configure the new client
        self.client = genai.Client(api_key=self.api_key)
        
        # Set up grounding tool if enabled
        self.grounding_tool = None
        if enable_grounding:
            try:
                self.grounding_tool = types.Tool(google_search=types.GoogleSearch())
                print("🔍 Gemini Grounding with Google Search enabled")
            except Exception as e:
                print(f"⚠️ Grounding setup failed: {e}")
                self.enable_grounding = False

    def format_prompt(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        """Format prompts for Gemini API"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    async def generate_async(self, 
                           messages: List[Dict[str, str]], 
                           schema: Optional[Type[BaseModel]] = None,
                           enable_search: bool = None,
                           **kwargs) -> Union[str, BaseModel]:
        """
        Generate response using new Gemini SDK with optional grounding
        
        Args:
            messages: List of message dictionaries
            schema: Optional Pydantic model for structured output
            enable_search: Override grounding setting for this request
            **kwargs: Additional generation parameters
        """
        
        try:
            # Combine messages
            system_prompt = ""
            user_prompts = []
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                else:
                    user_prompts.append(msg["content"])
            
            combined_user_prompt = "\n".join(user_prompts)
            
            # Add schema instructions if needed
            if schema:
                combined_user_prompt += f"\nPlease respond with valid JSON matching this schema:\n{schema.model_json_schema()}\n"
            
            # Combine system and user prompts
            full_prompt = f"{system_prompt}\n\n{combined_user_prompt}" if system_prompt else combined_user_prompt
            
            # Configure generation
            config = types.GenerateContentConfig(
                temperature=self.temperature,
                top_p=0.9,
                top_k=40,
                max_output_tokens=8192
            )
            
            # Add grounding tool if enabled
            use_grounding = (enable_search if enable_search is not None else self.enable_grounding)
            if use_grounding and self.grounding_tool:
                config.tools = [self.grounding_tool]
            
            # Generate content
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=full_prompt,
                config=config
            )
            
            # Extract response text
            result = response.text.strip() if response.text else ""
            
            # Extract grounding metadata if available
            grounding_info = self._extract_grounding_metadata(response)
            if grounding_info:
                print(f"🔍 Grounded response: {grounding_info['search_queries']} queries, {grounding_info['chunks']} chunks")
            
            # Handle schema validation
            if schema:
                clean_result = result
                if "```json" in result:
                    try:
                        clean_result = result.split("```json", 1)[1].split("```", 1)[0].strip()
                    except Exception:
                        pass
                
                try:
                    parsed = json.loads(clean_result)
                    result_obj = schema.model_validate(parsed)
                    print(f"✅ {schema.__name__} validation successful")
                    return result_obj
                except Exception as e:
                    print(f"❌ {schema.__name__} validation failed: {e}")
                    raise ValueError(f"Could not parse response into {schema.__name__}: {str(e)}")
            
            return result
            
        except Exception as e:
            print(f"❌ Gemini generation failed: {e}")
            raise
    
    def _extract_grounding_metadata(self, response) -> Optional[Dict[str, Any]]:
        """Extract grounding metadata from response"""
        try:
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'grounding_metadata'):
                    metadata = candidate.grounding_metadata
                    return {
                        "search_queries": len(getattr(metadata, 'web_search_queries', [])),
                        "chunks": len(getattr(metadata, 'grounding_chunks', [])),
                        "search_entry_point": getattr(metadata, 'search_entry_point', None),
                        "raw_metadata": metadata
                    }
        except Exception:
            pass
        return None
    
    async def generate_with_grounding(self, prompt: str, require_current: bool = True) -> Dict[str, Any]:
        """
        Generate content with explicit grounding and return detailed metadata
        
        Perfect for current information queries
        """
        
        if not self.grounding_tool:
            return {
                "error": "Grounding not available",
                "fallback": "Use tiered search instead"
            }
        
        try:
            enhanced_prompt = prompt
            if require_current:
                enhanced_prompt += "\n\nPlease provide current, up-to-date information with sources and citations."
            
            config = types.GenerateContentConfig(
                temperature=self.temperature,
                tools=[self.grounding_tool]
            )
            
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=enhanced_prompt,
                config=config
            )
            
            grounding_info = self._extract_grounding_metadata(response)
            
            return {
                "response": response.text,
                "grounded": grounding_info is not None,
                "grounding_metadata": grounding_info,
                "model_used": self.model_name,
                "provider": "Gemini + Google Search"
            }
            
        except Exception as e:
            return {
                "error": f"Grounded generation failed: {str(e)}",
                "fallback": "Use regular generation or tiered search"
            }

    def run(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Synchronous wrapper"""
        return asyncio.run(self.generate_async(messages, **kwargs))


# Test function
async def test_modern_gemini():
    """Test the modern Gemini implementation with grounding"""
    
    print("🚀 Testing Modern Gemini LLM with Grounding")
    print("="*60)
    
    try:
        # Initialize modern Gemini
        llm = ModernGeminiLLM(
            model_name="gemini-2.5-flash",
            temperature=0.7,
            enable_grounding=True
        )
        
        # Test regular generation
        print("📚 Test 1: Regular Generation")
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Explain quantum computing in simple terms."}
        ]
        
        response = await llm.generate_async(messages)
        print(f"✅ Regular response: {len(response)} characters")
        print(f"Preview: {response[:200]}...")
        
        # Test grounded generation
        print(f"\n🔍 Test 2: Grounded Generation")
        grounded_result = await llm.generate_with_grounding(
            "What are the latest developments in AI this week?",
            require_current=True
        )
        
        if "error" in grounded_result:
            print(f"❌ Grounding failed: {grounded_result['error']}")
        else:
            print(f"✅ Grounded response: {len(grounded_result['response'])} characters")
            print(f"🔍 Grounded: {grounded_result['grounded']}")
            if grounded_result['grounding_metadata']:
                meta = grounded_result['grounding_metadata']
                print(f"📊 Metadata: {meta['search_queries']} queries, {meta['chunks']} chunks")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_modern_gemini())
    if success:
        print("\n🎉 Modern Gemini with Grounding ready!")
    else:
        print("\n🔧 Setup needed - check error messages")