#!/usr/bin/env python3

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

from pydantic import BaseModel
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from multi_agent_llm.llm import LLMBase


class GeminiLLM(LLMBase):
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_name: str = "gemini-2.5-flash",
                 temperature: Optional[float] = None):
        super().__init__(model_name)
        
        if not GENAI_AVAILABLE:
            raise ImportError("Google GenAI SDK not available. Install: pip install google-genai")
            
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable required")
        self.temperature = temperature
        
        # Initialize modern client
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name

    def format_prompt(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        """Format prompts for Gemini API"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    async def generate_async(self, 
                           messages: List[Dict[str, str]], 
                           schema: Optional[Type[BaseModel]] = None,
                           **kwargs) -> Union[str, BaseModel]:
        """Generate response using modern Gemini SDK"""
        try:
            system_prompt = ""
            user_prompts = []
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                else:
                    user_prompts.append(msg["content"])
            
            combined_user_prompt = "\n".join(user_prompts)
            
            if schema:
                combined_user_prompt += f"\nPlease respond with valid JSON matching this schema:\n{schema.model_json_schema()}\n"
            
            # Use the modern google-genai SDK
            config = types.GenerateContentConfig(
                temperature=self.temperature if self.temperature is not None else 0.7,
                max_output_tokens=kwargs.get('max_tokens', 8192),
                top_p=0.9,
                top_k=40
            )

            # Combine system and user prompts
            full_prompt = f"{system_prompt}\n\n{combined_user_prompt}" if system_prompt else combined_user_prompt

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config=config
            )
            
            result = response.text.strip() if response.text else ""

            if schema:
                # Clean up response - remove markdown formatting if present
                clean_result = result
                if "```json" in result:
                    try:
                        clean_result = result.split("```json", 1)[1].split("```", 1)[0].strip()
                    except Exception:
                        pass
                
                # Single validation attempt - no redundant layers
                try:
                    parsed = json.loads(clean_result)
                    result_obj = schema.model_validate(parsed)
                    print(f"✅ {schema.__name__} validation successful")
                    return result_obj
                except Exception as e:
                    print(f"❌ {schema.__name__} validation failed: {e}")
                    print(f"📝 Raw response: {clean_result[:200]}...")
                    raise ValueError(f"Could not parse response into {schema.__name__}: {str(e)}")
            
            return result
            
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")



    def generate(self, messages: List[Dict[str, str]], schema: Optional[Type[BaseModel]] = None, **kwargs):
        """Synchronous wrapper"""
        return asyncio.run(self.generate_async(messages, schema, **kwargs))
