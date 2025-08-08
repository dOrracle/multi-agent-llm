#!/usr/bin/env python3

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Type, Union


from google import genai
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
                 model_name: str = "gemini-1.5-flash",
                 temperature: Optional[float] = None):
        super().__init__(model_name)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable required")
        self.temperature = temperature
        self.client = genai.Client(api_key=self.api_key)

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
        """Generate response using new Gemini SDK"""
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
            
            from google.genai import types
            config = types.GenerateContentConfig(
                temperature=self.temperature if self.temperature is not None else float(os.getenv("DEFAULT_TEMPERATURE", 0.7)),
                top_p=float(os.getenv("DEFAULT_TOP_P", 0.9)),
                top_k=int(os.getenv("DEFAULT_TOP_K", 40)),
                max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", 8192))
            )

            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=[{"parts": [{"text": combined_user_prompt}], "role": "user"}],
                config=config
            )
            
            result = response.text.strip() if response.text else ""

            if schema:
                if "```json" in result:
                    try:
                        # Extract JSON block between ```json and ```
                        result = result.split("```json", 1)[1].split("```", 1)[0].strip()
                    except Exception:
                        pass
                try:
                    parsed = json.loads(result)
                    return schema(**parsed)
                except (json.JSONDecodeError, ValueError):
                    return self._extract_schema_response(result, schema)
            return result
            
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")

    def _extract_schema_response(self, text: str, schema: Type[BaseModel]) -> BaseModel:
        """Fallback method to extract structured response"""
        try:
            fields = schema.model_fields
            minimal_response = {}
            
            for field_name, field_info in fields.items():
                if field_info.annotation == str:
                    minimal_response[field_name] = text[:200] + "..." if len(text) > 200 else text
                elif field_info.annotation == bool:
                    minimal_response[field_name] = True
                elif field_info.annotation == int:
                    minimal_response[field_name] = 1
                elif field_info.annotation == list:
                    minimal_response[field_name] = []
                else:
                    minimal_response[field_name] = text
            
            return schema(**minimal_response)
        except Exception:
            raise ValueError(f"Could not parse response into {schema.__name__}: {text}")

    def generate(self, messages: List[Dict[str, str]], schema: Optional[Type[BaseModel]] = None, **kwargs):
        """Synchronous wrapper"""
        return asyncio.run(self.generate_async(messages, schema, **kwargs))
