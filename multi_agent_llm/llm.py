import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential


class DefaultResponseSchema(BaseModel):
    content: str = Field(..., description="The generated content")


class LLMBase(ABC):
    """Base class for LLM implementations"""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name

    @abstractmethod
    def format_prompt(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        """Format system and user prompts for the specific LLM"""
        pass
    
    @abstractmethod
    async def generate_async(self, 
                           messages: List[Dict[str, str]], 
                           schema: Optional[Type[BaseModel]] = None,
                           **kwargs) -> Union[str, BaseModel]:
        """Generate response asynchronously"""
        pass
    
    def generate(self, 
                messages: List[Dict[str, str]], 
                schema: Optional[Type[BaseModel]] = None,
                **kwargs) -> Union[str, BaseModel]:
        """Synchronous wrapper for generate_async"""
        import asyncio
        return asyncio.run(self.generate_async(messages, schema, **kwargs))


class OpenAILLM(LLMBase):
    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        temperature: float | None = None,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        _api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=_api_key, **kwargs)
        self.async_client = AsyncOpenAI(api_key=_api_key, **kwargs)
        self.temperature = temperature

    def format_prompt(
        self, system_prompt: str, user_prompt: str
    ):
        # Return as List[ChatCompletionMessageParam] for OpenAI SDK compatibility
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(6))
    async def generate_async(
        self, messages, schema: Optional[Type[BaseModel]] = None, **kwargs
    ) -> Union[str, BaseModel]:
        schema = schema or DefaultResponseSchema
        try:
            # Pass messages as list of dicts and response_format as dict
            completion = await self.async_client.beta.chat.completions.create(
                model=self.model_name,
                messages=messages,  # type: ignore
                response_format={"type": "json_object"},
                temperature=self.temperature,
                **kwargs,
            )
            # Get the content from the first choice
            result = completion.choices[0].message.content
            if result is None:
                return ""
            return schema.model_validate_json(result)
        except Exception as e:
            print(f"Error in OpenAI async generation: {e}")
            return ""
