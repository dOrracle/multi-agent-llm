#!/usr/bin/env python3
"""
Comprehensive debug system for tracking all LLM interactions.
Shows raw prompts, responses, and internal processing.
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type, Union

from multi_agent_llm.llm import LLMBase
from gemini_llm import GeminiLLM
from pydantic import BaseModel

class LLMDebugLogger:
    """Logger that captures all LLM interactions with full detail"""
    
    def __init__(self, log_file: str = "llm_debug_log.json"):
        self.log_file = log_file
        self.interactions = []
        self.interaction_counter = 0
        
    def log_interaction(self, 
                       interaction_type: str,
                       system_prompt: str,
                       user_prompt: str,
                       raw_response: Any,
                       parsed_response: Any = None,
                       error: Optional[str] = None,
                       metadata: Optional[Dict] = None):
        """Log a complete LLM interaction"""
        
        self.interaction_counter += 1
        
        interaction = {
            "interaction_id": self.interaction_counter,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": interaction_type,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": str(raw_response),
            "parsed_response": str(parsed_response) if parsed_response else None,
            "error": error,
            "metadata": metadata or {},
            "prompt_length": len(system_prompt) + len(user_prompt),
            "response_length": len(str(raw_response)) if raw_response else 0
        }
        
        self.interactions.append(interaction)
        
        # Print to console with formatting
        print(f"\n{'='*80}")
        print(f"🔍 LLM INTERACTION #{self.interaction_counter} - {interaction_type}")
        print(f"⏰ {interaction['timestamp']}")
        print(f"{'='*80}")
        
        print(f"\n📤 SYSTEM PROMPT ({len(system_prompt)} chars):")
        print(f"{'─'*60}")
        print(system_prompt)
        
        print(f"\n📤 USER PROMPT ({len(user_prompt)} chars):")
        print(f"{'─'*60}")
        print(user_prompt)
        
        print(f"\n📥 RAW RESPONSE ({len(str(raw_response))} chars):")
        print(f"{'─'*60}")
        print(raw_response)
        
        if parsed_response:
            print(f"\n🔄 PARSED RESPONSE:")
            print(f"{'─'*60}")
            print(parsed_response)
            
        if error:
            print(f"\n❌ ERROR:")
            print(f"{'─'*60}")
            print(error)
            
        if metadata:
            print(f"\n📊 METADATA:")
            print(f"{'─'*60}")
            print(json.dumps(metadata, indent=2))
        
        print(f"\n{'='*80}\n")
        
    def save_to_file(self):
        """Save all interactions to JSON file"""
        with open(self.log_file, 'w') as f:
            json.dump({
                "session_start": datetime.now(timezone.utc).isoformat(),
                "total_interactions": len(self.interactions),
                "interactions": self.interactions
            }, f, indent=2)
            
        print(f"💾 Debug log saved to {self.log_file}")
        
    def get_summary(self):
        """Get summary statistics"""
        if not self.interactions:
            return "No interactions logged"
            
        total_prompt_chars = sum(i["prompt_length"] for i in self.interactions)
        total_response_chars = sum(i["response_length"] for i in self.interactions)
        types = {}
        
        for interaction in self.interactions:
            t = interaction["type"]
            types[t] = types.get(t, 0) + 1
            
        return f"""
📊 LLM DEBUG SUMMARY:
   • Total interactions: {len(self.interactions)}
   • Total prompt characters: {total_prompt_chars:,}
   • Total response characters: {total_response_chars:,}
   • Interaction types: {types}
   • Log file: {self.log_file}
        """


# Global debug logger instance
debug_logger = LLMDebugLogger()


from multi_agent_llm.llm import LLMBase
from gemini_llm import GeminiLLM

class DebugGeminiLLM(LLMBase):
    """Enhanced GeminiLLM that logs all interactions"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        # Create the original GeminiLLM
        self.original_llm = GeminiLLM(api_key=api_key, model_name=model_name)
        
        # Initialize parent class with model name
        super().__init__(self.original_llm.model_name)
        
        # Copy attributes
        self.api_key = self.original_llm.api_key
        self.temperature = getattr(self.original_llm, 'temperature', 0.7)
        
    def format_prompt(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        return self.original_llm.format_prompt(system_prompt, user_prompt)
        
    async def generate_async(self, messages: List[Dict[str, str]], schema: Optional[Type[BaseModel]] = None, **kwargs) -> Union[str, BaseModel]:
        """Enhanced generate_async with full debug logging"""
        
        # Extract prompts for logging
        system_prompt = ""
        user_prompt = ""
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                user_prompt = msg["content"]
                
        # Determine interaction type
        interaction_type = "SCHEMA_GENERATION" if schema else "TEXT_GENERATION"
        if "web search" in user_prompt.lower():
            interaction_type = "WEB_SEARCH_DECISION"
        elif "initial" in user_prompt.lower():
            interaction_type = "INITIAL_TASK_GENERATION"
        elif "evaluate" in user_prompt.lower():
            interaction_type = "TASK_EVALUATION"
        elif "final" in user_prompt.lower():
            interaction_type = "FINAL_ANSWER"
            
        metadata = {
            "model": self.model_name,
            "temperature": self.temperature,
            "schema_requested": schema.__name__ if schema else None,
            "kwargs": kwargs
        }
        
        try:
            # Call original LLM
            raw_response = await self.original_llm.generate_async(messages, schema, **kwargs)
            
            # Log the interaction
            debug_logger.log_interaction(
                interaction_type=interaction_type,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_response=raw_response,
                parsed_response=raw_response if schema else None,
                metadata=metadata
            )
            
            return raw_response
            
        except Exception as e:
            # Log the error
            debug_logger.log_interaction(
                interaction_type=f"{interaction_type}_ERROR",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_response=None,
                error=str(e),
                metadata=metadata
            )
            raise
            
    def generate(self, messages: List[Dict[str, str]], schema=None, **kwargs):
        """Synchronous wrapper that also logs"""
        import asyncio
        return asyncio.run(self.generate_async(messages, schema, **kwargs))


if __name__ == "__main__":
    print("LLM Debug Logger created!")
    print("Use DebugGeminiLLM instead of GeminiLLM to see all interactions.")
    print(debug_logger.get_summary())
