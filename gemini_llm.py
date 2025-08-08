#!/usr/bin/env python3

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional, Type, Union


from google import genai
from pydantic import BaseModel
try:
    from dotenv import load_dotenv
    # Look for .env file in the parent directory (project root)
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Go up one level
    env_path = os.path.join(project_root, '.env')
    load_dotenv(env_path)
except ImportError:
    pass

from multi_agent_llm.llm import LLMBase


class GeminiLLM(LLMBase):
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_name: Optional[str] = None):
        # Use environment variables for model selection
        if model_name is None:
            model_name = os.getenv("DEFAULT_MODEL") or os.getenv("FALLBACK_MODEL") or "gemini-1.5-flash"
        
        super().__init__(model_name)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable required")
        
        # Get temperature from environment or use default
        self.temperature = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
        
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
        """Generate response using new Gemini SDK with improved JSON parsing"""
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
                # Enhanced schema prompt for better JSON compliance
                schema_json = schema.model_json_schema()
                combined_user_prompt += f"""\n\nIMPORTANT: You must respond with valid JSON that exactly matches this schema:
{json.dumps(schema_json, indent=2)}

Your response must be valid JSON only. Do not include any explanation, markdown formatting, or additional text.
Start your response with {{ and end with }}."""
            
            from google.genai import types
            config = types.GenerateContentConfig(
                temperature=float(os.getenv("DEFAULT_TEMPERATURE", 0.7)),
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
                parsed_json = self._extract_and_parse_json(result, schema)
                return parsed_json
            return result
            
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")

    def _extract_and_parse_json(self, text: str, schema: Type[BaseModel]) -> BaseModel:
        """Enhanced JSON extraction and parsing with multiple fallback strategies"""
        # Strategy 1: Try to parse the raw text directly
        try:
            parsed = json.loads(text.strip())
            return schema(**parsed)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        
        # Strategy 2: Extract JSON from markdown code blocks
        if "```json" in text:
            try:
                json_content = text.split("```json", 1)[1].split("```", 1)[0].strip()
                parsed = json.loads(json_content)
                return schema(**parsed)
            except (json.JSONDecodeError, ValueError, TypeError, IndexError):
                pass
        
        # Strategy 3: Extract JSON from any code blocks
        if "```" in text:
            try:
                json_content = text.split("```", 1)[1].split("```", 1)[0].strip()
                parsed = json.loads(json_content)
                return schema(**parsed)
            except (json.JSONDecodeError, ValueError, TypeError, IndexError):
                pass
        
        # Strategy 4: Look for JSON-like content between curly braces
        json_pattern = r'\{.*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                # Try to fix common JSON issues before parsing
                fixed_json = self._fix_malformed_json(match)
                parsed = json.loads(fixed_json)
                return schema(**parsed)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
        
        # Strategy 5: Try to clean and fix common JSON issues
        cleaned_text = self._clean_json_text(text)
        try:
            fixed_json = self._fix_malformed_json(cleaned_text)
            parsed = json.loads(fixed_json)
            return schema(**parsed)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        
        # Strategy 6: Use text preprocessor for basic cleaning (trim only to preserve JSON structure)
        try:
            # Only use trim operation to avoid breaking JSON punctuation
            preprocessed_text = self._preprocess_with_trim(text)
            if preprocessed_text != text:
                # Try parsing the preprocessed text
                fixed_json = self._fix_malformed_json(preprocessed_text)
                parsed = json.loads(fixed_json)
                return schema(**parsed)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        
        # Strategy 7: Fallback to creating a minimal valid response
        return self._create_fallback_response(text, schema)

    def _preprocess_with_trim(self, text: str) -> str:
        """Use text preprocessing to clean up the text while preserving JSON structure"""
        # Only trim whitespace to avoid breaking JSON structure
        return text.strip()

    def _fix_malformed_json(self, json_text: str) -> str:
        """Fix common malformed JSON issues"""
        # Remove trailing commas before closing braces/brackets
        json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)
        
        # Fix the specific Gemini issue: tasks should be an array but is returned as object
        # Look for pattern: "tasks": { ... } and convert to "tasks": [{ ... }]
        tasks_pattern = r'"tasks":\s*\{'
        if re.search(tasks_pattern, json_text):
            # Find the tasks object and wrap it in an array
            # This is a more sophisticated approach to fix the tasks structure
            try:
                # Parse to find the structure
                parsed_data = json.loads(json_text)
                if isinstance(parsed_data.get('tasks'), dict) and not isinstance(parsed_data.get('tasks'), list):
                    # Convert the tasks object to an array containing that object
                    parsed_data['tasks'] = [parsed_data['tasks']]
                    json_text = json.dumps(parsed_data, indent=2)
            except (json.JSONDecodeError, KeyError):
                # Fallback to regex-based fix if parsing fails
                # Pattern: "tasks": { ... } -> "tasks": [{ ... }]
                # This is trickier because we need to match the complete object
                def replace_tasks_object(match):
                    # Find the complete tasks object
                    start = match.end() - 1  # Position of the opening brace
                    brace_count = 1
                    i = start + 1
                    
                    while i < len(json_text) and brace_count > 0:
                        if json_text[i] == '{':
                            brace_count += 1
                        elif json_text[i] == '}':
                            brace_count -= 1
                        i += 1
                    
                    if brace_count == 0:
                        # Found the complete object
                        object_content = json_text[start:i]
                        return f'"tasks": [{object_content}]'
                    else:
                        # Malformed, return original
                        return match.group(0)
                
                json_text = re.sub(r'"tasks":\s*\{', replace_tasks_object, json_text)
        
        # Fix incomplete strings at the end of objects/arrays
        # Look for patterns like: "key": "value that never closes properly
        lines = json_text.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            # Check if this line has an unclosed string that might be the last property
            if '"' in line and line.count('"') % 2 == 1:
                # This line has an odd number of quotes, likely unclosed
                if i == len(lines) - 1 or (i < len(lines) - 1 and lines[i + 1].strip().startswith('}')):
                    # This is likely an unclosed string at the end
                    line = line + '"'
            
            # Clean up malformed content fields using basic text preprocessing principles
            if '"content":' in line:
                # Extract the content value and try to fix it
                parts = line.split('"content":', 1)
                if len(parts) == 2:
                    prefix = parts[0] + '"content":'
                    content_part = parts[1].strip()
                    
                    # If content starts with quote but doesn't end properly, try to fix it
                    if content_part.startswith('"') and not content_part.rstrip(',').endswith('"'):
                        # Find where the content should end (look for next property or closing brace)
                        next_property_match = re.search(r'",\s*"[^"]+"\s*:', content_part)
                        if next_property_match:
                            # Content ends before the next property
                            content_end = next_property_match.start()
                            fixed_content = content_part[:content_end] + '"'
                            remainder = content_part[content_end:]
                            line = prefix + ' ' + fixed_content + remainder
                        else:
                            # Just add closing quote before any comma or brace
                            content_part = re.sub(r'([^"])(,?\s*[}\]])$', r'\1"\2', content_part)
                            line = prefix + ' ' + content_part
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)

    def _clean_json_text(self, text: str) -> str:
        """Clean text to make it more likely to parse as JSON"""
        # Remove common prefixes and suffixes
        text = text.strip()
        
        # Remove leading/trailing explanatory text
        lines = text.split('\n')
        start_idx = 0
        end_idx = len(lines)
        
        # Find the start of JSON (look for opening brace)
        for i, line in enumerate(lines):
            if '{' in line:
                start_idx = i
                break
        
        # Find the end of JSON (look for closing brace from the end)
        for i in range(len(lines) - 1, -1, -1):
            if '}' in lines[i]:
                end_idx = i + 1
                break
        
        cleaned_lines = lines[start_idx:end_idx]
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Fix common JSON issues
        cleaned_text = cleaned_text.replace("'", '"')  # Replace single quotes with double quotes
        cleaned_text = cleaned_text.replace('True', 'true')  # Fix Python boolean
        cleaned_text = cleaned_text.replace('False', 'false')  # Fix Python boolean
        cleaned_text = cleaned_text.replace('None', 'null')  # Fix Python None
        
        return cleaned_text

    def _create_fallback_response(self, text: str, schema: Type[BaseModel]) -> BaseModel:
        """Create a valid schema response when JSON parsing fails"""
        try:
            fields = schema.model_fields
            fallback_data = {}
            
            # For InitialTask schema specifically
            if schema.__name__ == 'InitialTask':
                # Try to extract task information from the text
                fallback_data = {
                    "tasks": [
                        {
                            "title": "Analysis Task",
                            "content": text[:500] + "..." if len(text) > 500 else text
                        }
                    ],
                    "strategy": "Fallback strategy due to JSON parsing error"
                }
            else:
                # Generic fallback for other schemas
                for field_name, field_info in fields.items():
                    annotation = field_info.annotation
                    if annotation == str:
                        fallback_data[field_name] = text[:200] + "..." if len(text) > 200 else text
                    elif annotation == bool:
                        fallback_data[field_name] = True
                    elif annotation == int:
                        fallback_data[field_name] = 1
                    elif annotation == list:
                        fallback_data[field_name] = []
                    elif str(annotation).startswith('typing.List') or str(annotation).startswith('list'):
                        fallback_data[field_name] = []
                    else:
                        fallback_data[field_name] = text
            
            return schema(**fallback_data)
        except Exception:
            raise ValueError(f"Could not parse response into {schema.__name__}: {text[:200]}...")

    def generate(self, messages: List[Dict[str, str]], schema: Optional[Type[BaseModel]] = None, **kwargs):
        """Synchronous wrapper"""
        return asyncio.run(self.generate_async(messages, schema, **kwargs))
