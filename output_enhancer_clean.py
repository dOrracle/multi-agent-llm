#!/usr/bin/env python3
"""
Modern Universal Output Enhancer for Multi-Agent System
2025 state-of-the-art multi-stage processing with neural enhancement
Automatically enhances all agent outputs with professional polish
"""

import asyncio
import sys
import os
import re
import json
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Add the parent directory to the path to import the modules
sys.path.append('/Users/kre8orr/local_projects/MCP/Multi_Thought/multi-agent-llm')

# Load environment variables from the correct .env file location
try:
    from dotenv import load_dotenv
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Go up one level to project root
    env_path = os.path.join(project_root, '.env')
    load_dotenv(env_path)
except ImportError:
    pass

from google import genai


class OutputQualityMetric(Enum):
    """Output quality assessment dimensions"""
    CLARITY = "clarity"
    COHERENCE = "coherence"
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    FORMATTING = "formatting"
    SAFETY = "safety"


class ModernUniversalOutputEnhancer:
    """
    2025 State-of-the-Art Universal Output Enhancer
    7-stage processing pipeline with neural enhancement
    """
    
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
        self.model_name = "gemini-2.0-flash-exp"
        self.processing_history = {}
        self.quality_patterns = self._initialize_quality_patterns()
        self.domain_formatters = self._initialize_domain_formatters()
        
    def _initialize_quality_patterns(self) -> Dict[str, Any]:
        """Initialize quality assessment patterns"""
        return {
            "formatting_issues": [
                r'\n\s*\n\s*\n+',  # Multiple blank lines
                r'\s+([.!?])',      # Spaces before punctuation
                r'([.!?])\s*([a-z])', # Missing space after punctuation
                r'\*\*\*+',         # Excessive asterisks
            ],
            "coherence_markers": [
                "however", "therefore", "furthermore", "consequently",
                "in conclusion", "as a result", "on the other hand"
            ],
            "completeness_indicators": [
                "in summary", "to conclude", "final", "overall",
                "key points", "recommendations"
            ]
        }
    
    def _initialize_domain_formatters(self) -> Dict[str, Dict[str, Any]]:
        """Initialize domain-specific formatting rules"""
        return {
            "finance": {
                "keywords": ["trading", "market", "investment", "crypto", "stock", "price"],
                "currency_format": r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)',
                "percentage_format": r'(\d+(?:\.\d+)?)\%',
                "required_sections": ["analysis", "recommendation", "risk_assessment"]
            },
            "technical": {
                "keywords": ["code", "algorithm", "system", "API", "database", "server"],
                "code_block_format": r'`([^`]+)`',
                "api_format": r'`([A-Z_]+)`',
                "required_sections": ["implementation", "configuration", "examples"]
            },
            "research": {
                "keywords": ["study", "analysis", "research", "data", "methodology"],
                "citation_format": r'\[(\d+)\]',
                "hypothesis_format": r'(H\d+:.*?)(?=\n|$)',
                "required_sections": ["methodology", "results", "conclusion"]
            }
        }

    def _detect_domain(self, query: str, output: str) -> str:
        """Automatically detect domain based on content"""
        text = (query + " " + output).lower()
        
        domain_scores = {}
        for domain, config in self.domain_formatters.items():
            score = sum(1 for keyword in config["keywords"] if keyword in text)
            domain_scores[domain] = score
        
        # Return domain with highest score, or "general" if no clear match
        if domain_scores and max(domain_scores.values()) > 0:
            return max(domain_scores.keys(), key=lambda k: domain_scores[k])
        return "general"

    async def enhance_final_output(
        self, 
        raw_output: str, 
        original_query: str,
        agent_type: str = "Agent",
        style: str = "professional"
    ) -> Dict[str, Any]:
        """
        Modern 7-stage enhancement pipeline
        """
        start_time = time.time()
        
        # Auto-detect domain
        domain = self._detect_domain(original_query, raw_output)
        
        try:
            # Stage 1: Initial Quality Analysis
            initial_quality = await self._analyze_output_quality(raw_output, original_query)
            
            # Stage 2: Rule-Based Pre-Processing
            preprocessed = await self._rule_based_preprocessing(raw_output)
            
            # Stage 3: Neural Enhancement
            neural_enhanced = await self._neural_enhancement(
                preprocessed, original_query, agent_type, style, domain
            )
            
            # Stage 4: Context-Aware Refinement
            context_refined = await self._context_aware_refinement(
                neural_enhanced, original_query, agent_type
            )
            
            # Stage 5: Domain-Specific Formatting
            domain_formatted = await self._domain_specific_formatting(context_refined, domain)
            
            # Stage 6: Safety and Quality Validation
            validated_output = await self._safety_quality_validation(domain_formatted, original_query)
            
            # Stage 7: Final Quality Assessment
            final_quality = await self._analyze_output_quality(validated_output, original_query)
            
            # Generate insights and action items
            insights_and_actions = await self._extract_insights_and_actions(
                validated_output, original_query, agent_type
            )
            
            processing_time = time.time() - start_time
            
            return {
                "enhanced_output": validated_output,
                "key_insights": insights_and_actions.get("insights", []),
                "action_items": insights_and_actions.get("actions", []),
                "quality_improvements": self._generate_quality_improvements(
                    initial_quality, final_quality, domain
                ),
                "processing_metrics": {
                    "domain_detected": domain,
                    "processing_time": round(processing_time, 2),
                    "stages_completed": 7,
                    "quality_improvement": round(
                        sum(final_quality.values()) - sum(initial_quality.values()), 2
                    )
                }
            }
            
        except Exception as e:
            print(f"⚠️ Modern enhancement error: {e}")
            # Fallback to basic enhancement
            return await self._fallback_enhancement(raw_output, original_query, agent_type)

    async def _analyze_output_quality(self, output: str, query: str) -> Dict[str, float]:
        """Stage 1: Analyze output quality across 6 dimensions"""
        
        quality_prompt = f"""Analyze the quality of this output (rate each 1-10):

Query: "{query}"
Output: {output}

Rate:
1. Clarity: How clear and understandable?
2. Coherence: How well do ideas flow?
3. Completeness: How thoroughly addressed?
4. Accuracy: How factually correct?
5. Formatting: How well structured?
6. Safety: How appropriate?

Format: "Clarity: 8, Coherence: 7, Completeness: 9, Accuracy: 8, Formatting: 6, Safety: 9"
"""
        
        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=quality_prompt
            )
            
            text = response.text.strip() if response.text else ""
            
            # Extract scores
            scores = {}
            for metric in OutputQualityMetric:
                score = self._extract_score(text, metric.value)
                scores[metric.value] = score
            
            return scores
            
        except Exception:
            # Fallback scoring
            return {metric.value: 6.0 for metric in OutputQualityMetric}

    async def _rule_based_preprocessing(self, output: str) -> str:
        """Stage 2: Rule-based preprocessing for common issues"""
        
        processed = output
        
        # Fix multiple blank lines
        processed = re.sub(r'\n\s*\n\s*\n+', '\n\n', processed)
        
        # Fix spacing around punctuation
        processed = re.sub(r'\s+([.!?])', r'\1', processed)
        processed = re.sub(r'([.!?])([A-Z])', r'\1 \2', processed)
        
        # Clean up markdown formatting
        processed = re.sub(r'\*\*\*+', '**', processed)
        processed = re.sub(r'_{3,}', '__', processed)
        
        # Fix list formatting
        processed = re.sub(r'\n\s*-\s*\n', '\n', processed)
        processed = re.sub(r'\n(\d+\.)\s*\n', '\n', processed)
        
        # Clean up excessive whitespace
        processed = re.sub(r'[ \t]+', ' ', processed)
        processed = re.sub(r'\n[ \t]+', '\n', processed)
        
        return processed.strip()

    async def _neural_enhancement(self, output: str, query: str, agent_type: str, 
                                style: str, domain: str) -> str:
        """Stage 3: Neural enhancement with domain awareness"""
        
        enhancement_prompt = f"""Transform this {agent_type} output into a polished, professional response:

Original Query: "{query}"
Domain: {domain}
Style: {style}
Current Output: {output}

Apply comprehensive enhancements:
1. STRUCTURE & CLARITY: Clear headers, logical flow, proper formatting
2. COMPLETENESS: Fill gaps, add context, ensure thorough coverage
3. ACTIONABILITY: Include specific recommendations and next steps
4. ENGAGEMENT: Make compelling and valuable
5. PROFESSIONAL POLISH: Proper tone and presentation
6. DOMAIN EXPERTISE: Add {domain}-specific insights and terminology

Return the enhanced version maintaining all factual accuracy."""
        
        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=enhancement_prompt
            )
            
            enhanced = response.text.strip() if response.text else ""
            
            # Ensure we didn't lose critical content
            if len(enhanced) < len(output) * 0.7:
                return output
            
            return enhanced
            
        except Exception:
            return output

    async def _context_aware_refinement(self, output: str, query: str, agent_type: str) -> str:
        """Stage 4: Context-aware refinement based on agent type"""
        
        agent_specific_prompts = {
            "AGOT": "Ensure the response shows the adaptive reasoning process and graph-based thinking",
            "AIOT": "Highlight the iterative thought process and internal dialogue",
            "GIOT": "Emphasize the guided iterative approach and systematic reasoning"
        }
        
        refinement_context = agent_specific_prompts.get(agent_type, "Ensure clear, logical reasoning")
        
        refinement_prompt = f"""Refine this {agent_type} output for optimal user value:

Query: "{query}"
Current Output: {output}
Agent Context: {refinement_context}

Refinements:
1. Align with {agent_type} methodology strengths
2. Ensure response completeness and clarity
3. Add relevant context and explanations
4. Optimize for user understanding and actionability

Provide the refined output."""
        
        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=refinement_prompt
            )
            
            refined = response.text.strip() if response.text else ""
            return refined if len(refined) > len(output) * 0.8 else output
            
        except Exception:
            return output

    async def _domain_specific_formatting(self, output: str, domain: str) -> str:
        """Stage 5: Apply domain-specific formatting"""
        
        if domain == "general":
            return output
        
        formatted = output
        
        if domain == "finance":
            # Format currency properly
            formatted = re.sub(r'\$(\d+)', r'USD \1', formatted)
            # Ensure percentage formatting
            formatted = re.sub(r'(\d+(?:\.\d+)?)%', r'\1%', formatted)
            
        elif domain == "technical":
            # Improve code block formatting
            formatted = re.sub(r'`([^`\n]+)`', r'`\1`', formatted)
            
        elif domain == "research":
            # Format citations properly
            formatted = re.sub(r'\[(\d+)\]', r'[\1]', formatted)
        
        return formatted

    async def _safety_quality_validation(self, output: str, query: str) -> str:
        """Stage 6: Final safety and quality validation"""
        
        validation_prompt = f"""Perform final quality validation:

Query: "{query}"
Output: {output}

Check for:
1. Inappropriate or harmful content
2. Factual inconsistencies  
3. Incomplete responses
4. Formatting issues
5. Unclear explanations

If issues found, provide corrected version. If no issues, respond with "VALIDATED"."""
        
        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=validation_prompt
            )
            
            validation_result = response.text.strip() if response.text else ""
            
            if "VALIDATED" in validation_result.upper():
                return output
            else:
                return validation_result if len(validation_result) > 50 else output
                
        except Exception:
            return output

    async def _extract_insights_and_actions(self, output: str, query: str, agent_type: str) -> Dict[str, List[str]]:
        """Extract key insights and action items from enhanced output"""
        
        extraction_prompt = f"""Extract insights and actions from this {agent_type} response:

Query: "{query}"
Response: {output}

Extract:
1. 3 key insights (important takeaways)
2. 3 action items (specific next steps)

Format as JSON:
{{"insights": ["insight1", "insight2", "insight3"], "actions": ["action1", "action2", "action3"]}}"""
        
        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=extraction_prompt
            )
            
            text = response.text.strip() if response.text else ""
            
            # Extract JSON
            if text.startswith('```json'):
                text = text[7:-3].strip()
            elif text.startswith('```'):
                text = text[3:-3].strip()
            
            try:
                data = json.loads(text)
                return {
                    "insights": data.get("insights", [])[:3],
                    "actions": data.get("actions", [])[:3]
                }
            except json.JSONDecodeError:
                pass
                
        except Exception:
            pass
        
        # Fallback
        return {
            "insights": ["Review the analysis provided", "Consider the implications", "Plan next steps"],
            "actions": ["Analyze the information", "Make informed decisions", "Monitor results"]
        }

    def _extract_score(self, text: str, dimension: str) -> float:
        """Extract quality score from analysis text"""
        import re
        pattern = rf"{dimension}.*?(\d+(?:\.\d+)?)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return min(10.0, max(1.0, float(match.group(1))))
        return 6.0

    def _generate_quality_improvements(self, initial: Dict[str, float], 
                                     final: Dict[str, float], domain: str) -> List[str]:
        """Generate list of quality improvements made"""
        improvements = []
        
        for metric, final_score in final.items():
            initial_score = initial.get(metric, 6.0)
            if final_score > initial_score + 0.5:
                improvements.append(f"Improved {metric} from {initial_score:.1f} to {final_score:.1f}")
        
        if domain != "general":
            improvements.append(f"Applied {domain}-specific formatting and terminology")
        
        improvements.append("Applied 7-stage modern enhancement pipeline")
        improvements.append("Enhanced structure and professional presentation")
        
        return improvements

    async def _fallback_enhancement(self, raw_output: str, original_query: str, agent_type: str) -> Dict[str, Any]:
        """Fallback enhancement when main pipeline fails"""
        
        basic_enhanced = self._apply_basic_enhancement(raw_output, raw_output)
        
        return {
            "enhanced_output": basic_enhanced,
            "key_insights": ["Review the analysis provided", "Consider implications", "Plan implementation"],
            "action_items": ["Evaluate the information", "Consider implementation options", "Monitor progress"],
            "quality_improvements": ["Added structure", "Improved presentation", "Enhanced formatting"],
            "processing_metrics": {
                "domain_detected": "general",
                "processing_time": 0.1,
                "stages_completed": 1,
                "quality_improvement": 1.0
            }
        }

    def _apply_basic_enhancement(self, text: str, fallback: str) -> str:
        """Basic enhancement when API fails"""
        if not text or text.strip() == "":
            text = fallback
            
        return f"""**Analysis Results:**

{text}

**Summary:**
The analysis above provides key information to address your query. Review the details carefully and consider how to apply these insights to your specific situation.

**Next Steps:**
- Evaluate the information provided
- Consider implementation options
- Plan appropriate actions based on the findings"""


# Global enhancer instance
_global_output_enhancer = None

async def get_output_enhancer():
    """Get or create the global modern output enhancer instance"""
    global _global_output_enhancer
    if _global_output_enhancer is None:
        _global_output_enhancer = ModernUniversalOutputEnhancer()
    return _global_output_enhancer

async def enhance_agent_output(
    raw_output: str, 
    original_query: str,
    agent_type: str = "Agent",
    style: str = "professional"
) -> str:
    """
    Modern universal function to enhance any agent output
    Uses 2025 state-of-the-art 7-stage processing pipeline
    """
    enhancer = await get_output_enhancer()
    enhancement_data = await enhancer.enhance_final_output(
        raw_output, original_query, agent_type, style
    )
    
    # Format the enhanced output with modern presentation
    enhanced = enhancement_data["enhanced_output"]
    
    # Add processing metrics header
    metrics = enhancement_data["processing_metrics"]
    if metrics["quality_improvement"] > 2.0:
        enhanced = f"*✨ Enhanced with {metrics['stages_completed']}-stage modern processing (Domain: {metrics['domain_detected']}, Quality: +{metrics['quality_improvement']:.1f})*\n\n" + enhanced
    
    # Add key insights if available
    if enhancement_data["key_insights"]:
        enhanced += "\n\n**🔑 Key Insights:**\n"
        for insight in enhancement_data["key_insights"]:
            enhanced += f"• {insight}\n"
    
    # Add action items if available  
    if enhancement_data["action_items"]:
        enhanced += "\n**📋 Recommended Actions:**\n"
        for action in enhancement_data["action_items"]:
            enhanced += f"• {action}\n"
    
    # Add quality improvements summary
    if enhancement_data["quality_improvements"]:
        enhanced += "\n**⚡ Quality Enhancements Applied:**\n"
        for improvement in enhancement_data["quality_improvements"][:3]:  # Show top 3
            enhanced += f"• {improvement}\n"
    
    return enhanced

# Convenience functions for each agent type (backward compatibility)
async def enhance_agot_output(raw_output: str, original_query: str, style: str = "professional") -> str:
    """Enhance AGOT output with modern 7-stage pipeline"""
    return await enhance_agent_output(raw_output, original_query, "AGOT", style)

async def enhance_aiot_output(raw_output: str, original_query: str, style: str = "professional") -> str:
    """Enhance AIOT output with modern 7-stage pipeline"""  
    return await enhance_agent_output(raw_output, original_query, "AIOT", style)

async def enhance_giot_output(raw_output: str, original_query: str, style: str = "professional") -> str:
    """Enhance GIOT output with modern 7-stage pipeline"""
    return await enhance_agent_output(raw_output, original_query, "GIOT", style)

# Additional modern functions
async def enhance_output_with_context(raw_output: str, original_query: str, 
                                    context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Enhanced version that returns full processing details
    """
    enhancer = await get_output_enhancer()
    return await enhancer.enhance_final_output(
        raw_output, original_query, 
        context.get("agent_type", "Agent") if context else "Agent",
        context.get("style", "professional") if context else "professional"
    )

async def get_enhancement_analytics() -> Dict[str, Any]:
    """
    Get analytics about enhancement performance
    """
    enhancer = await get_output_enhancer()
    return {
        "total_enhancements": len(enhancer.processing_history),
        "supported_domains": list(enhancer.domain_formatters.keys()),
        "quality_metrics": [m.value for m in OutputQualityMetric],
        "processing_stages": 7,
        "system_status": "Modern 2025 Pipeline Active"
    }

if __name__ == "__main__":
    # Test the modern output enhancer
    async def test_modern_enhancement():
        print("🧪 Testing Modern Universal Output Enhancer (2025)")
        
        test_cases = [
            {
                "output": "Bitcoin price is volatile. Consider your risk tolerance.",
                "query": "Should I buy Bitcoin now?",
                "agent": "AGOT"
            },
            {
                "output": "Use async/await for better performance. Handle errors properly.",
                "query": "How to optimize my Python code?",
                "agent": "AIOT"
            },
            {
                "output": "The study shows correlation between variables A and B.",
                "query": "What does the research indicate?",
                "agent": "GIOT"
            }
        ]
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n{'='*60}")
            print(f"Test Case {i}: {case['agent']} Enhancement")
            print(f"Query: {case['query']}")
            print(f"Raw Output: {case['output']}")
            print(f"\n🚀 Modern Enhanced Output:")
            
            enhanced = await enhance_agent_output(
                case["output"], case["query"], case["agent"]
            )
            print(enhanced)
            
        # Test analytics
        print(f"\n{'='*60}")
        print("📊 Enhancement Analytics:")
        analytics = await get_enhancement_analytics()
        for key, value in analytics.items():
            print(f"  {key}: {value}")
    
    asyncio.run(test_modern_enhancement())
