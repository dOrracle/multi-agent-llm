#!/usr/bin/env python3
"""
Advanced Query/Input Enhancer for Multi-Agent System
Transforms basic user queries into comprehensive research investigations
"""

import asyncio
import sys
import os
from typing import Dict, List, Optional
from dataclasses import dataclass

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

import google.generativeai as genai_sdk
from google import genai
import json

@dataclass
class EnhancedQuery:
    original: str
    enhanced: str
    sub_queries: List[str]
    context_dimensions: List[str]
    research_angles: List[str]
    confidence_score: float

class AdvancedQueryEnhancer:
    """Advanced query enhancement using modern research methodologies"""
    
    def __init__(self):
        # Use the google.genai client for consistency with other modules
        self.client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
        self.model_name = "gemini-2.5-flash"

    async def enhance_query(self, raw_query: str) -> EnhancedQuery:
        """Transform a basic query into a research-grade investigation"""
        
        enhancement_prompt = f"""Transform this basic query into a comprehensive research investigation.

Original Query: "{raw_query}"

Analyze and enhance using these modern research methodologies:

1. SEMANTIC DECOMPOSITION: Break into core concepts and relationships
2. CONTEXTUAL EXPANSION: Add relevant dimensions (temporal, geographic, technical, economic)  
3. MULTI-ANGLE INVESTIGATION: Consider different perspectives and contrarian views
4. IMPLICIT NEEDS INFERENCE: What does the user really need to know?
5. RESEARCH DEPTH OPTIMIZATION: Structure for progressive investigation

Return a JSON response with this exact structure:
{{
  "enhanced_query": "A comprehensive, research-focused version of the original query",
  "sub_queries": [
    "Specific focused question 1",
    "Specific focused question 2", 
    "Specific focused question 3",
    "Specific focused question 4"
  ],
  "context_dimensions": [
    "Important context factor 1",
    "Important context factor 2",
    "Important context factor 3"
  ],
  "research_angles": [
    "Primary research angle",
    "Alternative perspective", 
    "Contrarian viewpoint",
    "Practical application angle"
  ],
  "confidence_score": 0.85
}}

Make the enhanced query specific, actionable, and designed for deep research. Include temporal context (August 2025), specify domains, and frame for comprehensive investigation."""

        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=enhancement_prompt
            )
            
            # Clean and parse JSON
            text = response.text.strip() if response.text else ""
            if text.startswith('```json'):
                text = text[7:-3]
            elif text.startswith('```'):
                text = text[3:-3]
            
            data = json.loads(text)
            
            return EnhancedQuery(
                original=raw_query,
                enhanced=data["enhanced_query"],
                sub_queries=data["sub_queries"],
                context_dimensions=data["context_dimensions"], 
                research_angles=data["research_angles"],
                confidence_score=data["confidence_score"]
            )
            
        except Exception as e:
            print(f"⚠️  Query enhancement error: {e}")
            # Fallback enhancement
            return EnhancedQuery(
                original=raw_query,
                enhanced=f"Comprehensive analysis of {raw_query} including current trends, implications, and practical applications in August 2025",
                sub_queries=[
                    raw_query,
                    f"What are the current trends related to {raw_query}?",
                    f"What are the implications of {raw_query}?",
                    f"What are practical applications of {raw_query}?"
                ],
                context_dimensions=["Current market conditions", "Regulatory environment", "Technological trends"],
                research_angles=["Technical analysis", "Market impact", "Future outlook", "Practical implementation"],
                confidence_score=0.5
            )

    async def enhance_simple(self, raw_query: str, domain: str = "general") -> Dict:
        """Simple enhancement that returns a dictionary (backward compatibility)"""
        
        enhancement_prompt = f"""Transform this basic user query into a comprehensive, research-focused investigation that will produce the best possible results.

Original Query: "{raw_query}"
Domain: "{domain}"

Apply these enhancement techniques:
1. SEMANTIC EXPANSION: Add relevant context and dimensions
2. SPECIFICITY: Make the query more precise and actionable  
3. COMPLETENESS: Include related aspects the user likely wants to know
4. RESEARCH DEPTH: Structure for thorough investigation
5. CLARITY: Remove ambiguity and improve focus

Return JSON format:
{{
  "enhanced_query": "A comprehensive, well-structured version that will get better results",
  "improvements_made": [
    "How the query was improved 1",
    "How the query was improved 2",
    "How the query was improved 3"
  ],
  "focus_areas": [
    "Key area to investigate 1",
    "Key area to investigate 2", 
    "Key area to investigate 3"
  ]
}}

Make the enhanced query significantly more likely to produce valuable, comprehensive results."""

        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=enhancement_prompt
            )
            
            text = response.text.strip() if response.text else ""
            if text.startswith('```json'):
                text = text[7:-3]
            elif text.startswith('```'):
                text = text[3:-3]
            
            try:
                data = json.loads(text)
                return {
                    "enhanced_query": data.get("enhanced_query", raw_query),
                    "improvements_made": data.get("improvements_made", []),
                    "focus_areas": data.get("focus_areas", [])
                }
            except json.JSONDecodeError:
                return {
                    "enhanced_query": f"Comprehensive analysis of {raw_query}, including current context, implications, and practical applications",
                    "improvements_made": ["Added comprehensiveness", "Improved clarity"],
                    "focus_areas": ["Primary analysis", "Current context", "Practical implications"]
                }
                
        except Exception as e:
            print(f"⚠️  Query enhancement error: {e}")
            return {
                "enhanced_query": raw_query,
                "improvements_made": ["No enhancement applied"],
                "focus_areas": ["Basic inquiry"]
            }

# Global enhancer instance
_global_enhancer = None

async def get_query_enhancer():
    """Get or create the global query enhancer instance"""
    global _global_enhancer
    if _global_enhancer is None:
        _global_enhancer = AdvancedQueryEnhancer()
    return _global_enhancer

async def enhance_user_query(raw_query: str, domain: str = "general") -> Dict:
    """
    Enhance user input query for better agent processing
    Returns both the enhanced query and improvement details
    """
    enhancer = await get_query_enhancer()
    return await enhancer.enhance_simple(raw_query, domain)

async def enhance_user_query_advanced(raw_query: str) -> EnhancedQuery:
    """
    Advanced enhancement that returns a full EnhancedQuery object
    """
    enhancer = await get_query_enhancer()
    return await enhancer.enhance_query(raw_query)

# Integration with research systems
class PowerfulResearchSystem:
    """Complete pipeline: enhance query → deep research"""
    
    def __init__(self):
        self.enhancer = AdvancedQueryEnhancer()
        self.researcher = None  # Your research instance

    async def intelligent_research(self, raw_query: str) -> Dict:
        """Complete pipeline: enhance query → deep research"""
        
        # Step 1: Enhance the query
        enhanced = await self.enhancer.enhance_query(raw_query)
        
        # Step 2: Research using enhanced query and sub-queries
        # Note: Research functionality disabled due to integration issues
        # if self.researcher and hasattr(self.researcher, 'research'):
        #     try:
        #         primary_research = await self.researcher.research(enhanced.enhanced, depth=4)
        #         
        #         # Step 3: Research each angle for comprehensive coverage
        #         angle_research = []
        #         for angle in enhanced.research_angles[:2]:  # Limit to 2 angles for cost
        #             angle_query = f"{enhanced.enhanced} - focus on: {angle}"
        #             angle_result = await self.researcher.research(angle_query, depth=2)
        #             angle_research.append({
        #                 "angle": angle,
        #                 "findings": angle_result.synthesis if hasattr(angle_result, 'synthesis') else str(angle_result)
        #             })
        #         
        #         return {
        #             "original_query": raw_query,
        #             "enhanced_query": enhanced.enhanced,
        #             "primary_research": primary_research.synthesis if hasattr(primary_research, 'synthesis') else str(primary_research),
        #             "alternative_perspectives": angle_research,
        #             "context_dimensions": enhanced.context_dimensions,
        #             "total_sources": len(primary_research.sources) if hasattr(primary_research, 'sources') else 0,
        #             "confidence": enhanced.confidence_score
        #         }
        #     except Exception as e:
        #         print(f"⚠️  Research error: {e}")
        #         # Fallback to query enhancement only
        #         pass
        
        # Return enhanced query structure only (fallback)
        return {
            "original_query": raw_query,
            "enhanced_query": enhanced.enhanced,
            "sub_queries": enhanced.sub_queries,
            "context_dimensions": enhanced.context_dimensions,
            "research_angles": enhanced.research_angles,
            "confidence": enhanced.confidence_score,
            "note": "No researcher configured - returning enhanced query structure only"
        }

# Simple usage function  
async def smart_research(query: str) -> Dict:
    """One-liner for intelligent research with query enhancement"""
    system = PowerfulResearchSystem()
    return await system.intelligent_research(query)

# Intelligent hybrid enhancement system
async def smart_enhance_query(raw_query: str, force_advanced: bool = False) -> Dict:
    """
    Intelligently choose between simple and advanced enhancement based on query complexity
    
    Args:
        raw_query: The original user query
        force_advanced: Force advanced enhancement regardless of complexity analysis
        
    Returns:
        Dict with enhanced_query, method_used, and other relevant data
    """
    
    # Analyze query complexity
    query_length = len(raw_query.split())
    query_lower = raw_query.lower()
    
    # Complex topic indicators
    complex_terms = [
        'investment', 'analysis', 'comparison', 'strategy', 'research', 'evaluate',
        'should i buy', 'what are the risks', 'pros and cons', 'analyze', 'compare',
        'market', 'financial', 'economic', 'regulatory', 'technology', 'blockchain',
        'cryptocurrency', 'bitcoin', 'ethereum', 'stock', 'portfolio', 'risk',
        'long-term', 'short-term', 'future', 'prediction', 'forecast', 'trend'
    ]
    
    # Decision logic indicators
    decision_terms = ['should', 'would', 'could', 'recommend', 'advice', 'choose']
    
    # Multi-faceted question indicators  
    multifaceted_terms = ['and', 'versus', 'vs', 'or', 'both', 'either', 'multiple']
    
    # Calculate complexity score
    complexity_score = 0
    
    # Length factor (longer queries often more complex)
    if query_length > 10: complexity_score += 2
    if query_length > 20: complexity_score += 3
    
    # Complex terms
    complexity_score += sum(1 for term in complex_terms if term in query_lower)
    
    # Decision-making queries (benefit from multiple perspectives)
    complexity_score += sum(2 for term in decision_terms if term in query_lower)
    
    # Multi-faceted queries
    complexity_score += sum(1 for term in multifaceted_terms if term in query_lower)
    
    # Question marks (often indicate complex inquiries)
    if '?' in raw_query: complexity_score += 1
    if raw_query.count('?') > 1: complexity_score += 2
    
    # Determine enhancement method
    use_advanced = force_advanced or complexity_score >= 5
    
    print(f"🤖 Query Complexity Analysis:")
    print(f"   Complexity Score: {complexity_score}")
    print(f"   Method Selected: {'Advanced' if use_advanced else 'Simple'}")
    
    if use_advanced:
        print("   🚀 Using Advanced Enhancement (comprehensive research analysis)")
        advanced_result = await enhance_user_query_advanced(raw_query)
        return {
            "enhanced_query": advanced_result.enhanced,
            "method_used": "advanced",
            "complexity_score": complexity_score,
            "original_query": raw_query,
            "sub_queries": advanced_result.sub_queries,
            "research_angles": advanced_result.research_angles,
            "context_dimensions": advanced_result.context_dimensions,
            "confidence_score": advanced_result.confidence_score
        }
    else:
        print("   ⚡ Using Simple Enhancement (efficient improvement)")
        simple_result = await enhance_user_query(raw_query)
        return {
            "enhanced_query": simple_result["enhanced_query"],
            "method_used": "simple", 
            "complexity_score": complexity_score,
            "original_query": raw_query,
            "improvements_made": simple_result["improvements_made"],
            "focus_areas": simple_result["focus_areas"]
        }

# One-liner function for easy integration
async def enhance_query_auto(raw_query: str) -> str:
    """
    Auto-enhance query and return just the enhanced query string
    Perfect for drop-in replacement in existing code
    """
    result = await smart_enhance_query(raw_query)
    return result["enhanced_query"]

if __name__ == "__main__":
    # Test the enhancer
    async def test():
        enhancer = AdvancedQueryEnhancer()
        
        test_query = "Should I buy Bitcoin now?"
        print(f"Original: {test_query}")
        
        enhanced = await enhancer.enhance_query(test_query)
        print(f"\nEnhanced: {enhanced.enhanced}")
        print(f"Sub-queries: {enhanced.sub_queries}")
        print(f"Research angles: {enhanced.research_angles}")
        print(f"Confidence: {enhanced.confidence_score}")
    
    asyncio.run(test())
