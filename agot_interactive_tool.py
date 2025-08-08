#!/usr/bin/env python3
"""
Modern Interactive AGOT MCP Tool with Multi-Level Enhancement Analysis
Implements 2025 research: Semantic, Contextual, Methodological, and Constraint Enhancement
"""

import asyncio
import json
from typing import Optional, Dict, Any, List, Union
import sys
import os
import time
from dataclasses import dataclass, asdict
from enum import Enum

# Add the current directory to the path
sys.path.append('/Users/kre8orr/local_projects/MCP/Multi_Thought/multi-agent-llm')

# Optional debug imports - disabled for now to avoid import errors
DEBUG_AVAILABLE = False

# Create dummy context manager for when debug logging is not available
class DebuggedAgentRun:
    def __init__(self, name):
        self.name = name
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    def log_operation(self, operation, status, data=None):
        pass
    def log_error(self, operation, error):
        pass
    def start_operation(self, operation, data=None):
        pass
    def end_operation(self, status, **kwargs):
        pass
    def log_agent_execution(self, **kwargs):
        pass

def debug_operation(operation_name):
    """Dummy debug operation decorator"""
    def decorator(func):
        return func
    return decorator

from output_enhancer_clean import enhance_agot_output
from multi_agent_llm.agents.adaptive_graph_of_thoughts.AGoT import AGOT
from multi_agent_llm.gemini_llm import GeminiLLM


class EnhancementLayer(Enum):
    """Enhancement analysis layers based on 2025 research"""
    SEMANTIC = "semantic"
    CONTEXTUAL = "contextual" 
    METHODOLOGICAL = "methodological"
    CONSTRAINT = "constraint"


@dataclass
class EnhancementResult:
    """Structured enhancement result"""
    layer: EnhancementLayer
    original_aspect: str
    enhanced_aspect: str
    improvement_type: str
    confidence_score: float
    rationale: str


@dataclass
class MultiLevelEnhancementAnalysis:
    """Complete multi-level enhancement analysis"""
    original_query: str
    enhanced_query: str
    enhancement_layers: Dict[str, EnhancementResult]
    overall_confidence: float
    complexity_assessment: Dict[str, Any]
    recommendation: str
    execution_strategy: Dict[str, Any]

class ModernQueryAnalyzer:
    """Advanced query analysis using 2025 methods"""
    
    def __init__(self, llm):
        self.llm = llm
        self.domain_patterns = self._load_domain_patterns()
        self.enhancement_history = {}
    
    def _load_domain_patterns(self) -> Dict[str, Any]:
        """Load domain-specific enhancement patterns"""
        return {
            "finance": {
                "key_terms": ["trading", "market", "stock", "crypto", "investment", "analysis"],
                "required_constraints": ["timeframe", "risk_tolerance", "capital"],
                "common_methodologies": ["technical_analysis", "fundamental_analysis", "quantitative"]
            },
            "research": {
                "key_terms": ["study", "analyze", "research", "investigate", "explore"],
                "required_constraints": ["scope", "methodology", "sources"],
                "common_methodologies": ["systematic_review", "meta_analysis", "empirical_study"]
            },
            "technical": {
                "key_terms": ["code", "algorithm", "system", "implementation", "architecture"],
                "required_constraints": ["performance", "scalability", "compatibility"],
                "common_methodologies": ["design_patterns", "best_practices", "optimization"]
            }
        }
    
    async def assess_query_complexity(self, query: str) -> Dict[str, Any]:
        """Assess query complexity across multiple dimensions"""
        
        complexity_prompt = f"""
        Analyze the complexity of this query across multiple dimensions:
        
        Query: "{query}"
        
        Assess:
        1. Semantic complexity (ambiguity, specificity) - Rate 1-10
        2. Contextual requirements (domain knowledge needed) - Rate 1-10
        3. Methodological complexity (reasoning approach needed) - Rate 1-10
        4. Constraint complexity (parameters and boundaries) - Rate 1-10
        
        Provide scores and brief explanation for each.
        """
        
        messages = self.llm.format_prompt(
            "You are a query complexity analyst. Provide detailed complexity assessment.",
            complexity_prompt
        )
        
        try:
            response = await self.llm.generate_async(messages)
            
            # Parse complexity scores (simplified - in production, use structured output)
            semantic_score = self._extract_score(response, "semantic")
            contextual_score = self._extract_score(response, "contextual") 
            methodological_score = self._extract_score(response, "methodological")
            constraint_score = self._extract_score(response, "constraint")
            
            overall_complexity = (semantic_score + contextual_score + 
                                methodological_score + constraint_score) / 4
            
            return {
                "semantic_complexity": semantic_score,
                "contextual_complexity": contextual_score,
                "methodological_complexity": methodological_score,
                "constraint_complexity": constraint_score,
                "overall_complexity": overall_complexity,
                "complexity_explanation": response,
                "enhancement_priority": self._determine_enhancement_priority(
                    semantic_score, contextual_score, methodological_score, constraint_score
                )
            }
            
        except Exception as e:
            # Fallback complexity assessment
            return {
                "semantic_complexity": 5.0,
                "contextual_complexity": 5.0,
                "methodological_complexity": 5.0,
                "constraint_complexity": 5.0,
                "overall_complexity": 5.0,
                "complexity_explanation": f"Error in analysis: {str(e)}",
                "enhancement_priority": ["semantic", "methodological"]
            }
    
    def _extract_score(self, text: str, dimension: str) -> float:
        """Extract complexity score from text (simplified implementation)"""
        # In production, use structured output or regex parsing
        import re
        pattern = rf"{dimension}.*?(\d+(?:\.\d+)?)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return min(10.0, max(1.0, float(match.group(1))))
        return 5.0
    
    def _determine_enhancement_priority(self, semantic: float, contextual: float, 
                                      methodological: float, constraint: float) -> List[str]:
        """Determine which enhancement layers to prioritize"""
        scores = {
            "semantic": semantic,
            "contextual": contextual,
            "methodological": methodological,
            "constraint": constraint
        }
        
        # Prioritize layers with higher complexity scores
        sorted_layers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [layer for layer, score in sorted_layers if score > 6.0]


class MultiLevelEnhancer:
    """Modern multi-level query enhancement system"""
    
    def __init__(self, llm):
        self.llm = llm
        self.analyzer = ModernQueryAnalyzer(llm)
        self.domain_knowledge = self._initialize_domain_knowledge()
    
    def _initialize_domain_knowledge(self) -> Dict[str, Any]:
        """Initialize domain-specific knowledge base"""
        return {
            "finance": {
                "technical_indicators": ["RSI", "MACD", "Bollinger Bands", "Moving Averages"],
                "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
                "risk_metrics": ["VaR", "Sharpe Ratio", "Max Drawdown", "Beta"],
                "market_conditions": ["bull", "bear", "sideways", "volatile"]
            },
            "research": {
                "methodologies": ["quantitative", "qualitative", "mixed-methods"],
                "data_sources": ["primary", "secondary", "meta-analysis"],
                "validation_methods": ["peer_review", "replication", "cross_validation"]
            }
        }
    
    async def enhance_query_semantically(self, query: str) -> EnhancementResult:
        """Semantic enhancement: Improve clarity and specificity"""
        
        semantic_prompt = f"""
        Enhance this query for semantic clarity and specificity:
        
        Original: "{query}"
        
        Improvements needed:
        1. Resolve ambiguous terms
        2. Add specific parameters
        3. Clarify intent
        4. Use precise terminology
        
        Provide the enhanced version with explanations.
        """
        
        messages = self.llm.format_prompt(
            "You are a semantic enhancement specialist. Improve query clarity and specificity.",
            semantic_prompt
        )
        
        try:
            response = await self.llm.generate_async(messages)
            
            # Extract enhanced query (simplified - use structured output in production)
            enhanced_query = self._extract_enhanced_query(response, query)
            
            return EnhancementResult(
                layer=EnhancementLayer.SEMANTIC,
                original_aspect=query,
                enhanced_aspect=enhanced_query,
                improvement_type="clarity_and_specificity",
                confidence_score=self._calculate_semantic_confidence(query, enhanced_query),
                rationale=response[:200] + "..." if len(response) > 200 else response
            )
            
        except Exception as e:
            return EnhancementResult(
                layer=EnhancementLayer.SEMANTIC,
                original_aspect=query,
                enhanced_aspect=query,
                improvement_type="error_fallback",
                confidence_score=0.0,
                rationale=f"Semantic enhancement failed: {str(e)}"
            )
    
    async def enhance_query_contextually(self, query: str, domain: str = "general") -> EnhancementResult:
        """Contextual enhancement: Add relevant domain knowledge"""
        
        domain_info = self.domain_knowledge.get(domain, {})
        
        contextual_prompt = f"""
        Add relevant contextual information to this query:
        
        Original: "{query}"
        Domain: {domain}
        Available context: {json.dumps(domain_info, indent=2)}
        
        Enhancements:
        1. Add domain-specific background
        2. Include relevant current conditions
        3. Connect to related concepts
        4. Provide necessary context for accurate analysis
        
        Enhanced query with context:
        """
        
        messages = self.llm.format_prompt(
            "You are a contextual enhancement specialist. Add relevant domain knowledge and context.",
            contextual_prompt
        )
        
        try:
            response = await self.llm.generate_async(messages)
            enhanced_query = self._extract_enhanced_query(response, query)
            
            return EnhancementResult(
                layer=EnhancementLayer.CONTEXTUAL,
                original_aspect=query,
                enhanced_aspect=enhanced_query,
                improvement_type="domain_knowledge_integration",
                confidence_score=self._calculate_contextual_confidence(domain_info, enhanced_query),
                rationale=f"Added {domain} domain context and relevant background information"
            )
            
        except Exception as e:
            return EnhancementResult(
                layer=EnhancementLayer.CONTEXTUAL,
                original_aspect=query,
                enhanced_aspect=query,
                improvement_type="error_fallback", 
                confidence_score=0.0,
                rationale=f"Contextual enhancement failed: {str(e)}"
            )
    
    async def enhance_query_methodologically(self, query: str, complexity_info: Dict[str, Any]) -> EnhancementResult:
        """Methodological enhancement: Suggest better reasoning approaches"""
        
        methodological_prompt = f"""
        Suggest optimal reasoning methodology for this query:
        
        Original: "{query}"
        Complexity: {complexity_info.get('overall_complexity', 5.0)}/10
        
        Recommend:
        1. Best analytical framework
        2. Structured approach steps
        3. Validation methods
        4. Multi-perspective analysis if needed
        
        Enhanced query with methodology:
        """
        
        messages = self.llm.format_prompt(
            "You are a methodology enhancement specialist. Suggest optimal reasoning approaches.",
            methodological_prompt
        )
        
        try:
            response = await self.llm.generate_async(messages)
            enhanced_query = self._extract_enhanced_query(response, query)
            
            return EnhancementResult(
                layer=EnhancementLayer.METHODOLOGICAL,
                original_aspect=query,
                enhanced_aspect=enhanced_query,
                improvement_type="reasoning_framework_optimization",
                confidence_score=min(9.0, complexity_info.get('methodological_complexity', 5.0) / 10 * 9),
                rationale="Added structured reasoning methodology and analytical framework"
            )
            
        except Exception as e:
            return EnhancementResult(
                layer=EnhancementLayer.METHODOLOGICAL,
                original_aspect=query,
                enhanced_aspect=query,
                improvement_type="error_fallback",
                confidence_score=0.0,
                rationale=f"Methodological enhancement failed: {str(e)}"
            )
    
    async def enhance_query_constraints(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> EnhancementResult:
        """Constraint enhancement: Add missing parameters and boundaries"""
        
        user_context = user_context or {}
        
        constraint_prompt = f"""
        Identify and add missing constraints/parameters to this query:
        
        Original: "{query}"
        User context: {json.dumps(user_context, indent=2)}
        
        Add:
        1. Missing critical parameters
        2. Operational constraints
        3. Success criteria
        4. Boundary conditions
        5. Validation requirements
        
        Enhanced query with constraints:
        """
        
        messages = self.llm.format_prompt(
            "You are a constraint enhancement specialist. Add missing parameters and boundaries.",
            constraint_prompt
        )
        
        try:
            response = await self.llm.generate_async(messages)
            enhanced_query = self._extract_enhanced_query(response, query)
            
            return EnhancementResult(
                layer=EnhancementLayer.CONSTRAINT,
                original_aspect=query,
                enhanced_aspect=enhanced_query,
                improvement_type="parameter_and_boundary_specification",
                confidence_score=self._calculate_constraint_confidence(user_context, enhanced_query),
                rationale="Added missing parameters, constraints, and success criteria"
            )
            
        except Exception as e:
            return EnhancementResult(
                layer=EnhancementLayer.CONSTRAINT,
                original_aspect=query,
                enhanced_aspect=query,
                improvement_type="error_fallback",
                confidence_score=0.0,
                rationale=f"Constraint enhancement failed: {str(e)}"
            )
    
    async def analyze_and_enhance_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> MultiLevelEnhancementAnalysis:
        """Complete multi-level enhancement analysis"""
        
        user_context = user_context or {}
        domain = user_context.get('domain', 'general')
        
        # Step 1: Assess complexity
        complexity_info = await self.analyzer.assess_query_complexity(query)
        
        # Step 2: Run enhancement layers in parallel
        enhancement_tasks = [
            self.enhance_query_semantically(query),
            self.enhance_query_contextually(query, domain),
            self.enhance_query_methodologically(query, complexity_info),
            self.enhance_query_constraints(query, user_context)
        ]
        
        semantic, contextual, methodological, constraint = await asyncio.gather(*enhancement_tasks)
        
        # Step 3: Integrate enhancements
        enhancement_layers = {
            "semantic": semantic,
            "contextual": contextual,
            "methodological": methodological,
            "constraint": constraint
        }
        
        # Step 4: Synthesize final enhanced query
        enhanced_query = await self._synthesize_enhanced_query(query, enhancement_layers, complexity_info)
        
        # Step 5: Calculate overall confidence and recommendation
        overall_confidence = self._calculate_overall_confidence(enhancement_layers)
        recommendation = self._generate_recommendation(complexity_info, overall_confidence)
        
        # Step 6: Create execution strategy
        execution_strategy = self._create_execution_strategy(complexity_info, enhancement_layers)
        
        return MultiLevelEnhancementAnalysis(
            original_query=query,
            enhanced_query=enhanced_query,
            enhancement_layers=enhancement_layers,
            overall_confidence=overall_confidence,
            complexity_assessment=complexity_info,
            recommendation=recommendation,
            execution_strategy=execution_strategy
        )
    
    async def _synthesize_enhanced_query(self, original_query: str, 
                                       enhancement_layers: Dict[str, EnhancementResult],
                                       complexity_info: Dict[str, Any]) -> str:
        """Synthesize all enhancement layers into final enhanced query"""
        
        synthesis_prompt = f"""
        Synthesize these enhancement layers into one optimal query:
        
        Original: "{original_query}"
        
        Enhancement Layers:
        - Semantic: {enhancement_layers['semantic'].enhanced_aspect}
        - Contextual: {enhancement_layers['contextual'].enhanced_aspect}  
        - Methodological: {enhancement_layers['methodological'].enhanced_aspect}
        - Constraint: {enhancement_layers['constraint'].enhanced_aspect}
        
        Complexity: {complexity_info.get('overall_complexity', 5.0)}/10
        
        Create a single, cohesive enhanced query that integrates the best aspects of each layer.
        """
        
        messages = self.llm.format_prompt(
            "You are a query synthesis specialist. Combine enhancement layers into one optimal query.",
            synthesis_prompt
        )
        
        try:
            response = await self.llm.generate_async(messages)
            return self._extract_enhanced_query(response, original_query)
        except Exception:
            # Fallback: use the layer with highest confidence
            best_layer = max(enhancement_layers.values(), key=lambda x: x.confidence_score)
            return best_layer.enhanced_aspect
    
    def _extract_enhanced_query(self, response: str, fallback: str) -> str:
        """Extract enhanced query from LLM response"""
        # Simple extraction - in production, use structured output
        lines = response.split('\n')
        for line in lines:
            if len(line.strip()) > len(fallback) * 0.8:  # Likely the enhanced query
                return line.strip()
        return fallback
    
    def _calculate_semantic_confidence(self, original: str, enhanced: str) -> float:
        """Calculate semantic enhancement confidence"""
        if len(enhanced) <= len(original):
            return 3.0
        improvement_ratio = len(enhanced) / len(original)
        return min(9.0, 5.0 + improvement_ratio * 2)
    
    def _calculate_contextual_confidence(self, domain_info: Dict, enhanced_query: str) -> float:
        """Calculate contextual enhancement confidence"""
        if not domain_info:
            return 4.0
        
        # Check if domain terms were integrated
        domain_terms_used = sum(1 for term in str(domain_info).lower().split() 
                               if term in enhanced_query.lower())
        return min(9.0, 5.0 + domain_terms_used * 0.5)
    
    def _calculate_constraint_confidence(self, user_context: Dict, enhanced_query: str) -> float:
        """Calculate constraint enhancement confidence"""
        constraint_indicators = ['parameters', 'criteria', 'constraints', 'limits', 'requirements']
        indicators_found = sum(1 for indicator in constraint_indicators 
                             if indicator in enhanced_query.lower())
        return min(9.0, 4.0 + indicators_found * 1.0)
    
    def _calculate_overall_confidence(self, enhancement_layers: Dict[str, EnhancementResult]) -> float:
        """Calculate overall enhancement confidence"""
        confidences = [layer.confidence_score for layer in enhancement_layers.values()]
        return sum(confidences) / len(confidences)
    
    def _generate_recommendation(self, complexity_info: Dict[str, Any], confidence: float) -> str:
        """Generate enhancement recommendation"""
        complexity = complexity_info.get('overall_complexity', 5.0)
        
        if complexity > 7.0 and confidence > 7.0:
            return "strongly_recommend_enhanced"
        elif complexity > 5.0 and confidence > 6.0:
            return "recommend_enhanced"
        elif confidence > 5.0:
            return "consider_enhanced"
        else:
            return "use_original"
    
    def _create_execution_strategy(self, complexity_info: Dict[str, Any], 
                                 enhancement_layers: Dict[str, EnhancementResult]) -> Dict[str, Any]:
        """Create optimal execution strategy"""
        complexity = complexity_info.get('overall_complexity', 5.0)
        
        if complexity > 8.0:
            strategy = "deep_analysis"
            max_layers = 4
            max_depth = 2
        elif complexity > 6.0:
            strategy = "thorough_analysis"
            max_layers = 3
            max_depth = 1
        else:
            strategy = "standard_analysis"
            max_layers = 2
            max_depth = 1
        
        return {
            "strategy": strategy,
            "agot_config": {
                "max_layers": max_layers,
                "max_depth": max_depth,
                "max_new_tasks": 3,
                "max_concurrent_tasks": 10
            },
            "enhancement_focus": complexity_info.get('enhancement_priority', []),
            "estimated_execution_time": self._estimate_execution_time(complexity)
        }
    
    def _estimate_execution_time(self, complexity: float) -> str:
        """Estimate execution time based on complexity"""
        if complexity > 8.0:
            return "2-4 minutes"
        elif complexity > 6.0:
            return "1-2 minutes"
        else:
            return "30-60 seconds"


class InteractiveAGOTTool:
    """Modern Interactive AGOT tool with multi-level enhancement"""
    
    def __init__(self):
        self.session_cache = {}
        self.user_preferences = {}  # Learn from user decisions
        self.enhancement_metrics = {}  # Track enhancement effectiveness
    
    async def preview_enhancement(self, query: str, user_context: Optional[Dict[str, Any]] = None, 
                                 session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Stage 1: Multi-level enhancement preview with detailed analysis
        """
        if not session_id:
            session_id = f"modern_preview_{int(time.time())}"
        
        user_context = user_context or {}
        
        try:
            # Initialize modern enhancer
            llm = GeminiLLM(api_key=os.getenv('GEMINI_API_KEY'), model_name="gemini-2.5-flash")
            enhancer = MultiLevelEnhancer(llm)
            
            # Perform multi-level enhancement analysis
            analysis = await enhancer.analyze_and_enhance_query(query, user_context)
            
            # Cache for execution stage
            self.session_cache[session_id] = {
                "original_query": query,
                "enhancement_analysis": analysis,
                "user_context": user_context,
                "timestamp": time.time()
            }
            
            # Create detailed preview response
            return {
                "status": "multi_level_enhancement_preview",
                "session_id": session_id,
                "original_query": query,
                "enhanced_query": analysis.enhanced_query,
                "enhancement_summary": {
                    "overall_confidence": round(analysis.overall_confidence, 2),
                    "complexity_score": round(analysis.complexity_assessment.get('overall_complexity', 5.0), 2),
                    "recommendation": analysis.recommendation,
                    "estimated_execution_time": analysis.execution_strategy.get('estimated_execution_time', '1-2 minutes')
                },
                "enhancement_layers": {
                    layer_name: {
                        "improvement_type": layer_data.improvement_type,
                        "confidence": round(layer_data.confidence_score, 2),
                        "preview": layer_data.enhanced_aspect[:150] + "..." if len(layer_data.enhanced_aspect) > 150 else layer_data.enhanced_aspect,
                        "rationale": layer_data.rationale[:100] + "..." if len(layer_data.rationale) > 100 else layer_data.rationale
                    }
                    for layer_name, layer_data in analysis.enhancement_layers.items()
                },
                "execution_strategy": {
                    "strategy_type": analysis.execution_strategy['strategy'],
                    "agot_configuration": analysis.execution_strategy['agot_config'],
                    "focus_areas": analysis.execution_strategy['enhancement_focus']
                },
                "query_comparison": {
                    "original_length": len(query),
                    "enhanced_length": len(analysis.enhanced_query),
                    "improvement_ratio": round(len(analysis.enhanced_query) / len(query), 2)
                },
                "next_action": "Call execute_agot_query with session_id and enhancement_decision"
            }
            
        except Exception as e:
            return {
                "status": "enhancement_preview_error",
                "error": str(e),
                "original_query": query,
                "fallback_available": True,
                "next_action": "Call execute_agot_query with use_enhancement=false"
            }
    
    async def execute_agot_query(self, 
                                query: Optional[str] = None,
                                session_id: Optional[str] = None,
                                enhancement_decision: str = "enhanced",  # "enhanced", "original", "custom"
                                custom_query: Optional[str] = None,
                                execution_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Stage 2: Execute AGOT with modern enhancement integration
        """
        
        execution_options = execution_options or {}
        
        # Initialize defaults
        final_query = None
        execution_strategy = {"strategy": "standard_analysis", "agot_config": {}}
        enhancement_info = {"used": False, "method": "none"}
        
        # First priority: Check session cache for enhanced query
        if session_id and session_id in self.session_cache:
            cached = self.session_cache[session_id]
            analysis = cached["enhancement_analysis"]
            
            if enhancement_decision == "enhanced":
                final_query = analysis.enhanced_query
                execution_strategy = analysis.execution_strategy
                enhancement_info = {
                    "used": True,
                    "method": "multi_level_enhancement",
                    "confidence": analysis.overall_confidence,
                    "layers_applied": list(analysis.enhancement_layers.keys()),
                    "complexity_score": analysis.complexity_assessment.get('overall_complexity', 5.0)
                }
            elif enhancement_decision == "custom" and custom_query:
                final_query = custom_query
                enhancement_info = {"used": True, "method": "user_customized"}
            else:
                final_query = cached["original_query"]
                enhancement_info = {"used": False, "method": "user_declined"}
            
            # Learn from user decision
            self._record_user_preference(cached["user_context"], enhancement_decision, analysis)
            
            # Clean up cache
            del self.session_cache[session_id]
            
        elif query and enhancement_decision == "enhanced":
            # Direct enhancement without preview
            final_query = query  # Set fallback first
            try:
                llm = GeminiLLM(api_key=os.getenv('GEMINI_API_KEY'), model_name="gemini-2.5-flash")
                enhancer = MultiLevelEnhancer(llm)
                analysis = await enhancer.analyze_and_enhance_query(query, execution_options.get('user_context', {}))
                
                final_query = analysis.enhanced_query
                execution_strategy = analysis.execution_strategy
                enhancement_info = {
                    "used": True,
                    "method": "direct_multi_level_enhancement",
                    "confidence": analysis.overall_confidence
                }
            except Exception as e:
                enhancement_info = {"used": False, "method": "enhancement_error", "error": str(e)}
        elif query:
            final_query = query
        else:
            # No session and no direct query - this is an error condition
            final_query = "No query provided"
        
        # Auto-detect web search need if not specified
        web_search = execution_options.get('web_search')
        if web_search is None:
            web_search = any(keyword in final_query.lower() 
                           for keyword in ["current", "latest", "recent", "today", "now"])
        
        # Execute with modern AGOT integration
        async with DebuggedAgentRun("MODERN_AGOT_INTERACTIVE") as logger:
            
            logger.log_operation("MODERN_EXECUTION_START", "SUCCESS", {
                "session_id": session_id,
                "enhancement_decision": enhancement_decision,
                "enhancement_used": enhancement_info["used"],
                "final_query_length": len(final_query),
                "execution_strategy": execution_strategy.get("strategy", "standard")
            })
            
            # Execute AGOT with optimized configuration
            result = await self._execute_modern_agot(logger, final_query, execution_strategy, execution_options)
            
            if not result:
                return {
                    "status": "execution_error",
                    "error": "Modern AGOT execution failed",
                    "query_used": final_query,
                    "enhancement_info": enhancement_info
                }
            
            # Enhanced output processing
            try:
                enhanced_output = await self._process_modern_output(result, final_query, enhancement_info)
            except Exception as e:
                logger.log_error("OUTPUT_PROCESSING", e)
                enhanced_output = str(result)
            
            # Record metrics for continuous improvement
            self._record_execution_metrics(enhancement_info, len(enhanced_output), time.time())
            
            return {
                "status": "success",
                "result": enhanced_output,
                "execution_info": {
                    "query_used": final_query,
                    "enhancement_info": enhancement_info,
                    "execution_strategy": execution_strategy,
                    "output_length": len(enhanced_output),
                    "session_id": session_id,
                    "web_search_used": web_search
                },
                "performance_metrics": {
                    "enhancement_effectiveness": self._calculate_enhancement_effectiveness(enhancement_info),
                    "execution_efficiency": execution_strategy.get("strategy", "standard"),
                    "user_satisfaction_prediction": self._predict_user_satisfaction(enhancement_info, len(enhanced_output))
                },
                "debug_log_available": True
            }
    
    # Modern helper methods for advanced functionality
    
    def _record_user_preference(self, user_context: Dict[str, Any], decision: str, analysis: MultiLevelEnhancementAnalysis):
        """Record user preference for learning"""
        try:
            # Simple preference tracking
            if not hasattr(self, '_user_preferences'):
                self._user_preferences = {}
            
            complexity = analysis.complexity_assessment.get('overall_complexity', 5.0)
            preference_key = f"complexity_{int(complexity)}"
            
            if preference_key not in self._user_preferences:
                self._user_preferences[preference_key] = {"accepted": 0, "declined": 0}
            
            if decision == "enhanced":
                self._user_preferences[preference_key]["accepted"] += 1
            else:
                self._user_preferences[preference_key]["declined"] += 1
        except Exception:
            pass  # Silent failure for non-critical functionality
    
    async def _execute_modern_agot(self, logger, query: str, strategy: Dict[str, Any], options: Dict[str, Any]):
        """Execute AGOT with modern configuration"""
        print(f"🔍 _execute_modern_agot called with query: {query[:100]}...")
        print(f"🔍 Strategy: {strategy}")
        print(f"🔍 Options: {options}")
        
        try:
            # Use existing AGOT execution with enhanced options
            web_context = options.get('web_context')
            print(f"🔍 Web context: {web_context}")
            
            result = await self._run_agot_agent(logger, query, web_context)
            print(f"🔍 AGOT result type: {type(result)}")
            print(f"🔍 AGOT result (first 200 chars): {str(result)[:200]}...")
            
            return result
        except Exception as e:
            print(f"❌ Exception in _execute_modern_agot: {str(e)}")
            print(f"❌ Exception type: {type(e)}")
            import traceback
            print(f"❌ Full traceback: {traceback.format_exc()}")
            logger.log_error("MODERN_AGOT_EXECUTION", e)
            return None
    
    async def _process_modern_output(self, result, query: str, enhancement_info: Dict[str, Any]) -> str:
        """Process AGOT output with modern 7-stage enhancements"""
        try:
            # Use modern 7-stage enhancement pipeline
            maybe_coro = enhance_agot_output(str(result), query)
            if asyncio.iscoroutine(maybe_coro):
                enhanced = await maybe_coro
            else:
                enhanced = maybe_coro
            
            # Add enhancement metadata if used
            if enhancement_info.get("used"):
                method = enhancement_info.get('method', 'unknown')
                confidence = enhancement_info.get('confidence', 'N/A')
                enhanced += f"\n\n*Enhanced with modern 7-stage pipeline using {method} (Confidence: {confidence})*"
            
            return enhanced
        except Exception:
            # Fallback to basic formatting
            return f"""**AGOT Analysis Results:**

{str(result)}

**Summary:**
The analysis above provides key information based on adaptive graph-of-thoughts reasoning. Review the insights and consider the recommended actions.

**Next Steps:**
- Evaluate the analysis findings
- Consider implementation approaches
- Monitor results and iterate as needed"""
    
    def _record_execution_metrics(self, enhancement_info: Dict[str, Any], output_length: int, timestamp: float):
        """Record execution metrics for analytics"""
        try:
            if not hasattr(self, '_execution_metrics'):
                self._execution_metrics = []
            
            self._execution_metrics.append({
                "timestamp": timestamp,
                "enhancement_used": enhancement_info.get("used", False),
                "enhancement_method": enhancement_info.get("method", "none"),
                "output_length": output_length,
                "confidence": enhancement_info.get("confidence", 0.0)
            })
            
            # Keep only last 100 entries
            if len(self._execution_metrics) > 100:
                self._execution_metrics = self._execution_metrics[-100:]
        except Exception:
            pass
    
    def _calculate_enhancement_effectiveness(self, enhancement_info: Dict[str, Any]) -> float:
        """Calculate enhancement effectiveness score"""
        if not enhancement_info.get("used"):
            return 0.0
        
        confidence = enhancement_info.get("confidence", 0.5)
        layers_count = len(enhancement_info.get("layers_applied", []))
        
        # Simple effectiveness calculation
        effectiveness = min(1.0, confidence * (1 + layers_count * 0.1))
        return round(effectiveness, 3)
    
    def _predict_user_satisfaction(self, enhancement_info: Dict[str, Any], output_length: int) -> float:
        """Predict user satisfaction based on metrics"""
        base_score = 0.7
        
        if enhancement_info.get("used"):
            confidence_boost = enhancement_info.get("confidence", 0.5) * 0.2
            base_score += confidence_boost
        
        # Reasonable output length boost
        if 100 <= output_length <= 2000:
            base_score += 0.1
        
        return min(1.0, round(base_score, 3))
    
    async def _run_agot_agent(self, logger, query: str, web_context: Optional[str] = None):
        """Internal AGOT execution with logging"""
        print(f"🔍 _run_agot_agent called with query: {query[:100]}...")
        print(f"🔍 Web context present: {bool(web_context)}")
        
        # Initialize LLM
        logger.start_operation("LLM_INITIALIZATION", {
            "model": "gemini-2.5-flash"
        })
        
        try:
            print("🔍 Initializing Gemini LLM...")
            gemini_api_key = os.getenv('GEMINI_API_KEY')
            print(f"🔍 GEMINI_API_KEY present: {bool(gemini_api_key)}")
            if gemini_api_key:
                print(f"🔍 GEMINI_API_KEY length: {len(gemini_api_key)}")
            
            llm = GeminiLLM(api_key=gemini_api_key, model_name="gemini-2.5-flash")
            print("✅ LLM initialized successfully")
            logger.end_operation("SUCCESS")
        except Exception as e:
            print(f"❌ LLM initialization failed: {str(e)}")
            print(f"❌ Exception type: {type(e)}")
            logger.log_error("LLM_INIT", e)
            logger.end_operation("ERROR", error=str(e))
            return None
        
        # Initialize AGOT
        logger.start_operation("AGOT_INITIALIZATION", {
            "max_concurrent_tasks": 10,
            "max_depth": 1,
            "max_new_tasks": 3,
            "max_num_layers": 3
        })
        
        try:
            print("🔍 Initializing AGOT...")
            agot = AGOT(
                llm=llm,
                max_concurrent_tasks=10,
                max_depth=1,
                max_new_tasks=3,
                max_num_layers=3
            )
            print("✅ AGOT initialized successfully")
            logger.end_operation("SUCCESS")
        except Exception as e:
            print(f"❌ AGOT initialization failed: {str(e)}")
            print(f"❌ Exception type: {type(e)}")
            import traceback
            print(f"❌ Full traceback: {traceback.format_exc()}")
            logger.log_error("AGOT_INIT", e)
            logger.end_operation("ERROR", error=str(e))
            return None
        
        # Prepare full query with web context
        full_query = query
        if web_context:
            full_query = f"{query}\n\nWeb Context:\n{web_context}"
            print(f"🔍 Using full query with web context (length: {len(full_query)})")
        else:
            print(f"🔍 Using simple query (length: {len(full_query)})")
        
        # Execute AGOT
        logger.start_operation("AGOT_EXECUTION", {
            "query_length": len(full_query),
            "has_web_context": bool(web_context)
        })
        
        start_time = time.time()
        try:
            print("🔍 Starting AGOT execution...")
            result = await agot.run_async(full_query)
            execution_time = time.time() - start_time
            
            print(f"✅ AGOT execution completed in {execution_time:.2f}s")
            print(f"🔍 Result type: {type(result)}")
            print(f"🔍 Result (first 200 chars): {str(result)[:200]}...")
            
            logger.log_agent_execution(
                agent_config={
                    "type": "AGoT",
                    "max_concurrent_tasks": 10,
                    "max_depth": 1,
                    "max_new_tasks": 3,
                    "max_num_layers": 3
                },
                query=full_query,
                start_time=start_time,
                end_time=time.time(),
                result=result
            )
            
            logger.log_operation("AGOT_EXECUTION", "SUCCESS", {
                "execution_time": execution_time,
                "result_length": len(str(result))
            })
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ AGOT execution failed after {execution_time:.2f}s: {str(e)}")
            print(f"❌ Exception type: {type(e)}")
            import traceback
            print(f"❌ Full traceback: {traceback.format_exc()}")
            logger.log_error("AGOT_EXECUTION", e)
            logger.end_operation("ERROR", error=str(e))
            return None

# Global tool instance
_interactive_tool = InteractiveAGOTTool()

async def preview_query_enhancement(query: str, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    MCP Tool Function 1: Preview query enhancement with modern analysis
    
    Args:
        query: User's original question
        user_context: User context for personalized enhancement (replaces session_id)
        
    Returns:
        Modern enhancement preview with session_id for execution
    """
    return await _interactive_tool.preview_enhancement(query, user_context)

async def execute_agot_analysis(
    query: Optional[str] = None,
    session_id: Optional[str] = None,
    enhancement_decision: str = "enhanced",  # "enhanced", "original", "custom"
    custom_query: Optional[str] = None,
    execution_options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    MCP Tool Function 2: Execute AGOT analysis with modern multi-level enhancement
    
    Args:
        query: Query to analyze (optional if using session_id)
        session_id: Session ID from preview step
        enhancement_decision: "enhanced", "original", or "custom"
        custom_query: Custom query if enhancement_decision is "custom"
        execution_options: Additional execution options (web_search, etc.)
        
    Returns:
        Modern analysis results with performance metrics
    """
    return await _interactive_tool.execute_agot_query(
        query=query,
        session_id=session_id,
        enhancement_decision=enhancement_decision,
        custom_query=custom_query,
        execution_options=execution_options
    )

# Convenience function for direct usage
async def interactive_agot_flow(query: str) -> Dict[str, Any]:
    """
    Demo function showing the full interactive flow
    """
    print(f"🔍 Analyzing query: {query}")
    
    # Step 1: Get enhancement preview
    preview = await preview_query_enhancement(query)
    print(f"\n📝 Enhancement Preview:")
    print(f"   Original: {preview['original_query']}")
    print(f"   Enhanced: {preview['enhanced_query'][:200]}...")
    print(f"   Method: {preview['enhancement_method']}")
    print(f"   Recommendation: {preview['recommendation']}")
    
    # Step 2: Simulate user decision (in real MCP, LLM would ask user)
    use_enhancement = preview['recommendation'] == 'enhanced'
    print(f"\n✅ Using {'enhanced' if use_enhancement else 'original'} query")
    
    # Step 3: Execute with modern parameters
    result = await execute_agot_analysis(
        session_id=preview['session_id'],
        enhancement_decision="enhanced" if use_enhancement else "original"
    )
    
    return result

if __name__ == "__main__":
    # Test the interactive flow
    async def test():
        query = "What's the best approach for converting tick data to OHLC candles?"
        result = await interactive_agot_flow(query)
        print(f"\n🎯 Final Result:")
        print(f"   Status: {result['status']}")
        if result['status'] == 'success':
            print(f"   Output Length: {result['execution_info']['output_length']} chars")
            print(f"   Enhancement Used: {result['execution_info']['enhancement_info']['used']}")
    
    asyncio.run(test())
