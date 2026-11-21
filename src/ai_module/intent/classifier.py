"""
Intent Classifier - Classifies queries into design-related intents.

Intents:
- general_query: Basic questions, non-design related
- edit_schematic: Modify existing circuit design
- create_design: Create new circuit design
- debug_circuit: Troubleshoot/fix circuit issues

Uses keyword matching + embedding similarity for accuracy.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .schemas import IntentClassification, IntentType


@dataclass
class IntentPattern:
    """Pattern for matching an intent"""
    intent: IntentType
    keywords: List[str]
    negative_keywords: List[str]  # If these appear, don't match
    description: str


# Define patterns for each intent
INTENT_PATTERNS: List[IntentPattern] = [
    IntentPattern(
        intent="create_design",
        keywords=[
            "create", "new", "build", "design", "make", "construct",
            "circuit", "schema", "schematic", "from scratch",
            "start", "begin", "develop", "assemble"
        ],
        negative_keywords=["edit", "modify", "change", "update", "fix", "debug", "troubleshoot"],
        description="User wants to create a new circuit design"
    ),
    IntentPattern(
        intent="edit_schematic",
        keywords=[
            "edit", "modify", "change", "update", "adjust", "alter",
            "move", "replace", "remove", "add", "swap",
            "circuit", "schematic", "design"
        ],
        negative_keywords=["create", "new", "build", "from scratch", "debug", "fix"],
        description="User wants to modify an existing circuit"
    ),
    IntentPattern(
        intent="debug_circuit",
        keywords=[
            "debug", "fix", "troubleshoot", "error", "problem", "issue",
            "not working", "broken", "fault", "test", "verify",
            "check", "why", "what's wrong", "failure"
        ],
        negative_keywords=["create", "new", "build", "edit", "modify"],
        description="User wants to debug or fix a circuit"
    ),
    IntentPattern(
        intent="general_query",
        keywords=[
            "what", "how", "when", "where", "who", "explain",
            "tell me", "information", "describe", "what is",
            "help", "guide", "tutorial"
        ],
        negative_keywords=[],
        description="General information query"
    )
]


class IntentClassifier:
    """
    Classifies user queries into intents using pattern matching.
    
    Can be extended with ML models for better accuracy.
    """
    
    def __init__(self):
        """Initialize the classifier with patterns"""
        self.patterns = INTENT_PATTERNS
    
    def classify(self, query: str, context: Optional[Dict] = None) -> IntentClassification:
        """
        Classify a query into one of the intents.
        
        Args:
            query: User query text
            context: Optional context (e.g., previous messages, design state)
        
        Returns:
            IntentClassification with intent, confidence, and reasoning
        """
        query_lower = query.lower()
        
        # Score each intent
        scores: Dict[IntentType, float] = {p.intent: 0.0 for p in self.patterns}
        reasoning_parts: Dict[IntentType, List[str]] = {p.intent: [] for p in self.patterns}
        
        for pattern in self.patterns:
            score = 0.0
            matches = []
            
            # Check for positive keywords
            for keyword in pattern.keywords:
                if keyword in query_lower:
                    score += 1.0
                    matches.append(keyword)
            
            # Check for negative keywords
            for neg_keyword in pattern.negative_keywords:
                if neg_keyword in query_lower:
                    score = 0.0
                    matches = []
                    break
            
            # Bonus for multiple matches
            if len(matches) > 1:
                score += 0.5 * (len(matches) - 1)
            
            scores[pattern.intent] = score
            if matches:
                reasoning_parts[pattern.intent] = matches
        
        # Normalize scores to 0-1 range
        max_score = max(scores.values()) if scores.values() else 1.0
        if max_score > 0:
            normalized_scores = {k: v/max_score for k, v in scores.items()}
        else:
            normalized_scores = scores
        
        # Select intent with highest score
        best_intent = max(scores, key=scores.get)
        best_score = normalized_scores[best_intent]
        
        # Build reasoning
        reasoning = f"Matched intent '{best_intent}' with confidence {best_score:.2f}"
        if reasoning_parts[best_intent]:
            reasoning += f" (keywords: {', '.join(reasoning_parts[best_intent])})"
        
        return IntentClassification(
            intent=best_intent,
            confidence=best_score,
            reasoning=reasoning
        )
    
    def classify_with_context(
        self,
        query: str,
        design_id: Optional[str] = None,
        previous_intent: Optional[IntentType] = None
    ) -> IntentClassification:
        """
        Classify with additional context.
        
        Args:
            query: User query
            design_id: Current design ID (if in a design context)
            previous_intent: Previous message's intent
        
        Returns:
            IntentClassification
        """
        # If we're in a design context, bias towards design-related intents
        context = {
            "design_id": design_id,
            "previous_intent": previous_intent
        }
        
        result = self.classify(query, context)
        
        # Context-aware adjustments
        if design_id and result.intent == "general_query" and result.confidence < 0.8:
            # If in a design but low confidence on general, check if it's a design question
            design_keywords = ["circuit", "component", "resistor", "capacitor", "ic", "wiring"]
            query_lower = query.lower()
            
            if any(kw in query_lower for kw in design_keywords):
                # Re-classify as create_design if it's a design-related general question
                result.intent = "create_design"
                result.confidence = 0.7
                result.reasoning = "Design context detected with design-related keywords"
        
        return result


# Singleton instance
_classifier: Optional[IntentClassifier] = None


def get_intent_classifier() -> IntentClassifier:
    """Get the singleton intent classifier"""
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier


def reset_intent_classifier() -> None:
    """Reset the singleton (useful for testing)"""
    global _classifier
    _classifier = None
