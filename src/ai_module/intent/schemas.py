"""
Pydantic schemas for intent classification
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field

# Intent types
IntentType = Literal["general_query", "edit_schematic", "create_design", "debug_circuit"]


class IntentClassification(BaseModel):
    """Result of intent classification"""
    intent: IntentType
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0-1)")
    reasoning: Optional[str] = Field(default=None, description="Why this intent was chosen")


class ClassifyRequest(BaseModel):
    """Request to classify a query"""
    query: str = Field(..., description="User query to classify")
    context: Optional[Dict] = Field(default=None, description="Additional context (e.g., previous messages)")
