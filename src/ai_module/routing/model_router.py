"""
Model Router - Routes intents to appropriate models.

Routing Rules:
- general_query → gemini-flash (cost-effective for simple queries)
- edit_schematic → gemini-neon-4 (needs design understanding)
- create_design → gemini-neon-4 (complex generation)
- debug_circuit → gemini-neon-4 (technical reasoning)

Can be extended with:
- Cost tracking
- Fallback models
- Performance metrics
"""

from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

from ..intent.schemas import IntentType


class ModelType(str, Enum):
    """Supported model types"""
    GEMINI_FLASH = "gemini-flash"
    GEMINI_NEON_4 = "gemini-neon-4"


@dataclass
class ModelConfig:
    """Configuration for a model"""
    model_id: str  # e.g., "gemini-1.5-flash", "gemini-neon-4-pro"
    display_name: str
    cost_per_token: float  # Approximate cost for pricing
    max_tokens: int
    is_available: bool = True


@dataclass
class RoutingDecision:
    """Result of model routing"""
    model: ModelType
    model_id: str
    intent: IntentType
    reason: str
    estimated_cost: float


# Model configurations
MODEL_CONFIGS: Dict[ModelType, ModelConfig] = {
    ModelType.GEMINI_FLASH: ModelConfig(
        model_id="gemini-1.5-flash",
        display_name="Gemini Flash",
        cost_per_token=0.0000005,  # $0.50 per million tokens
        max_tokens=1000000
    ),
    ModelType.GEMINI_NEON_4: ModelConfig(
        model_id="gemini-neon-4-pro",
        display_name="Gemini Neon 4",
        cost_per_token=0.000002,  # $2.00 per million tokens
        max_tokens=1000000
    )
}

# Routing rules: intent -> model
ROUTING_RULES: Dict[IntentType, ModelType] = {
    "general_query": ModelType.GEMINI_FLASH,
    "edit_schematic": ModelType.GEMINI_NEON_4,
    "create_design": ModelType.GEMINI_NEON_4,
    "debug_circuit": ModelType.GEMINI_NEON_4
}


class ModelRouter:
    """
    Routes intents to appropriate models based on complexity and cost.
    
    Usage:
        router = ModelRouter()
        decision = router.route("general_query")
        print(decision.model_id)  # "gemini-1.5-flash"
    """
    
    def __init__(self, custom_routing: Optional[Dict[IntentType, ModelType]] = None):
        """
        Initialize the router.
        
        Args:
            custom_routing: Override default routing rules
        """
        self.routing_rules = custom_routing or ROUTING_RULES
        self.model_configs = MODEL_CONFIGS
    
    def route(self, intent: IntentType, context: Optional[Dict] = None) -> RoutingDecision:
        """
        Route an intent to a model.
        
        Args:
            intent: The classified intent
            context: Optional context (e.g., user tier, budget constraints)
        
        Returns:
            RoutingDecision with model choice and metadata
        """
        # Get the model for this intent
        model_type = self.routing_rules.get(intent)
        
        if model_type is None:
            # Fallback to flash for unknown intents
            model_type = ModelType.GEMINI_FLASH
        
        # Get model config
        config = self.model_configs.get(model_type)
        if config is None:
            raise ValueError(f"No configuration for model: {model_type}")
        
        # Build reason
        reason = f"Intent '{intent}' routed to {model_type.value} for optimal performance/cost"
        
        # Calculate estimated cost (placeholder - actual cost depends on tokens)
        estimated_cost = config.cost_per_token * 1000  # Example: 1000 tokens
        
        return RoutingDecision(
            model=model_type,
            model_id=config.model_id,
            intent=intent,
            reason=reason,
            estimated_cost=estimated_cost
        )
    
    def get_model_config(self, model_type: ModelType) -> ModelConfig:
        """Get configuration for a specific model"""
        return self.model_configs[model_type]
    
    def list_models(self) -> Dict[str, ModelConfig]:
        """List all available models"""
        return {k.value: v for k, v in self.model_configs.items()}
    
    def list_routing_rules(self) -> Dict[IntentType, ModelType]:
        """List all routing rules"""
        return self.routing_rules
    
    def set_routing_rule(self, intent: IntentType, model_type: ModelType) -> None:
        """Override a routing rule"""
        self.routing_rules[intent] = model_type
    
    def estimate_cost(self, intent: IntentType, token_count: int) -> float:
        """
        Estimate cost for a given intent and token count.
        
        Args:
            intent: The intent
            token_count: Number of tokens in the prompt/response
        
        Returns:
            Estimated cost in dollars
        """
        decision = self.route(intent)
        config = self.model_configs[decision.model]
        return config.cost_per_token * token_count


# Singleton instance
_router: Optional[ModelRouter] = None


def get_model_router() -> ModelRouter:
    """Get the singleton model router"""
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router


def reset_model_router() -> None:
    """Reset the singleton (useful for testing)"""
    global _router
    _router = None
