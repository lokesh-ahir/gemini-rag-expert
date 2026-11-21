"""
AI Module Main - Entry point for the modular AI system.

This module provides:
- Intent classification (general_query, edit_schematic, create_design, debug_circuit)
- Model routing (Gemini Flash for general, Gemini Neon 4 for design tasks)
- Circuit building from Qdrant components (1800+ components)
- Markdown guide generation
- PostgreSQL chat history (workspace_id, design_id based)
- Cost tracking

Usage:
    from ai_module import AIModule
    
    # Initialize
    module = AIModule(
        db_url="postgresql://user:pass@localhost/db",
        qdrant_url="http://localhost:6333"
    )
    
    # Classify intent
    intent = module.classify_intent("Create an amplifier circuit")
    
    # Route to model
    model = module.route_model(intent.intent)
    
    # Build circuit
    circuit = module.build_circuit("Create an amplifier with LM386")
    
    # Generate guide
    guide = module.generate_guide(circuit)
    
    # Chat with history
    module.add_chat_message(
        workspace_id="...",
        design_id="...",
        role="user",
        content="Create an amplifier"
    )
"""

from typing import Optional, Dict, Any, List, Tuple
from uuid import UUID

from .config import AIModuleConfig, get_config
from .intent.classifier import IntentClassifier, get_intent_classifier
from .intent.schemas import IntentClassification, IntentType
from .routing.model_router import ModelRouter, ModelType, get_model_router, RoutingDecision
from .routing.cost_tracker import CostTracker, get_cost_tracker
from .rag.circuit_retriever import CircuitRetriever, get_circuit_retriever, Component
from .circuit.builder import CircuitBuilder, get_circuit_builder
from .circuit.schemas import Circuit, CircuitValidationError
from .guide.generator import GuideGenerator, get_guide_generator
from .chat.history import ChatHistoryService, get_chat_history_service


class AIModule:
    """
    Main AI module that orchestrates all sub-modules.
    
    This is the primary interface for the modular AI system.
    All functionality is accessible through this class.
    """
    
    def __init__(
        self,
        db_url: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        config: Optional[AIModuleConfig] = None
    ):
        """
        Initialize the AI module.
        
        Args:
            db_url: PostgreSQL connection URL
            qdrant_url: Qdrant server URL
            qdrant_api_key: Qdrant API key (for cloud)
            google_api_key: Google API key for Gemini models
            config: Optional pre-configured config
        """
        # Load or create config
        if config:
            self.config = config
        else:
            self.config = AIModuleConfig(
                database=AIModuleConfig.DatabaseConfig(url=db_url or ""),
                qdrant=AIModuleConfig.QdrantConfig(
                    url=qdrant_url or "",
                    api_key=qdrant_api_key
                ),
                models=AIModuleConfig.ModelConfig(
                    google_api_key=google_api_key or ""
                )
            )
        
        # Initialize services
        self._intent_classifier: Optional[IntentClassifier] = None
        self._model_router: Optional[ModelRouter] = None
        self._cost_tracker: Optional[CostTracker] = None
        self._circuit_retriever: Optional[CircuitRetriever] = None
        self._circuit_builder: Optional[CircuitBuilder] = None
        self._guide_generator: Optional[GuideGenerator] = None
        self._chat_history: Optional[ChatHistoryService] = None
    
    # ==================== Intent Classification ====================
    
    @property
    def intent_classifier(self) -> IntentClassifier:
        """Get the intent classifier"""
        if self._intent_classifier is None:
            self._intent_classifier = get_intent_classifier()
        return self._intent_classifier
    
    def classify_intent(
        self,
        query: str,
        design_id: Optional[str] = None,
        previous_intent: Optional[IntentType] = None
    ) -> IntentClassification:
        """
        Classify a query into an intent.
        
        Args:
            query: User query text
            design_id: Optional design context
            previous_intent: Previous intent for context
        
        Returns:
            IntentClassification with intent, confidence, reasoning
        """
        return self.intent_classifier.classify_with_context(
            query=query,
            design_id=design_id,
            previous_intent=previous_intent
        )
    
    # ==================== Model Routing ====================
    
    @property
    def model_router(self) -> ModelRouter:
        """Get the model router"""
        if self._model_router is None:
            self._model_router = get_model_router()
        return self._model_router
    
    def route_model(self, intent: IntentType) -> RoutingDecision:
        """
        Route an intent to the appropriate model.
        
        Args:
            intent: The classified intent
        
        Returns:
            RoutingDecision with model choice and metadata
        """
        return self.model_router.route(intent)
    
    def get_model_for_query(self, query: str) -> RoutingDecision:
        """
        Convenience method: classify and route in one call.
        
        Args:
            query: User query
        
        Returns:
            RoutingDecision
        """
        intent = self.classify_intent(query)
        return self.route_model(intent.intent)
    
    # ==================== Cost Tracking ====================
    
    @property
    def cost_tracker(self) -> CostTracker:
        """Get the cost tracker"""
        if self._cost_tracker is None:
            self._cost_tracker = get_cost_tracker()
        return self._cost_tracker
    
    def record_usage(
        self,
        model_type: ModelType,
        intent: str,
        token_count: int,
        design_id: Optional[str] = None,
        workspace_id: Optional[str] = None
    ) -> None:
        """Record model usage for cost tracking"""
        self.cost_tracker.record_usage(
            model_type=model_type,
            intent=intent,
            token_count=token_count,
            design_id=design_id,
            workspace_id=workspace_id
        )
    
    # ==================== Circuit Retrieval & Building ====================
    
    @property
    def circuit_retriever(self) -> CircuitRetriever:
        """Get the circuit retriever"""
        if self._circuit_retriever is None:
            self._circuit_retriever = get_circuit_retriever(
                qdrant_url=self.config.qdrant.url,
                api_key=self.config.qdrant.api_key,
                collection_name=self.config.qdrant.collection_name
            )
        return self._circuit_retriever
    
    @property
    def circuit_builder(self) -> CircuitBuilder:
        """Get the circuit builder"""
        if self._circuit_builder is None:
            self._circuit_builder = get_circuit_builder(self.circuit_retriever)
        return self._circuit_builder
    
    def build_circuit(
        self,
        query: str,
        design_id: Optional[str] = None,
        workspace_id: Optional[str] = None
    ) -> Tuple[Optional[Circuit], List[CircuitValidationError]]:
        """
        Build a circuit from a query.
        
        Args:
            query: Circuit creation request
            design_id: Optional design ID for context
            workspace_id: Optional workspace ID
        
        Returns:
            Tuple of (circuit, validation_errors)
        """
        return self.circuit_builder.build(
            query=query,
            design_id=design_id,
            workspace_id=workspace_id
        )
    
    def search_components(
        self,
        query: str,
        component_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Component]:
        """
        Search for circuit components in Qdrant.
        
        Args:
            query: Search query
            component_type: Filter by type
            limit: Max results
        
        Returns:
            List of Component objects
        """
        return self.circuit_retriever.search_components(
            query=query,
            component_type=component_type,
            limit=limit
        )
    
    # ==================== Guide Generation ====================
    
    @property
    def guide_generator(self) -> GuideGenerator:
        """Get the guide generator"""
        if self._guide_generator is None:
            self._guide_generator = get_guide_generator()
        return self._guide_generator
    
    def generate_guide(self, circuit: Circuit) -> str:
        """
        Generate a markdown guide for a circuit.
        
        Args:
            circuit: The Circuit object
        
        Returns:
            Markdown string with step-by-step guide
        """
        return self.guide_generator.generate(circuit)
    
    # ==================== Chat History ====================
    
    @property
    def chat_history(self) -> ChatHistoryService:
        """Get the chat history service"""
        if self._chat_history is None:
            self._chat_history = get_chat_history_service(self.config.database.url)
        return self._chat_history
    
    def add_chat_message(
        self,
        workspace_id: UUID,
        design_id: UUID,
        role: str,
        content: str,
        intent: Optional[str] = None,
        model_used: Optional[str] = None,
        circuit_json: Optional[Dict[str, Any]] = None,
        guide_markdown: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add a message to a design's chat history.
        
        Args:
            workspace_id: Workspace UUID
            design_id: Design UUID
            role: 'user', 'assistant', or 'system'
            content: Message text
            intent: Classified intent
            model_used: Model that generated response
            circuit_json: Circuit JSON (for design messages)
            guide_markdown: Markdown guide (for design messages)
        
        Returns:
            The created message as dict
        """
        return self.chat_history.add_message(
            workspace_id=workspace_id,
            design_id=design_id,
            role=role,
            content=content,
            intent=intent,
            model_used=model_used,
            circuit_json=circuit_json,
            guide_markdown=guide_markdown
        )
    
    def get_chat_messages(
        self,
        design_id: UUID,
        page: int = 1,
        limit: int = 25
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        Get paginated messages for a design's chat.
        
        Args:
            design_id: Design UUID
            page: Page number (1-based)
            limit: Messages per page
        
        Returns:
            Tuple of (messages, total_messages, total_pages)
        """
        return self.chat_history.get_messages(
            design_id=design_id,
            page=page,
            limit=limit
        )
    
    def create_workspace(self, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new workspace"""
        return self.chat_history.create_workspace(name, description)
    
    def create_design(
        self,
        workspace_id: UUID,
        name: str,
        description: str = ""
    ) -> Dict[str, Any]:
        """Create a new design within a workspace"""
        return self.chat_history.create_design(workspace_id, name, description)
    
    # ==================== Full Chat Flow ====================
    
    def process_chat_message(
        self,
        workspace_id: UUID,
        design_id: UUID,
        user_query: str
    ) -> Dict[str, Any]:
        """
        Process a complete chat message through the full pipeline.
        
        This is the main method that:
        1. Classifies intent
        2. Routes to appropriate model
        3. Builds circuit if needed
        4. Generates guide if needed
        5. Stores in chat history
        6. Tracks costs
        
        Args:
            workspace_id: Workspace UUID
            design_id: Design UUID
            user_query: User's message
        
        Returns:
            Response with intent, model, circuit, guide, etc.
        """
        # Step 1: Classify intent
        intent_result = self.classify_intent(
            query=user_query,
            design_id=str(design_id)
        )
        
        # Step 2: Route to model
        routing = self.route_model(intent_result.intent)
        
        # Step 3: Build circuit if design-related intent
        circuit = None
        guide = None
        circuit_json = None
        guide_markdown = None
        
        if intent_result.intent in ["edit_schematic", "create_design", "debug_circuit"]:
            circuit, errors = self.build_circuit(
                query=user_query,
                design_id=str(design_id),
                workspace_id=str(workspace_id)
            )
            
            if circuit:
                guide = self.generate_guide(circuit)
                circuit_json = circuit.model_dump()
                guide_markdown = guide
        
        # Step 4: Store user message in history
        user_msg = self.add_chat_message(
            workspace_id=workspace_id,
            design_id=design_id,
            role="user",
            content=user_query,
            intent=intent_result.intent,
            model_used=None
        )
        
        # Step 5: Generate assistant response
        # (In production, this would call the actual LLM)
        assistant_response = self._generate_assistant_response(
            user_query=user_query,
            intent=intent_result.intent,
            circuit=circuit,
            guide=guide
        )
        
        # Step 6: Store assistant message
        assistant_msg = self.add_chat_message(
            workspace_id=workspace_id,
            design_id=design_id,
            role="assistant",
            content=assistant_response,
            intent=intent_result.intent,
            model_used=routing.model_id,
            circuit_json=circuit_json,
            guide_markdown=guide_markdown
        )
        
        # Step 7: Track cost (estimate based on query length)
        token_count = len(user_query.split()) * 4  # Rough estimate
        self.record_usage(
            model_type=routing.model,
            intent=intent_result.intent,
            token_count=token_count,
            design_id=str(design_id),
            workspace_id=str(workspace_id)
        )
        
        return {
            "intent": intent_result.intent,
            "confidence": intent_result.confidence,
            "model": routing.model_id,
            "model_type": routing.model.value,
            "cost_estimate": routing.estimated_cost,
            "circuit": circuit_json,
            "guide": guide_markdown,
            "response": assistant_response,
            "user_message_id": user_msg["id"],
            "assistant_message_id": assistant_msg["id"]
        }
    
    def _generate_assistant_response(
        self,
        user_query: str,
        intent: IntentType,
        circuit: Optional[Circuit] = None,
        guide: Optional[str] = None
    ) -> str:
        """Generate a placeholder assistant response"""
        if intent == "general_query":
            return f"I understand you're asking about: {user_query}"
        elif intent == "create_design":
            if circuit:
                return f"I've created a circuit design for you. Here's the JSON: {circuit.model_dump_json()}"
            return "I'll help you create that design. Let me fetch the necessary components."
        elif intent == "edit_schematic":
            return "I'll help you modify the schematic. Please provide more details about what you want to change."
        elif intent == "debug_circuit":
            return "Let's debug this circuit. Please describe the issue you're experiencing."
        else:
            return f"I understand you want to: {user_query}"
    
    # ==================== Cleanup ====================
    
    def close(self) -> None:
        """Close all connections and clean up resources"""
        if self._chat_history:
            self._chat_history.close()
        # Reset singletons
        from .intent.classifier import reset_intent_classifier
        from .routing.model_router import reset_model_router
        from .routing.cost_tracker import reset_cost_tracker
        from .rag.circuit_retriever import reset_circuit_retriever
        from .circuit.builder import reset_circuit_builder
        from .guide.generator import reset_guide_generator
        from .chat.history import reset_chat_history_service
        
        reset_intent_classifier()
        reset_model_router()
        reset_cost_tracker()
        reset_circuit_retriever()
        reset_circuit_builder()
        reset_guide_generator()
        reset_chat_history_service()


# Singleton instance
_module: Optional[AIModule] = None


def get_ai_module(
    db_url: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None
) -> AIModule:
    """Get or create the singleton AI module"""
    global _module
    if _module is None:
        _module = AIModule(
            db_url=db_url,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            google_api_key=google_api_key
        )
    return _module


def reset_ai_module() -> None:
    """Reset the singleton"""
    global _module
    if _module:
        _module.close()
        _module = None
