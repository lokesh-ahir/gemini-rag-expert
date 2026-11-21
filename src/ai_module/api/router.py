"""
FastAPI Router for AI Module Endpoints.

Provides REST API endpoints for:
- Intent classification
- Model routing
- Circuit building
- Guide generation
- Chat history
- Agent conversations
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field

from ..main import AIModule, get_ai_module, reset_ai_module
from ..intent.schemas import IntentType
from ..circuit.schemas import Circuit


router = APIRouter(prefix="/ai", tags=["AI Module"])


# ==================== Request/Response Models ====================

class IntentRequest(BaseModel):
    """Request for intent classification"""
    query: str = Field(..., description="User query to classify")
    design_id: Optional[str] = Field(default=None, description="Design ID for context")
    previous_intent: Optional[IntentType] = Field(default=None, description="Previous intent")


class IntentResponse(BaseModel):
    """Response for intent classification"""
    intent: IntentType
    confidence: float
    reasoning: Optional[str] = None


class ModelRoutingRequest(BaseModel):
    """Request for model routing"""
    intent: IntentType = Field(..., description="Intent to route")


class ModelRoutingResponse(BaseModel):
    """Response for model routing"""
    model: str
    model_type: str
    intent: IntentType
    reason: str
    estimated_cost: float


class CircuitRequest(BaseModel):
    """Request for circuit building"""
    query: str = Field(..., description="Circuit creation request")
    design_id: Optional[str] = Field(default=None, description="Design ID for context")
    workspace_id: Optional[str] = Field(default=None, description="Workspace ID for context")
    constraints: Optional[Dict[str, Any]] = Field(default=None, description="Design constraints")


class CircuitResponse(BaseModel):
    """Response for circuit building"""
    circuit: Optional[Dict[str, Any]] = Field(default=None, description="Generated circuit JSON")
    validation_errors: List[Dict[str, Any]] = Field(default_factory=list, description="Validation errors")
    success: bool = True


class GuideRequest(BaseModel):
    """Request for guide generation"""
    circuit: Dict[str, Any] = Field(..., description="Circuit JSON")


class GuideResponse(BaseModel):
    """Response for guide generation"""
    guide_markdown: str
    success: bool = True


class ChatRequest(BaseModel):
    """Request for chat message"""
    workspace_id: str = Field(..., description="Workspace UUID")
    design_id: str = Field(..., description="Design UUID")
    query: str = Field(..., description="User message")
    use_agent: bool = Field(default=True, description="Use LLM agent with tool calling")


class ChatResponse(BaseModel):
    """Response for chat message"""
    intent: IntentType
    confidence: float
    model: str
    model_type: str
    response: str
    circuit: Optional[Dict[str, Any]] = None
    guide: Optional[str] = None
    tool_calls: List[Any] = Field(default_factory=list)
    user_message_id: str
    assistant_message_id: str
    success: bool = True


class WorkspaceRequest(BaseModel):
    """Request for creating a workspace"""
    name: str = Field(..., description="Workspace name")
    description: Optional[str] = Field(default=None, description="Workspace description")


class DesignRequest(BaseModel):
    """Request for creating a design"""
    workspace_id: str = Field(..., description="Workspace UUID")
    name: str = Field(..., description="Design name")
    description: Optional[str] = Field(default=None, description="Design description")


class MessagesRequest(BaseModel):
    """Request for getting messages"""
    design_id: str = Field(..., description="Design UUID")
    page: int = Field(default=1, ge=1, description="Page number")
    limit: int = Field(default=25, ge=1, le=100, description="Messages per page")


class MessagesResponse(BaseModel):
    """Response for getting messages"""
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = 0
    total_pages: int = 0
    success: bool = True


# ==================== Dependency ====================

def get_module() -> AIModule:
    """Get the AI module instance"""
    try:
        return get_ai_module()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Module not initialized: {str(e)}")


# ==================== Endpoints ====================

@router.post("/intent/classify", response_model=IntentResponse)
async def classify_intent(
    request: IntentRequest,
    module: AIModule = Depends(get_module)
) -> IntentResponse:
    """
    Classify a query into an intent.
    
    Intents: general_query, edit_schematic, create_design, debug_circuit
    """
    try:
        result = module.classify_intent(
            query=request.query,
            design_id=request.design_id,
            previous_intent=request.previous_intent
        )
        return IntentResponse(
            intent=result.intent,
            confidence=result.confidence,
            reasoning=result.reasoning
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/model/route", response_model=ModelRoutingResponse)
async def route_model(
    request: ModelRoutingRequest,
    module: AIModule = Depends(get_module)
) -> ModelRoutingResponse:
    """
    Route an intent to the appropriate model.
    
    Models: gemini-flash (for general), gemini-neon-4 (for design tasks)
    """
    try:
        result = module.route_model(request.intent)
        return ModelRoutingResponse(
            model=result.model_id,
            model_type=result.model.value,
            intent=request.intent,
            reason=result.reason,
            estimated_cost=result.estimated_cost
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/circuit/build", response_model=CircuitResponse)
async def build_circuit(
    request: CircuitRequest,
    module: AIModule = Depends(get_module)
) -> CircuitResponse:
    """
    Build a circuit from a query.
    
    Returns circuit JSON with components and connections.
    """
    try:
        circuit, errors = module.build_circuit(
            query=request.query,
            design_id=request.design_id,
            workspace_id=request.workspace_id,
            constraints=request.constraints
        )
        return CircuitResponse(
            circuit=circuit.model_dump() if circuit else None,
            validation_errors=[e.model_dump() for e in errors],
            success=circuit is not None
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/guide/generate", response_model=GuideResponse)
async def generate_guide(
    request: GuideRequest,
    module: AIModule = Depends(get_module)
) -> GuideResponse:
    """
    Generate a step-by-step markdown guide for a circuit.
    """
    try:
        circuit = Circuit(**request.circuit)
        guide = module.generate_guide(circuit)
        return GuideResponse(guide_markdown=guide, success=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    module: AIModule = Depends(get_module)
) -> ChatResponse:
    """
    Send a chat message and get a response.
    
    Uses LLM agent with tool calling if use_agent=True.
    Otherwise uses the basic pipeline.
    """
    try:
        workspace_id = UUID(request.workspace_id)
        design_id = UUID(request.design_id)
        
        if request.use_agent:
            result = module.chat_with_agent(
                workspace_id=workspace_id,
                design_id=design_id,
                query=request.query,
                use_tools=True
            )
        else:
            result = module.process_chat_message(
                workspace_id=workspace_id,
                design_id=design_id,
                user_query=request.query
            )
        
        return ChatResponse(
            intent=result["intent"],
            confidence=result["confidence"],
            model=result["model"],
            model_type=result["model_type"],
            response=result["response"],
            circuit=result.get("circuit"),
            guide=result.get("guide"),
            tool_calls=result.get("tool_calls", []),
            user_message_id=result["user_message_id"],
            assistant_message_id=result["assistant_message_id"],
            success=True
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/workspace/create", response_model=Dict[str, Any])
async def create_workspace(
    request: WorkspaceRequest,
    module: AIModule = Depends(get_module)
) -> Dict[str, Any]:
    """Create a new workspace"""
    try:
        return module.create_workspace(
            name=request.name,
            description=request.description or ""
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/design/create", response_model=Dict[str, Any])
async def create_design(
    request: DesignRequest,
    module: AIModule = Depends(get_module)
) -> Dict[str, Any]:
    """Create a new design within a workspace"""
    try:
        workspace_id = UUID(request.workspace_id)
        return module.create_design(
            workspace_id=workspace_id,
            name=request.name,
            description=request.description or ""
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/messages/get", response_model=MessagesResponse)
async def get_messages(
    request: MessagesRequest,
    module: AIModule = Depends(get_module)
) -> MessagesResponse:
    """Get paginated messages for a design's chat"""
    try:
        design_id = UUID(request.design_id)
        messages, total, total_pages = module.get_chat_messages(
            design_id=design_id,
            page=request.page,
            limit=request.limit
        )
        return MessagesResponse(
            messages=messages,
            total=total,
            total_pages=total_pages,
            success=True
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Check if the AI module is healthy"""
    return {
        "status": "healthy",
        "module": "ai_module",
        "version": "1.0.0"
    }
