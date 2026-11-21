"""
FastAPI Router for AI Module Endpoints.

Simplified to only 2 endpoints:
- POST /ai/chat - Send a chat message and get response
- GET /ai/chat/{design_id} - Retrieve chat messages for a design

All AI functionality (intent classification, model routing, tool calling,
circuit building, guide generation) happens inside the module.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field

from ..main import AIModule, get_ai_module, reset_ai_module


router = APIRouter(prefix="/ai", tags=["AI Module"])


# ==================== Request/Response Models ====================

class ChatRequest(BaseModel):
    """Request for sending a chat message"""
    workspace_id: str = Field(..., description="Workspace UUID")
    design_id: str = Field(..., description="Design UUID")
    message: str = Field(..., description="User message")


class ChatResponse(BaseModel):
    """Response for chat message"""
    intent: str
    model: str
    response: str
    circuit: Optional[Dict[str, Any]] = Field(default=None, description="Generated circuit JSON")
    guide: Optional[str] = Field(default=None, description="Generated markdown guide")
    message_id: str
    success: bool = True


class MessagesResponse(BaseModel):
    """Response for retrieving chat messages"""
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = 0
    total_pages: int = 0
    page: int = 1
    success: bool = True


# ==================== Dependency ====================

def get_module() -> AIModule:
    """Get the AI module instance"""
    try:
        return get_ai_module()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Module not initialized: {str(e)}")


# ==================== Endpoints ====================

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    module: AIModule = Depends(get_module)
) -> ChatResponse:
    """
    Send a chat message and get a response.
    
    This endpoint handles the complete AI pipeline:
    1. Intent classification (general_query, edit_schematic, create_design, debug_circuit)
    2. Model routing (Gemini Flash for general, Gemini Neon 4 for design tasks)
    3. Tool calling (search components, build circuits, generate guides)
    4. Store messages in PostgreSQL
    5. Return response with circuit/guide if applicable
    
    Request body:
    {
        "workspace_id": "uuid",
        "design_id": "uuid", 
        "message": "Create an amplifier with LM386"
    }
    """
    try:
        workspace_id = UUID(request.workspace_id)
        design_id = UUID(request.design_id)
        
        # Use the agent with tool calling
        result = module.chat_with_agent(
            workspace_id=workspace_id,
            design_id=design_id,
            query=request.message,
            use_tools=True
        )
        
        return ChatResponse(
            intent=result["intent"],
            model=result["model"],
            response=result["response"],
            circuit=result.get("circuit"),
            guide=result.get("guide"),
            message_id=result["assistant_message_id"],
            success=True
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/chat/{design_id}", response_model=MessagesResponse)
async def get_chat_messages(
    design_id: str,
    page: int = Query(default=1, ge=1, description="Page number"),
    limit: int = Query(default=25, ge=1, le=100, description="Messages per page"),
    module: AIModule = Depends(get_module)
) -> MessagesResponse:
    """
    Retrieve chat messages for a design.
    
    Returns paginated messages (25 per page by default).
    Each design has exactly one chat with multiple messages.
    
    Query params:
    - design_id: Design UUID
    - page: Page number (default: 1)
    - limit: Messages per page (default: 25)
    """
    try:
        design_uuid = UUID(design_id)
        
        messages, total, total_pages = module.get_chat_messages(
            design_id=design_uuid,
            page=page,
            limit=limit
        )
        
        return MessagesResponse(
            messages=messages,
            total=total,
            total_pages=total_pages,
            page=page,
            success=True
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
