"""
Chat Tools - Tools for managing chat history.
"""

from typing import Optional, List, Dict, Any
from uuid import UUID
from pydantic import Field

from .base import AITool, ToolInput, ToolOutput


# ==================== Input Schemas ====================

class AddMessageInput(ToolInput):
    """Input for adding a chat message"""
    workspace_id: str = Field(..., description="Workspace UUID")
    design_id: str = Field(..., description="Design UUID")
    role: str = Field(..., description="Message role (user/assistant/system)")
    content: str = Field(..., description="Message content")
    intent: Optional[str] = Field(default=None, description="Classified intent")
    model_used: Optional[str] = Field(default=None, description="Model used")
    circuit_json: Optional[Dict[str, Any]] = Field(default=None, description="Circuit JSON")
    guide_markdown: Optional[str] = Field(default=None, description="Markdown guide")


class GetMessagesInput(ToolInput):
    """Input for getting chat messages"""
    design_id: str = Field(..., description="Design UUID")
    page: int = Field(default=1, ge=1, description="Page number")
    limit: int = Field(default=25, ge=1, le=100, description="Messages per page")


class CreateWorkspaceInput(ToolInput):
    """Input for creating a workspace"""
    name: str = Field(..., description="Workspace name")
    description: Optional[str] = Field(default=None, description="Workspace description")


class CreateDesignInput(ToolInput):
    """Input for creating a design"""
    workspace_id: str = Field(..., description="Workspace UUID")
    name: str = Field(..., description="Design name")
    description: Optional[str] = Field(default=None, description="Design description")


# ==================== Output Schemas ====================

class AddMessageOutput(ToolOutput):
    """Output for adding a message"""
    message: Optional[Dict[str, Any]] = Field(default=None, description="Created message")


class GetMessagesOutput(ToolOutput):
    """Output for getting messages"""
    messages: List[Dict[str, Any]] = Field(default_factory=list, description="List of messages")
    total: int = Field(default=0, description="Total messages")
    total_pages: int = Field(default=0, description="Total pages")


class CreateWorkspaceOutput(ToolOutput):
    """Output for creating a workspace"""
    workspace: Optional[Dict[str, Any]] = Field(default=None, description="Created workspace")


class CreateDesignOutput(ToolOutput):
    """Output for creating a design"""
    design: Optional[Dict[str, Any]] = Field(default=None, description="Created design")


# ==================== Tool Implementations ====================

class AddMessageTool(AITool):
    """Add a message to a design's chat history"""
    
    name = "add_chat_message"
    description = """
    Add a message to a design's chat history in PostgreSQL.
    Use this to store user messages, assistant responses, and metadata.
    """
    input_schema = AddMessageInput
    output_schema = AddMessageOutput
    
    def __init__(self, chat_service):
        self.chat_service = chat_service
    
    def execute(self, **kwargs) -> AddMessageOutput:
        """Execute message addition"""
        try:
            input_data = AddMessageInput(**kwargs)
            
            message = self.chat_service.add_message(
                workspace_id=UUID(input_data.workspace_id),
                design_id=UUID(input_data.design_id),
                role=input_data.role,
                content=input_data.content,
                intent=input_data.intent,
                model_used=input_data.model_used,
                circuit_json=input_data.circuit_json,
                guide_markdown=input_data.guide_markdown
            )
            
            return AddMessageOutput(
                success=True,
                message="Message added to chat history",
                data={"message": message}
            )
        except Exception as e:
            return AddMessageOutput(
                success=False,
                message="Error adding message",
                error=str(e)
            )


class GetMessagesTool(AITool):
    """Get chat messages for a design"""
    
    name = "get_chat_messages"
    description = """
    Get paginated chat messages for a design.
    Use this to retrieve conversation history.
    """
    input_schema = GetMessagesInput
    output_schema = GetMessagesOutput
    
    def __init__(self, chat_service):
        self.chat_service = chat_service
    
    def execute(self, **kwargs) -> GetMessagesOutput:
        """Execute message retrieval"""
        try:
            input_data = GetMessagesInput(**kwargs)
            
            messages, total, total_pages = self.chat_service.get_messages(
                design_id=UUID(input_data.design_id),
                page=input_data.page,
                limit=input_data.limit
            )
            
            return GetMessagesOutput(
                success=True,
                message=f"Retrieved {len(messages)} messages",
                data={
                    "messages": messages,
                    "total": total,
                    "total_pages": total_pages
                }
            )
        except Exception as e:
            return GetMessagesOutput(
                success=False,
                message="Error getting messages",
                error=str(e)
            )


class CreateWorkspaceTool(AITool):
    """Create a new workspace"""
    
    name = "create_workspace"
    description = """
    Create a new workspace for organizing designs.
    """
    input_schema = CreateWorkspaceInput
    output_schema = CreateWorkspaceOutput
    
    def __init__(self, chat_service):
        self.chat_service = chat_service
    
    def execute(self, **kwargs) -> CreateWorkspaceOutput:
        """Execute workspace creation"""
        try:
            input_data = CreateWorkspaceInput(**kwargs)
            workspace = self.chat_service.create_workspace(
                name=input_data.name,
                description=input_data.description or ""
            )
            
            return CreateWorkspaceOutput(
                success=True,
                message="Workspace created",
                data={"workspace": workspace}
            )
        except Exception as e:
            return CreateWorkspaceOutput(
                success=False,
                message="Error creating workspace",
                error=str(e)
            )


class CreateDesignTool(AITool):
    """Create a new design"""
    
    name = "create_design"
    description = """
    Create a new design within a workspace.
    Each design has its own chat.
    """
    input_schema = CreateDesignInput
    output_schema = CreateDesignOutput
    
    def __init__(self, chat_service):
        self.chat_service = chat_service
    
    def execute(self, **kwargs) -> CreateDesignOutput:
        """Execute design creation"""
        try:
            input_data = CreateDesignInput(**kwargs)
            design = self.chat_service.create_design(
                workspace_id=UUID(input_data.workspace_id),
                name=input_data.name,
                description=input_data.description or ""
            )
            
            return CreateDesignOutput(
                success=True,
                message="Design created",
                data={"design": design}
            )
        except Exception as e:
            return CreateDesignOutput(
                success=False,
                message="Error creating design",
                error=str(e)
            )
