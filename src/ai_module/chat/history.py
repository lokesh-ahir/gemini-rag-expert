"""
Chat History Service - Manages PostgreSQL storage for design chats.
Each design has exactly ONE chat with multiple messages.
"""

from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID
from datetime import datetime
from sqlalchemy import create_engine, select, desc, asc
from sqlalchemy.orm import sessionmaker

from .models import Base, Workspace, Design, Message, INTENT_TYPES, MODEL_TYPES


class ChatHistoryService:
    """
    Service for managing chat history in PostgreSQL.
    
    Usage:
        # Initialize
        service = ChatHistoryService("postgresql://user:pass@localhost/db")
        service.initialize()
        
        # Add message to a design's chat
        service.add_message(
            workspace_id="...",
            design_id="...",
            role="user",
            content="Create a new amplifier circuit"
        )
        
        # Get messages for a design (paginated)
        messages = service.get_messages(design_id="...", page=1, limit=25)
    """
    
    def __init__(self, database_url: str):
        """
        Initialize the chat history service.
        
        Args:
            database_url: PostgreSQL connection URL
        """
        self.database_url = database_url
        self.engine = create_engine(database_url, pool_pre_ping=True)
        self.Session = sessionmaker(bind=self.engine)
    
    def initialize(self) -> None:
        """Create all tables if they don't exist"""
        Base.metadata.create_all(self.engine)
    
    def create_workspace(self, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new workspace"""
        with self.Session() as session:
            workspace = Workspace(name=name, description=description)
            session.add(workspace)
            session.commit()
            session.refresh(workspace)
            return workspace.to_dict()
    
    def get_workspace(self, workspace_id: UUID) -> Optional[Dict[str, Any]]:
        """Get workspace by ID"""
        with self.Session() as session:
            workspace = session.get(Workspace, workspace_id)
            return workspace.to_dict() if workspace else None
    
    def create_design(self, workspace_id: UUID, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new design within a workspace"""
        with self.Session() as session:
            design = Design(
                workspace_id=workspace_id,
                name=name,
                description=description
            )
            session.add(design)
            session.commit()
            session.refresh(design)
            return design.to_dict()
    
    def get_design(self, design_id: UUID) -> Optional[Dict[str, Any]]:
        """Get design by ID"""
        with self.Session() as session:
            design = session.get(Design, design_id)
            return design.to_dict() if design else None
    
    def add_message(
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
        Add a message to a design's chat.
        
        Args:
            workspace_id: UUID of the workspace
            design_id: UUID of the design
            role: 'user', 'assistant', or 'system'
            content: Message text content
            intent: Classified intent (general_query, edit_schematic, etc.)
            model_used: Model that generated the response
            circuit_json: Circuit JSON (for design-related messages)
            guide_markdown: Markdown guide (for design-related messages)
        
        Returns:
            The created message as a dictionary
        """
        with self.Session() as session:
            message = Message(
                design_id=design_id,
                workspace_id=workspace_id,
                role=role,
                content=content,
                intent=intent,
                model_used=model_used,
                circuit_json=circuit_json,
                guide_markdown=guide_markdown
            )
            session.add(message)
            session.commit()
            session.refresh(message)
            return message.to_dict()
    
    def get_messages(
        self,
        design_id: UUID,
        page: int = 1,
        limit: int = 25
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        Get paginated messages for a design's chat.
        
        Args:
            design_id: UUID of the design
            page: Page number (1-based)
            limit: Messages per page (default: 25)
        
        Returns:
            Tuple of (messages, total_messages, total_pages)
        """
        with self.Session() as session:
            # Count total messages
            count_stmt = select(Message).where(Message.design_id == design_id)
            total = session.scalar(select([count_stmt]).subquery()) or 0
            total_pages = (total + limit - 1) // limit
            
            # Get paginated messages
            offset = (page - 1) * limit
            stmt = (
                select(Message)
                .where(Message.design_id == design_id)
                .order_by(asc(Message.created_at))
                .offset(offset)
                .limit(limit)
            )
            messages = session.execute(stmt).scalars().all()
            
            return [m.to_dict() for m in messages], total, total_pages
    
    def get_design_chat(
        self,
        design_id: UUID
    ) -> List[Dict[str, Any]]:
        """
        Get ALL messages for a design's chat (no pagination).
        Useful for loading entire chat history.
        """
        with self.Session() as session:
            stmt = (
                select(Message)
                .where(Message.design_id == design_id)
                .order_by(asc(Message.created_at))
            )
            messages = session.execute(stmt).scalars().all()
            return [m.to_dict() for m in messages]
    
    def get_recent_chats(
        self,
        workspace_id: UUID,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent designs with their latest message for a workspace.
        
        Returns:
            List of design info with latest message preview
        """
        with self.Session() as session:
            # Get designs in workspace
            designs = session.query(Design).filter(
                Design.workspace_id == workspace_id
            ).order_by(desc(Design.updated_at)).limit(limit).all()
            
            result = []
            for design in designs:
                # Get latest message
                latest_msg = session.query(Message).filter(
                    Message.design_id == design.id
                ).order_by(desc(Message.created_at)).first()
                
                result.append({
                    **design.to_dict(),
                    "latest_message": latest_msg.to_dict() if latest_msg else None
                })
            
            return result
    
    def delete_design_chat(self, design_id: UUID) -> bool:
        """Delete all messages for a design (reset chat)"""
        with self.Session() as session:
            stmt = select(Message).where(Message.design_id == design_id)
            messages = session.execute(stmt).scalars().all()
            
            for msg in messages:
                session.delete(msg)
            
            session.commit()
            return True
    
    def close(self) -> None:
        """Close database connections"""
        self.engine.dispose()


# Singleton instance (can be configured at startup)
_chat_history_service: Optional[ChatHistoryService] = None


def get_chat_history_service(database_url: Optional[str] = None) -> ChatHistoryService:
    """Get or create the singleton chat history service"""
    global _chat_history_service
    
    if _chat_history_service is None:
        if database_url is None:
            raise ValueError("Database URL must be provided on first call")
        _chat_history_service = ChatHistoryService(database_url)
        _chat_history_service.initialize()
    
    return _chat_history_service


def reset_chat_history_service() -> None:
    """Reset the singleton (useful for testing)"""
    global _chat_history_service
    if _chat_history_service:
        _chat_history_service.close()
        _chat_history_service = None
