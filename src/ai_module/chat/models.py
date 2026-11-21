"""
Database Models for Chat History
Uses SQLAlchemy ORM for PostgreSQL
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4
from sqlalchemy import create_engine, Column, String, Text, JSON, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

Base = declarative_base()


class Workspace(Base):
    """Represents a workspace containing multiple designs"""
    __tablename__ = "workspaces"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    designs = relationship("Design", back_populates="workspace", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class Design(Base):
    """Represents a design within a workspace. Each design has ONE chat."""
    __tablename__ = "designs"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    workspace_id = Column(PG_UUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    status = Column(String(50), default="active")
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    workspace = relationship("Workspace", back_populates="designs")
    messages = relationship("Message", back_populates="design", cascade="all, delete-orphan", 
                           order_by="asc(Message.created_at)")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "workspace_id": str(self.workspace_id),
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class Message(Base):
    """Represents a single message in a design's chat"""
    __tablename__ = "messages"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    design_id = Column(PG_UUID(as_uuid=True), ForeignKey("designs.id"), nullable=False)
    workspace_id = Column(PG_UUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=False)
    
    # Message content
    role = Column(String(20), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    
    # AI metadata
    intent = Column(String(50))  # general_query, edit_schematic, create_design, debug_circuit
    model_used = Column(String(100))  # gemini-flash, gemini-neon-4, etc.
    
    # Circuit-specific fields (NULL for general queries)
    circuit_json = Column(JSON)
    guide_markdown = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    design = relationship("Design", back_populates="messages")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "design_id": str(self.design_id),
            "workspace_id": str(self.workspace_id),
            "role": self.role,
            "content": self.content,
            "intent": self.intent,
            "model_used": self.model_used,
            "circuit_json": self.circuit_json,
            "guide_markdown": self.guide_markdown,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


# Intent types
INTENT_TYPES = [
    "general_query",
    "edit_schematic", 
    "create_design",
    "debug_circuit"
]

# Model types
MODEL_TYPES = [
    "gemini-flash",
    "gemini-neon-4"
]
