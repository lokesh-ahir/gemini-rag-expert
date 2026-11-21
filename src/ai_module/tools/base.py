"""
Base tool classes for LangChain/LangGraph integration.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field


class ToolInput(BaseModel):
    """Base model for tool inputs"""
    pass


class ToolOutput(BaseModel):
    """Base model for tool outputs"""
    success: bool = True
    message: str = ""
    data: Optional[Any] = None
    error: Optional[str] = None


class AITool(ABC):
    """
    Base class for AI Module tools.
    Compatible with LangChain and LangGraph.
    """
    
    name: str = ""
    description: str = ""
    input_schema: type = ToolInput
    output_schema: type = ToolOutput
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolOutput:
        """Execute the tool with given arguments"""
        pass
    
    def __call__(self, **kwargs) -> ToolOutput:
        """Make the tool callable"""
        return self.execute(**kwargs)
    
    def to_langchain_tool(self):
        """Convert to LangChain tool format"""
        from langchain_core.tools import tool
        
        @tool
        def _tool(**kwargs):
            return self.execute(**kwargs)
        
        _tool.name = self.name
        _tool.description = self.description
        return _tool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema.model_json_schema(),
            "output_schema": self.output_schema.model_json_schema()
        }
