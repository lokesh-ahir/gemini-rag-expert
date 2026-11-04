from typing import Dict, List
from src.interfaces.tool import ToolInterface


class ToolRegistry:
    """Registry for agent tools"""

    _tools: Dict[str, ToolInterface] = {}

    @classmethod
    def register(cls, tool: ToolInterface):
        """Register a tool"""
        cls._tools[tool.name] = tool

    @classmethod
    def get_tool(cls, name: str) -> ToolInterface:
        """Get a tool by name"""
        return cls._tools.get(name)

    @classmethod
    def get_all_tools(cls) -> List[ToolInterface]:
        """Get all registered tools"""
        return list(cls._tools.values())

    @classmethod
    def clear(cls):
        """Clear all registered tools"""
        cls._tools.clear()
