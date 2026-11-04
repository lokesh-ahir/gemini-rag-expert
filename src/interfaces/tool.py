from abc import ABC, abstractmethod
from typing import Any


class ToolInterface(ABC):
    """Abstract interface for agent tools"""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def execute(self, query: str, **kwargs) -> Any:
        """Execute the tool"""
        pass
