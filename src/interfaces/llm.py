from abc import ABC, abstractmethod
from typing import Any


class LLMInterface(ABC):
    """Abstract interface for LLM providers"""

    @abstractmethod
    def invoke(self, prompt: str, **kwargs) -> str:
        """Invoke the LLM with a prompt"""
        pass

    @abstractmethod
    def stream(self, prompt: str, **kwargs):
        """Stream responses from the LLM"""
        pass
