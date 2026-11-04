from abc import ABC, abstractmethod
from typing import List


class EmbeddingInterface(ABC):
    """Abstract interface for embedding models"""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        pass
