from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain_core.documents import Document


class VectorStoreInterface(ABC):
    """Abstract interface for all vector store implementations"""

    @abstractmethod
    def ensure_collection(self, collection_name: str, **kwargs) -> None:
        """Ensure a collection exists"""
        pass

    @abstractmethod
    def add_documents(self, collection_name: str, texts: List[str], metadatas: List[Dict]) -> None:
        """Add documents to the vector store"""
        pass

    @abstractmethod
    def similarity_search(self, collection_name: str, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents"""
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection"""
        pass
