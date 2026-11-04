from abc import ABC, abstractmethod
from typing import List


class FileProcessorInterface(ABC):
    """Abstract interface for file processors"""

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """List of supported file extensions"""
        pass

    @abstractmethod
    def extract_text(self, file_content: bytes, filename: str, **kwargs) -> str:
        """Extract text from file"""
        pass
