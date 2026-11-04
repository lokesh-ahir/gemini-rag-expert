from typing import Dict, List, Type
from src.interfaces.file_processor import FileProcessorInterface
from src.implementations.file_processors.pdf import PDFProcessor
from src.implementations.file_processors.docx import DOCXProcessor
from src.implementations.file_processors.txt import TXTProcessor


class FileProcessorRegistry:
    """Registry for file processors"""

    _processors: Dict[str, FileProcessorInterface] = {}

    @classmethod
    def register(cls, processor: FileProcessorInterface):
        """Register a file processor"""
        for ext in processor.supported_extensions:
            cls._processors[ext.lower()] = processor

    @classmethod
    def get_processor(cls, filename: str) -> FileProcessorInterface:
        """Get processor for a given filename"""
        for ext, processor in cls._processors.items():
            if filename.lower().endswith(ext):
                return processor
        raise ValueError(f"No processor found for file: {filename}")

    @classmethod
    def load_default_processors(cls):
        """Load default processors"""
        cls.register(PDFProcessor())
        cls.register(DOCXProcessor())
        cls.register(TXTProcessor())

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get all supported file extensions"""
        return list(cls._processors.keys())


# Initialize with default processors
FileProcessorRegistry.load_default_processors()
