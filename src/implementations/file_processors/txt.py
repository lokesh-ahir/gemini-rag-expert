from typing import List
from src.interfaces.file_processor import FileProcessorInterface


class TXTProcessor(FileProcessorInterface):
    """TXT file processor implementation"""

    @property
    def supported_extensions(self) -> List[str]:
        return [".txt"]

    def extract_text(self, file_content: bytes, filename: str, **kwargs) -> str:
        return file_content.decode("utf-8")
