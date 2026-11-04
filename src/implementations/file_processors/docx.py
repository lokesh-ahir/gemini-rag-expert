from typing import List
from docx import Document
import io
from src.interfaces.file_processor import FileProcessorInterface


class DOCXProcessor(FileProcessorInterface):
    """DOCX file processor implementation"""

    @property
    def supported_extensions(self) -> List[str]:
        return [".docx"]

    def extract_text(self, file_content: bytes, filename: str, **kwargs) -> str:
        doc = Document(io.BytesIO(file_content))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
