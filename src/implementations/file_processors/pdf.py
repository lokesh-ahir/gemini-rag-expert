from typing import List
import PyPDF2
import io
from src.interfaces.file_processor import FileProcessorInterface


class PDFProcessor(FileProcessorInterface):
    """PDF file processor implementation"""

    @property
    def supported_extensions(self) -> List[str]:
        return [".pdf"]

    def extract_text(self, file_content: bytes, filename: str, **kwargs) -> str:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
