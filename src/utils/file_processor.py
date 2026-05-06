import PyPDF2
from docx import Document
import io

class FileProcessor:
    @staticmethod
    def extract_text(file_content: bytes, filename: str) -> str:
        text = ""
        if filename.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        elif filename.endswith(".docx"):
            doc = Document(io.BytesIO(file_content))
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif filename.endswith(".txt"):
            text = file_content.decode("utf-8")
        print(f"Extracted {len(text)} characters from {filename}")
        return text
