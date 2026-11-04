from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.interfaces.embedding import EmbeddingInterface


class GeminiEmbedding(EmbeddingInterface):
    """Gemini embedding implementation"""

    def __init__(self, model: str, api_key: str, **kwargs):
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=api_key
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        return self.embedding_model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.embedding_model.embed_query(text)
