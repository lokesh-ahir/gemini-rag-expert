from langchain.tools import tool
from src.services.qdrant_service import QdrantService

qdrant_service = QdrantService()

@tool
def search_knowledge_base(query: str) -> str:
    """Search the company knowledge base for relevant information about the provided query."""
    docs = qdrant_service.similarity_search(query)
    # Join documents into a single string for the LLM to read
    return "\n\n".join([doc.page_content for doc in docs])
