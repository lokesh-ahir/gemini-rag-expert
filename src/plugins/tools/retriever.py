from typing import Any
from src.interfaces.tool import ToolInterface
from src.core.container import ServiceContainer


class RetrieverTool(ToolInterface):
    """Retriever tool for agent"""

    @property
    def name(self) -> str:
        return "search_my_documents"

    @property
    def description(self) -> str:
        return "Search through your uploaded documents to find relevant information for the question."

    def execute(self, query: str, **kwargs) -> Any:
        user_id = kwargs.get("user_id")
        if not user_id:
            raise ValueError("user_id is required")

        vector_store = ServiceContainer.get_service("vector_store")
        collection_name = f"user_{user_id}"

        docs = vector_store.similarity_search(collection_name, query, k=5)
        return "\n\n".join([doc.page_content for doc in docs])
