from deepagents import create_deep_agent
from langchain_core.tools import tool
from src.core.config import Config
from src.services.qdrant_service import QdrantService

class GeminiService:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.qdrant_service = QdrantService()
        
        @tool
        def search_my_documents(query: str) -> str:
            """Search through your uploaded documents to find relevant information for the question."""
            print(f"Searching documents for: {query}")
            try:
                vectorstore = self.qdrant_service.get_vectorstore(self.user_id)
                collection_info = self.qdrant_service.client.get_collection(f"user_{self.user_id}")
                print(f"Collection info: {collection_info.points_count} total chunks stored.")
                
                docs = vectorstore.similarity_search(query, k=5)
                print(f"Found {len(docs)} relevant document chunks.")
                return "\n\n".join([doc.page_content for doc in docs])
            except Exception as e:
                print(f"Search Error: {e}")
                return f"Error searching documents: {str(e)}"

        self.tools = [search_my_documents]
        
        self.agent = create_deep_agent(
            model=f"google_genai:{Config.GEMINI_MODEL}",
            tools=self.tools,
            system_prompt=Config.SYSTEM_PROMPT + f"\n\nYou are currently assisting User: {self.user_id}."
        )

    def ask(self, question: str):
        print(f"Starting task for {self.user_id}")
        try:
            response = self.agent.invoke({
                "messages": [{"role": "user", "content": question}]
            })
            last_message = response["messages"][-1]
            content = last_message.content if hasattr(last_message, 'content') else str(last_message)
            print("Task Complete")
            return content
        except Exception as e:
            print(f"Agent Error: {e}")
            return f"Error: {str(e)}"
