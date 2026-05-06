from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.core.config import Config

class QdrantService:
    def __init__(self):
        self.client = QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2", google_api_key=Config.GOOGLE_API_KEY)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    def ensure_collection(self, user_id: str, force_recreate: bool = False):
        collection_name = f"user_{user_id}"
        if force_recreate:
            try:
                self.client.delete_collection(collection_name)
                print(f"Deleted old collection: {collection_name}")
            except Exception:
                pass

        try:
            self.client.get_collection(collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
            )
            print(f"Created fresh collection: {collection_name}")
        return collection_name

    def add_documents(self, user_id: str, text: str, metadata: dict):
        collection_name = self.ensure_collection(user_id, force_recreate=False)
        chunks = self.text_splitter.split_text(text)
        print(f"Split text into {len(chunks)} chunks for user {user_id}")
        
        vectorstore = self.get_vectorstore(user_id)
        
        batch_size = 5
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vectorstore.add_texts(batch, metadatas=[metadata] * len(batch))
            print(f"Added batch {i//batch_size + 1}")
        
        final_info = self.client.get_collection(collection_name)
        print(f"Final Database State for {user_id}: {final_info.points_count} chunks total.")

    def get_vectorstore(self, user_id: str):
        collection_name = f"user_{user_id}"
        return QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings
        )
