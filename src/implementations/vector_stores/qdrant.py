from typing import List, Dict, Any
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.interfaces.vector_store import VectorStoreInterface
from src.core.container import ServiceContainer


class QdrantVectorStore(VectorStoreInterface):
    """Qdrant implementation of VectorStore"""

    def __init__(self, url: str, api_key: str = None, **kwargs):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self._embedding = None

    @property
    def embedding(self):
        """Lazy load embedding service"""
        if self._embedding is None:
            self._embedding = ServiceContainer.get_service("embedding")
        return self._embedding

    def ensure_collection(self, collection_name: str, **kwargs) -> None:
        """Ensure a collection exists"""
        try:
            self.client.get_collection(collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
            )
            print(f"Created fresh collection: {collection_name}")

    def add_documents(self, collection_name: str, texts: List[str], metadatas: List[Dict]) -> None:
        """Add documents to the vector store"""
        self.ensure_collection(collection_name)

        # Split texts into chunks
        all_chunks = []
        all_metadata = []
        
        for text, metadata in zip(texts, metadatas):
            chunks = self.text_splitter.split_text(text)
            print(f"Split text into {len(chunks)} chunks")
            all_chunks.extend(chunks)
            all_metadata.extend([metadata] * len(chunks))

        # Get embeddings for all chunks
        embeddings = self.embedding.embed_documents(all_chunks)

        # Add to Qdrant in batches
        batch_size = 5
        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_metadata = all_metadata[i:i + batch_size]
            
            self.client.upload_points(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=i + j,
                        vector=embedding,
                        payload={"text": chunk, **metadata}
                    )
                    for j, (chunk, embedding, metadata) in enumerate(
                        zip(batch_chunks, batch_embeddings, batch_metadata)
                    )
                ]
            )
            print(f"Added batch {i//batch_size + 1}")
        
        final_info = self.client.get_collection(collection_name)
        print(f"Final Database State: {final_info.points_count} chunks total.")

    def similarity_search(self, collection_name: str, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents"""
        query_embedding = self.embedding.embed_query(query)

        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=k,
        )

        return [
            Document(
                page_content=result.payload["text"],
                metadata=result.payload
            )
            for result in results
        ]

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection"""
        self.client.delete_collection(collection_name)
