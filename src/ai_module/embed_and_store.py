"""
Embed and Store Components in Qdrant Cloud

This script reads all component files from a folder, generates embeddings using
Gemini embedding model, and stores them in Qdrant Cloud vector database.

Usage:
    python embed_and_store.py \
        --folder /path/to/components \
        --qdrant-url https://your-cluster.qdrant.io \
        --qdrant-api-key your-api-key \
        --collection circuit_components \
        --google-api-key your-google-api-key

Supported file types: .txt, .md, .pdf, .docx, .json

Features:
- Automatic file type detection
- Text extraction from various formats
- Batch processing for efficiency
- Metadata preservation
- Progress tracking
- Error handling and retries
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

# Try to import required libraries
try:
    import PyPDF2
    from docx import Document
    from qdrant_client import QdrantClient, models
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    LIBRARIES_AVAILABLE = True
except ImportError as e:
    LIBRARIES_AVAILABLE = False
    MISSING_LIBRARIES = str(e)


@dataclass
class Component:
    """Represents a circuit component with its metadata"""
    id: str
    name: str
    type: str
    description: str
    value: Optional[str] = None
    pins: Optional[List[Dict[str, Any]]] = None
    specifications: Optional[Dict[str, Any]] = None
    category: Optional[str] = None
    datasheet_url: Optional[str] = None
    source_file: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Convert component to Qdrant payload format"""
        payload = {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "description": self.description,
        }
        
        if self.value:
            payload["value"] = self.value
        if self.pins:
            payload["pins"] = self.pins
        if self.specifications:
            payload["specifications"] = self.specifications
        if self.category:
            payload["category"] = self.category
        if self.datasheet_url:
            payload["datasheet_url"] = self.datasheet_url
        if self.source_file:
            payload["source_file"] = self.source_file
        if self.metadata:
            payload.update(self.metadata)
        
        # Add text for semantic search
        payload["text"] = self._generate_text()
        
        return payload
    
    def _generate_text(self) -> str:
        """Generate text representation for embedding"""
        parts = [
            f"Component: {self.name}",
            f"Type: {self.type}",
            self.description or "",
        ]
        
        if self.value:
            parts.append(f"Value: {self.value}")
        if self.category:
            parts.append(f"Category: {self.category}")
        if self.pins:
            pin_text = ", ".join([p.get("name", "") for p in self.pins])
            parts.append(f"Pins: {pin_text}")
        if self.specifications:
            spec_text = " ".join([f"{k}={v}" for k, v in self.specifications.items()])
            parts.append(f"Specifications: {spec_text}")
        
        return " \n ".join(parts)


class ComponentEmbedder:
    """
    Embeds circuit components and stores them in Qdrant.
    """
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        ".txt": "text",
        ".md": "markdown",
        ".pdf": "pdf",
        ".docx": "docx",
        ".json": "json"
    }
    
    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        google_api_key: str,
        collection_name: str = "circuit_components",
        embedding_model: str = "models/gemini-embedding-2",
        vector_size: int = 3072,
        batch_size: int = 100
    ):
        """
        Initialize the embedder.
        
        Args:
            qdrant_url: Qdrant Cloud cluster URL
            qdrant_api_key: Qdrant Cloud API key
            google_api_key: Google API key for Gemini embeddings
            collection_name: Name of the Qdrant collection
            embedding_model: Gemini embedding model to use
            vector_size: Size of embedding vectors
            batch_size: Number of components to process at once
        """
        if not LIBRARIES_AVAILABLE:
            raise ImportError(f"Missing required libraries: {MISSING_LIBRARIES}")
        
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.google_api_key = google_api_key
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.vector_size = vector_size
        self.batch_size = batch_size
        
        # Initialize clients
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=google_api_key
        )
    
    def extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text from a file based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text content
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            
            elif ext == ".md":
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            
            elif ext == ".pdf":
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    return text
            
            elif ext == ".docx":
                doc = Document(file_path)
                return "\n".join([para.text for para in doc.paragraphs])
            
            elif ext == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # If it's a component JSON, extract relevant fields
                    if isinstance(data, dict):
                        text_parts = []
                        for key, value in data.items():
                            text_parts.append(f"{key}: {value}")
                        return "\n".join(text_parts)
                    return json.dumps(data)
            
            else:
                raise ValueError(f"Unsupported file type: {ext}")
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""
    
    def parse_component_from_text(self, text: str, file_path: str) -> Optional[Component]:
        """
        Parse component information from text.
        
        This is a simple parser. For more complex formats, you may need
        to customize this method or use structured data files (JSON).
        
        Args:
            text: Extracted text from file
            file_path: Path to the source file
            
        Returns:
            Component object or None if parsing fails
        """
        if not text.strip():
            return None
        
        # Try to parse as JSON first
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return self._parse_component_from_dict(data, file_path)
        except json.JSONDecodeError:
            pass
        
        # For text/markdown files, use simple parsing
        return self._parse_component_from_text_simple(text, file_path)
    
    def _parse_component_from_dict(self, data: Dict[str, Any], file_path: str) -> Component:
        """Parse component from dictionary (JSON) data"""
        component_id = data.get("id", os.path.splitext(os.path.basename(file_path))[0])
        
        return Component(
            id=str(component_id),
            name=data.get("name", ""),
            type=data.get("type", "component"),
            description=data.get("description", ""),
            value=data.get("value"),
            pins=data.get("pins"),
            specifications=data.get("specifications", {}),
            category=data.get("category"),
            datasheet_url=data.get("datasheet_url"),
            source_file=file_path,
            metadata=data.get("metadata", {})
        )
    
    def _parse_component_from_text_simple(self, text: str, file_path: str) -> Component:
        """Parse component from simple text/markdown"""
        lines = text.strip().split("\n")
        
        component_data = {
            "name": "",
            "type": "component",
            "description": "",
            "value": None,
            "pins": [],
            "specifications": {},
            "category": None,
            "datasheet_url": None
        }
        
        current_key = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for key-value pairs
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key in ["name", "type", "description", "category", "datasheet_url"]:
                    component_data[key] = value
                elif key == "value":
                    component_data["value"] = value
                elif key == "pins":
                    component_data["pins"] = [p.strip() for p in value.split(",")]
                else:
                    component_data["specifications"][key] = value
            elif current_key == "description":
                component_data["description"] += " " + line
        
        # Generate ID from filename if not specified
        component_id = os.path.splitext(os.path.basename(file_path))[0]
        
        return Component(
            id=component_id,
            name=component_data["name"] or component_id,
            type=component_data["type"],
            description=component_data["description"],
            value=component_data["value"],
            pins=component_data["pins"],
            specifications=component_data["specifications"],
            category=component_data["category"],
            datasheet_url=component_data["datasheet_url"],
            source_file=file_path
        )
    
    def ensure_collection(self) -> None:
        """Ensure the Qdrant collection exists"""
        try:
            self.qdrant_client.get_collection(self.collection_name)
        except Exception:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection: {self.collection_name}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for text using Gemini.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            return self.embedding_model.embed_query(text)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * self.vector_size
    
    def embed_components(self, folder_path: str) -> Tuple[int, int]:
        """
        Embed all components from a folder and store in Qdrant.
        
        Args:
            folder_path: Path to folder containing component files
            
        Returns:
            Tuple of (success_count, failure_count)
        """
        # Validate folder
        if not os.path.isdir(folder_path):
            raise ValueError(f"Folder does not exist: {folder_path}")
        
        # Ensure collection exists
        self.ensure_collection()
        
        # Find all supported files
        files = []
        for root, _, filenames in os.walk(folder_path):
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    files.append(os.path.join(root, filename))
        
        if not files:
            print(f"No supported files found in {folder_path}")
            return 0, 0
        
        print(f"Found {len(files)} component files to process")
        
        # Process in batches
        success_count = 0
        failure_count = 0
        
        for i in range(0, len(files), self.batch_size):
            batch_files = files[i:i + self.batch_size]
            print(f"Processing batch {i//self.batch_size + 1}/{(len(files) + self.batch_size - 1)//self.batch_size}")
            
            batch_success, batch_failure = self._process_batch(batch_files)
            success_count += batch_success
            failure_count += batch_failure
        
        print(f"\nEmbedding complete!")
        print(f"Success: {success_count}")
        print(f"Failed: {failure_count}")
        
        return success_count, failure_count
    
    def _process_batch(self, files: List[str]) -> Tuple[int, int]:
        """Process a batch of files"""
        components = []
        
        # Parse all files in batch
        for file_path in files:
            try:
                text = self.extract_text_from_file(file_path)
                if text:
                    component = self.parse_component_from_text(text, file_path)
                    if component:
                        components.append(component)
                        print(f"  ✓ {file_path}")
                    else:
                        print(f"  ✗ {file_path} (failed to parse)")
                else:
                    print(f"  ✗ {file_path} (empty)")
            except Exception as e:
                print(f"  ✗ {file_path} ({e})")
        
        if not components:
            return 0, len(files)
        
        # Generate embeddings for all components
        texts = [comp._generate_text() for comp in components]
        
        try:
            embeddings = self.embedding_model.embed_documents(texts)
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return 0, len(components)
        
        # Upload to Qdrant
        points = []
        for idx, (component, embedding) in enumerate(zip(components, embeddings)):
            payload = component.to_qdrant_payload()
            
            # Use file path hash + index as ID to ensure uniqueness
            point_id = f"{abs(hash(file_path))}_{idx}"
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
            )
        
        # Upload in smaller batches to avoid timeouts
        upload_batch_size = 50
        for i in range(0, len(points), upload_batch_size):
            batch = points[i:i + upload_batch_size]
            try:
                self.qdrant_client.upload_points(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True
                )
            except Exception as e:
                print(f"Error uploading batch: {e}")
                return 0, len(components)
        
        return len(components), 0
    
    def clear_collection(self) -> None:
        """Clear all points from the collection"""
        try:
            self.qdrant_client.delete_collection(self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            print(f"Error deleting collection: {e}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "vectors_config": info.vectors_config.dict()
            }
        except Exception as e:
            return {"error": str(e)}


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Embed circuit components and store in Qdrant Cloud"
    )
    
    parser.add_argument(
        "--folder", 
        type=str, 
        required=True,
        help="Path to folder containing component files"
    )
    
    parser.add_argument(
        "--qdrant-url",
        type=str,
        required=True,
        help="Qdrant Cloud cluster URL (e.g., https://your-cluster.qdrant.io)"
    )
    
    parser.add_argument(
        "--qdrant-api-key",
        type=str,
        required=True,
        help="Qdrant Cloud API key"
    )
    
    parser.add_argument(
        "--google-api-key",
        type=str,
        required=True,
        help="Google API key for Gemini embeddings"
    )
    
    parser.add_argument(
        "--collection",
        type=str,
        default="circuit_components",
        help="Name of the Qdrant collection (default: circuit_components)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of components to process at once (default: 100)"
    )
    
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the collection before embedding"
    )
    
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show collection info and exit"
    )
    
    args = parser.parse_args()
    
    # Check libraries
    if not LIBRARIES_AVAILABLE:
        print(f"ERROR: Missing required libraries: {MISSING_LIBRARIES}")
        print("\nInstall them with:")
        print("  pip install PyPDF2 python-docx qdrant-client langchain-google-genai")
        sys.exit(1)
    
    # Initialize embedder
    print("Initializing embedder...")
    embedder = ComponentEmbedder(
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        google_api_key=args.google_api_key,
        collection_name=args.collection,
        batch_size=args.batch_size
    )
    
    # Show collection info if requested
    if args.info:
        info = embedder.get_collection_info()
        print("\nCollection Info:")
        print(json.dumps(info, indent=2))
        sys.exit(0)
    
    # Clear collection if requested
    if args.clear:
        embedder.clear_collection()
    
    # Ensure collection exists
    embedder.ensure_collection()
    
    # Process folder
    print(f"\nProcessing folder: {args.folder}")
    start_time = time.time()
    
    success, failure = embedder.embed_components(args.folder)
    
    elapsed = time.time() - start_time
    print(f"\nTime taken: {elapsed:.2f} seconds")
    
    # Show final collection info
    info = embedder.get_collection_info()
    print(f"\nFinal collection info:")
    print(f"  Total points: {info.get('points_count', 0)}")
    
    if failure > 0:
        print(f"\nWARNING: {failure} files failed to process")
        sys.exit(1)
    
    print("\n✅ All components embedded and stored successfully!")


if __name__ == "__main__":
    main()
