"""
Circuit Component Retriever - Fetches components from Qdrant.

The Qdrant database contains 1800+ components with:
- Component details (type, value, specifications)
- Pin configurations
- Descriptions
- Categories

This retriever searches for relevant components based on:
- Component type (resistor, capacitor, IC, etc.)
- Specifications (value, tolerance, voltage, etc.)
- Pin requirements
- Text descriptions
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct


@dataclass
class Component:
    """Represents a circuit component from Qdrant"""
    id: str
    type: str  # resistor, capacitor, ic, transistor, etc.
    name: str
    value: Optional[str] = None
    description: Optional[str] = None
    pins: List[Dict[str, Any]] = None  # List of pin definitions
    specifications: Dict[str, Any] = None  # Voltage, tolerance, etc.
    category: Optional[str] = None
    datasheet_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "value": self.value,
            "description": self.description,
            "pins": self.pins or [],
            "specifications": self.specifications or {},
            "category": self.category,
            "datasheet_url": self.datasheet_url
        }


class CircuitRetriever:
    """
    Retrieves circuit components from Qdrant vector store.
    
    Usage:
        retriever = CircuitRetriever(
            qdrant_url="http://localhost:6333",
            collection_name="circuit_components"
        )
        
        # Search for resistors
        components = retriever.search_components(
            query="1k ohm resistor",
            component_type="resistor",
            limit=10
        )
        
        # Get component by ID
        component = retriever.get_component("resistor_1k_5")
    """
    
    def __init__(
        self,
        qdrant_url: str,
        collection_name: str = "circuit_components",
        api_key: Optional[str] = None
    ):
        """
        Initialize the retriever.
        
        Args:
            qdrant_url: URL of Qdrant server
            collection_name: Name of the collection (default: circuit_components)
            api_key: Optional API key for Qdrant Cloud
        """
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.client = QdrantClient(url=qdrant_url, api_key=api_key)
        self._embedding_model = None
    
    def ensure_collection(self, vector_size: int = 3072) -> None:
        """Ensure the collection exists"""
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
    
    def search_components(
        self,
        query: str,
        component_type: Optional[str] = None,
        category: Optional[str] = None,
        specifications: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[Component]:
        """
        Search for components matching the query and filters.
        
        Args:
            query: Text query (e.g., "1k resistor", "555 timer")
            component_type: Filter by type (resistor, capacitor, etc.)
            category: Filter by category (passive, active, etc.)
            specifications: Filter by specs (e.g., {"value": "1k", "tolerance": "5%"})
            limit: Maximum results to return
        
        Returns:
            List of matching Component objects
        """
        # Build the query vector (using Qdrant's neural search)
        # For now, we'll use keyword search and filter by metadata
        
        # Build must conditions for filtering
        must_conditions = []
        
        if component_type:
            must_conditions.append(
                models.FieldCondition(
                    key="type",
                    match=models.MatchValue(value=component_type)
                )
            )
        
        if category:
            must_conditions.append(
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value=category)
                )
            )
        
        # For specifications, we need to match all provided specs
        if specifications:
            for key, value in specifications.items():
                must_conditions.append(
                    models.FieldCondition(
                        key=f"specifications.{key}",
                        match=models.MatchValue(value=value)
                    )
                )
        
        # Build the search query
        search_query = models.SearchRequest(
            collection_name=self.collection_name,
            query=query,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        
        # If we have filters, use scroll for more control
        if must_conditions:
            search_query = models.ScrollRequest(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=must_conditions
                ) if must_conditions else None,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
        
        try:
            # Execute search
            if isinstance(search_query, models.ScrollRequest):
                results = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=models.Filter(must=must_conditions) if must_conditions else None,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False
                )
                points = results[0]  # First batch
            else:
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=self._get_query_vector(query),
                    limit=limit,
                    with_payload=True,
                    with_vectors=False
                )
                points = results
            
            # Convert to Component objects
            components = []
            for point in points:
                payload = point.payload
                component = Component(
                    id=point.id,
                    type=payload.get("type", ""),
                    name=payload.get("name", ""),
                    value=payload.get("value"),
                    description=payload.get("description"),
                    pins=payload.get("pins"),
                    specifications=payload.get("specifications"),
                    category=payload.get("category"),
                    datasheet_url=payload.get("datasheet_url")
                )
                components.append(component)
            
            return components
            
        except Exception as e:
            print(f"Error searching components: {e}")
            return []
    
    def get_component(self, component_id: str) -> Optional[Component]:
        """Get a specific component by ID"""
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[component_id],
                with_payload=True,
                with_vectors=False
            )
            
            if points:
                payload = points[0].payload
                return Component(
                    id=points[0].id,
                    type=payload.get("type", ""),
                    name=payload.get("name", ""),
                    value=payload.get("value"),
                    description=payload.get("description"),
                    pins=payload.get("pins"),
                    specifications=payload.get("specifications"),
                    category=payload.get("category"),
                    datasheet_url=payload.get("datasheet_url")
                )
            return None
        except Exception as e:
            print(f"Error getting component: {e}")
            return None
    
    def search_by_pins(
        self,
        required_pins: List[str],
        component_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Component]:
        """
        Search for components that have all required pins.
        
        Args:
            required_pins: List of pin names that must be present
            component_type: Optional filter by component type
            limit: Maximum results
        
        Returns:
            List of components with matching pins
        """
        must_conditions = []
        
        # Each pin must be present in the pins array
        for pin in required_pins:
            must_conditions.append(
                models.FieldCondition(
                    key="pins.name",
                    match=models.MatchValue(value=pin)
                )
            )
        
        if component_type:
            must_conditions.append(
                models.FieldCondition(
                    key="type",
                    match=models.MatchValue(value=component_type)
                )
            )
        
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(must=must_conditions) if must_conditions else None,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            components = []
            for point in results[0]:
                payload = point.payload
                component = Component(
                    id=point.id,
                    type=payload.get("type", ""),
                    name=payload.get("name", ""),
                    value=payload.get("value"),
                    description=payload.get("description"),
                    pins=payload.get("pins"),
                    specifications=payload.get("specifications"),
                    category=payload.get("category"),
                    datasheet_url=payload.get("datasheet_url")
                )
                components.append(component)
            
            return components
            
        except Exception as e:
            print(f"Error searching by pins: {e}")
            return []
    
    def get_all_components(
        self,
        component_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Component]:
        """Get all components, optionally filtered by type"""
        must_conditions = []
        
        if component_type:
            must_conditions.append(
                models.FieldCondition(
                    key="type",
                    match=models.MatchValue(value=component_type)
                )
            )
        
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(must=must_conditions) if must_conditions else None,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            components = []
            for point in results[0]:
                payload = point.payload
                component = Component(
                    id=point.id,
                    type=payload.get("type", ""),
                    name=payload.get("name", ""),
                    value=payload.get("value"),
                    description=payload.get("description"),
                    pins=payload.get("pins"),
                    specifications=payload.get("specifications"),
                    category=payload.get("category"),
                    datasheet_url=payload.get("datasheet_url")
                )
                components.append(component)
            
            return components
            
        except Exception as e:
            print(f"Error getting all components: {e}")
            return []
    
    def _get_query_vector(self, query: str) -> List[float]:
        """
        Get embedding vector for a query.
        For now, returns a dummy vector. In production, use actual embedding model.
        """
        # TODO: Integrate with actual embedding model
        # For now, return a vector of zeros (Qdrant will handle keyword search)
        return [0.0] * 3072


# Singleton instance
_retriever: Optional[CircuitRetriever] = None


def get_circuit_retriever(
    qdrant_url: str,
    collection_name: str = "circuit_components",
    api_key: Optional[str] = None
) -> CircuitRetriever:
    """Get or create the singleton circuit retriever"""
    global _retriever
    
    if _retriever is None:
        _retriever = CircuitRetriever(qdrant_url, collection_name, api_key)
    
    return _retriever


def reset_circuit_retriever() -> None:
    """Reset the singleton"""
    global _retriever
    _retriever = None
