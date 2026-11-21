"""
Circuit Tools - Tools for component retrieval and circuit building.
"""

from typing import Optional, List, Dict, Any
from pydantic import Field

from .base import AITool, ToolInput, ToolOutput


# ==================== Input Schemas ====================

class SearchComponentsInput(ToolInput):
    """Input for searching circuit components"""
    query: str = Field(..., description="Search query for components")
    component_type: Optional[str] = Field(default=None, description="Filter by component type (resistor, capacitor, etc.)")
    category: Optional[str] = Field(default=None, description="Filter by category")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of results")


class BuildCircuitInput(ToolInput):
    """Input for building a circuit"""
    query: str = Field(..., description="Circuit creation request")
    design_id: Optional[str] = Field(default=None, description="Design ID for context")
    workspace_id: Optional[str] = Field(default=None, description="Workspace ID for context")
    constraints: Optional[Dict[str, Any]] = Field(default=None, description="Design constraints")


class GetComponentInput(ToolInput):
    """Input for getting a specific component"""
    component_id: str = Field(..., description="Component ID")


# ==================== Output Schemas ====================

class SearchComponentsOutput(ToolOutput):
    """Output for component search"""
    components: List[Dict[str, Any]] = Field(default_factory=list, description="List of found components")


class BuildCircuitOutput(ToolOutput):
    """Output for circuit building"""
    circuit: Optional[Dict[str, Any]] = Field(default=None, description="Generated circuit JSON")
    validation_errors: List[Dict[str, Any]] = Field(default_factory=list, description="Validation errors")


class GetComponentOutput(ToolOutput):
    """Output for getting a component"""
    component: Optional[Dict[str, Any]] = Field(default=None, description="Component data")


# ==================== Tool Implementations ====================

class SearchComponentsTool(AITool):
    """Search for circuit components in Qdrant"""
    
    name = "search_components"
    description = """
    Search for circuit components in the Qdrant vector database.
    Use this when you need to find specific components like resistors, capacitors, ICs, etc.
    
    Example usage:
    - "Search for 1k ohm resistors"
    - "Find LM386 amplifier IC"
    - "Get all capacitors"
    """
    input_schema = SearchComponentsInput
    output_schema = SearchComponentsOutput
    
    def __init__(self, retriever):
        self.retriever = retriever
    
    def execute(self, **kwargs) -> SearchComponentsOutput:
        """Execute the component search"""
        try:
            input_data = SearchComponentsInput(**kwargs)
            
            components = self.retriever.search_components(
                query=input_data.query,
                component_type=input_data.component_type,
                category=input_data.category,
                limit=input_data.limit
            )
            
            components_dict = [comp.to_dict() for comp in components]
            
            return SearchComponentsOutput(
                success=True,
                message=f"Found {len(components_dict)} components",
                data={"components": components_dict}
            )
        except Exception as e:
            return SearchComponentsOutput(
                success=False,
                message="Error searching components",
                error=str(e)
            )


class BuildCircuitTool(AITool):
    """Build a circuit from requirements"""
    
    name = "build_circuit"
    description = """
    Build a circuit JSON from user requirements.
    Use this when the user wants to create a new circuit design.
    
    This tool will:
    1. Search for relevant components based on the query
    2. Build a valid circuit JSON
    3. Validate the circuit
    
    Example usage:
    - "Build an amplifier circuit with LM386"
    - "Create a 555 timer oscillator"
    - "Design a power supply circuit"
    """
    input_schema = BuildCircuitInput
    output_schema = BuildCircuitOutput
    
    def __init__(self, builder):
        self.builder = builder
    
    def execute(self, **kwargs) -> BuildCircuitOutput:
        """Execute circuit building"""
        try:
            input_data = BuildCircuitInput(**kwargs)
            
            circuit, errors = self.builder.build(
                query=input_data.query,
                design_id=input_data.design_id,
                workspace_id=input_data.workspace_id,
                constraints=input_data.constraints
            )
            
            return BuildCircuitOutput(
                success=True,
                message="Circuit built successfully" if circuit else "Circuit building failed",
                data={
                    "circuit": circuit.model_dump() if circuit else None,
                    "validation_errors": [e.model_dump() for e in errors]
                }
            )
        except Exception as e:
            return BuildCircuitOutput(
                success=False,
                message="Error building circuit",
                error=str(e)
            )


class GetComponentTool(AITool):
    """Get a specific component by ID"""
    
    name = "get_component"
    description = """
    Get detailed information about a specific component by its ID.
    Use this when you need to know more about a particular component.
    """
    input_schema = GetComponentInput
    output_schema = GetComponentOutput
    
    def __init__(self, retriever):
        self.retriever = retriever
    
    def execute(self, **kwargs) -> GetComponentOutput:
        """Execute component retrieval"""
        try:
            input_data = GetComponentInput(**kwargs)
            component = self.retriever.get_component(input_data.component_id)
            
            return GetComponentOutput(
                success=True,
                message="Component retrieved",
                data={"component": component.to_dict() if component else None}
            )
        except Exception as e:
            return GetComponentOutput(
                success=False,
                message="Error getting component",
                error=str(e)
            )
