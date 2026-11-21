"""
Circuit Builder - Creates JSON circuits from retrieved components.

Uses RAG to fetch relevant components, then builds valid circuits.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

from .schemas import Circuit, CircuitComponent, Connection, Pin, CircuitValidationError
from ..rag.circuit_retriever import CircuitRetriever, Component


@dataclass
class BuildContext:
    """Context for building a circuit"""
    query: str
    design_id: Optional[str] = None
    workspace_id: Optional[str] = None
    constraints: Dict[str, Any] = None


class CircuitBuilder:
    """
    Builds circuit JSON from user queries and retrieved components.
    
    Usage:
        builder = CircuitBuilder(retriever)
        circuit = builder.build(
            query="Create an amplifier with LM386",
            design_id="design_123"
        )
    """
    
    def __init__(self, retriever: CircuitRetriever):
        """
        Initialize the builder.
        
        Args:
            retriever: CircuitRetriever instance for fetching components
        """
        self.retriever = retriever
    
    def build(
        self,
        query: str,
        design_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[Circuit], List[CircuitValidationError]]:
        """
        Build a circuit from a query.
        
        Args:
            query: User's circuit creation request
            design_id: Optional design ID for context
            workspace_id: Optional workspace ID
            constraints: Optional constraints (e.g., max components, budget)
        
        Returns:
            Tuple of (circuit, validation_errors)
        """
        context = BuildContext(
            query=query,
            design_id=design_id,
            workspace_id=workspace_id,
            constraints=constraints or {}
        )
        
        # Parse the query to extract requirements
        requirements = self._parse_query(query)
        
        # Search for relevant components
        components = self._search_components(requirements)
        
        if not components:
            return None, [CircuitValidationError(
                message="No components found matching the query",
                severity="error"
            )]
        
        # Build the circuit
        circuit = self._build_circuit(components, requirements, context)
        
        # Validate the circuit
        errors = self._validate_circuit(circuit)
        
        return circuit, errors
    
    def _parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse the user query to extract circuit requirements.
        
        Returns:
            Dict with: component_types, values, constraints, etc.
        """
        query_lower = query.lower()
        requirements: Dict[str, Any] = {
            "component_types": [],
            "values": {},
            "connections": [],
            "constraints": {}
        }
        
        # Common component types
        component_keywords = {
            "resistor": ["resistor", "resistors"],
            "capacitor": ["capacitor", "capacitors", "cap"],
            "ic": ["ic", "chip", "integrated circuit", "lm386", "555"],
            "transistor": ["transistor", "npn", "pnp"],
            "diode": ["diode", "led"],
            "inductor": ["inductor", "coil"],
            "switch": ["switch", "button"],
            "battery": ["battery", "power"],
            "ground": ["ground", "gnd"]
        }
        
        # Detect component types
        for comp_type, keywords in component_keywords.items():
            if any(kw in query_lower for kw in keywords):
                requirements["component_types"].append(comp_type)
        
        # Extract values (this is simplified - would use regex in production)
        if "1k" in query_lower or "1 k" in query_lower:
            requirements["values"]["resistor"] = "1k"
        if "100uf" in query_lower or "100 uf" in query_lower:
            requirements["values"]["capacitor"] = "100uF"
        
        # Extract circuit type
        if "amplifier" in query_lower:
            requirements["circuit_type"] = "amplifier"
        elif "oscillator" in query_lower:
            requirements["circuit_type"] = "oscillator"
        elif "filter" in query_lower:
            requirements["circuit_type"] = "filter"
        
        return requirements
    
    def _search_components(self, requirements: Dict[str, Any]) -> List[Component]:
        """
        Search for components matching the requirements.
        
        Args:
            requirements: Parsed requirements from query
        
        Returns:
            List of matching Component objects
        """
        component_types = requirements.get("component_types", [])
        
        # Search for each component type
        all_components: List[Component] = []
        
        for comp_type in component_types:
            # Get value if specified
            value = requirements.get("values", {}).get(comp_type)
            
            # Build specifications filter
            specs = {}
            if value:
                specs["value"] = value
            
            # Search for this component type
            components = self.retriever.search_components(
                query=comp_type,
                component_type=comp_type,
                specifications=specs,
                limit=5
            )
            
            all_components.extend(components)
        
        # If no specific types, do a general search
        if not all_components and requirements.get("circuit_type"):
            all_components = self.retriever.search_components(
                query=requirements["circuit_type"],
                limit=10
            )
        
        return all_components
    
    def _build_circuit(
        self,
        components: List[Component],
        requirements: Dict[str, Any],
        context: BuildContext
    ) -> Circuit:
        """
        Build a Circuit object from retrieved components.
        
        This is where the actual circuit design logic goes.
        For now, it creates a simple circuit with the components.
        In production, this would use AI to intelligently connect components.
        """
        circuit_components: List[CircuitComponent] = []
        connections: List[Connection] = []
        
        # Convert Component to CircuitComponent
        for i, comp in enumerate(components):
            comp_id = comp.id or f"comp_{i}"
            
            # Build pins
            pins = []
            if comp.pins:
                for pin_data in comp.pins:
                    pin = Pin(
                        name=pin_data.get("name", f"pin{i}"),
                        type=pin_data.get("type", "signal"),
                        description=pin_data.get("description")
                    )
                    pins.append(pin)
            else:
                # Default pins if not specified
                pins = [Pin(name="pin1"), Pin(name="pin2")]
            
            # Build specs
            specs_data = comp.specifications or {}
            specs = {
                "value": comp.value,
                **specs_data
            }
            
            circuit_comp = CircuitComponent(
                id=comp_id,
                type=comp.type,
                name=comp.name or comp.type,
                description=comp.description,
                specs=specs,
                pins=pins
            )
            
            circuit_components.append(circuit_comp)
        
        # Create basic connections (simplified - would use AI in production)
        if len(circuit_components) >= 2:
            # Connect first component's pin1 to second component's pin1
            connections.append(Connection(
                from_component=circuit_components[0].id,
                from_pin="pin1",
                to_component=circuit_components[1].id,
                to_pin="pin1",
                type="signal",
                description="Basic connection"
            ))
        
        # Generate circuit name
        circuit_type = requirements.get("circuit_type", "circuit")
        circuit_name = f"{circuit_type}_design"
        
        if context.design_id:
            circuit_name = f"{context.design_id}_{circuit_type}"
        
        # Build metadata
        metadata = {
            "query": context.query,
            "design_id": context.design_id,
            "workspace_id": context.workspace_id,
            "component_count": len(circuit_components),
            "connection_count": len(connections)
        }
        
        return Circuit(
            name=circuit_name,
            description=requirements.get("description"),
            components=circuit_components,
            connections=connections,
            metadata=metadata
        )
    
    def _validate_circuit(self, circuit: Circuit) -> List[CircuitValidationError]:
        """
        Validate the circuit for common issues.
        
        Checks:
        - All components have valid IDs
        - All pins referenced in connections exist
        - No duplicate component IDs
        - etc.
        """
        errors: List[CircuitValidationError] = []
        
        if not circuit.components:
            errors.append(CircuitValidationError(
                message="Circuit has no components",
                severity="error"
            ))
            return errors
        
        # Check for duplicate component IDs
        component_ids = [c.id for c in circuit.components]
        if len(component_ids) != len(set(component_ids)):
            errors.append(CircuitValidationError(
                message="Duplicate component IDs found",
                severity="error"
            ))
        
        # Check that all connection pins exist
        component_pins: Dict[str, List[str]] = {}
        for comp in circuit.components:
            component_pins[comp.id] = [p.name for p in comp.pins]
        
        for conn in circuit.connections:
            # Check from pin
            if conn.from_component not in component_pins:
                errors.append(CircuitValidationError(
                    component_id=conn.from_component,
                    message=f"Component '{conn.from_component}' not found",
                    severity="error"
                ))
            elif conn.from_pin not in component_pins.get(conn.from_component, []):
                errors.append(CircuitValidationError(
                    component_id=conn.from_component,
                    pin_name=conn.from_pin,
                    message=f"Pin '{conn.from_pin}' not found on component '{conn.from_component}'",
                    severity="error"
                ))
            
            # Check to pin
            if conn.to_component not in component_pins:
                errors.append(CircuitValidationError(
                    component_id=conn.to_component,
                    message=f"Component '{conn.to_component}' not found",
                    severity="error"
                ))
            elif conn.to_pin not in component_pins.get(conn.to_component, []):
                errors.append(CircuitValidationError(
                    component_id=conn.to_component,
                    pin_name=conn.to_pin,
                    message=f"Pin '{conn.to_pin}' not found on component '{conn.to_component}'",
                    severity="error"
                ))
        
        return errors
    
    def to_json(self, circuit: Circuit) -> str:
        """Convert circuit to JSON string"""
        return circuit.model_dump_json(indent=2)
    
    def from_json(self, json_str: str) -> Circuit:
        """Load circuit from JSON string"""
        return Circuit.model_validate_json(json_str)


# Singleton instance
_builder: Optional[CircuitBuilder] = None


def get_circuit_builder(retriever: CircuitRetriever) -> CircuitBuilder:
    """Get or create the singleton circuit builder"""
    global _builder
    if _builder is None:
        _builder = CircuitBuilder(retriever)
    return _builder


def reset_circuit_builder() -> None:
    """Reset the singleton"""
    global _builder
    _builder = None
