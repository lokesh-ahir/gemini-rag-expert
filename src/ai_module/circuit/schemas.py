"""
Pydantic schemas for circuit JSON
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class Pin(BaseModel):
    """A pin on a component"""
    name: str = Field(..., description="Pin name (e.g., 'pin1', 'VCC', 'GND')")
    type: str = Field(default="signal", description="Pin type (signal, power, ground)")
    description: Optional[str] = Field(default=None, description="Pin description")


class ComponentSpec(BaseModel):
    """Specifications for a component"""
    value: Optional[str] = Field(default=None, description="Component value (e.g., '1k', '100uF')")
    tolerance: Optional[str] = Field(default=None, description="Tolerance (e.g., '5%', '10%')")
    voltage: Optional[str] = Field(default=None, description="Voltage rating")
    power: Optional[str] = Field(default=None, description="Power rating")
    package: Optional[str] = Field(default=None, description="Package type")
    
    # Allow any additional specs
    class Config:
        extra = "allow"


class CircuitComponent(BaseModel):
    """A component in the circuit"""
    id: str = Field(..., description="Unique component ID")
    type: str = Field(..., description="Component type (resistor, capacitor, ic, etc.)")
    name: str = Field(..., description="Component name/label")
    description: Optional[str] = Field(default=None, description="Component description")
    specs: ComponentSpec = Field(default_factory=ComponentSpec)
    pins: List[Pin] = Field(default_factory=list, description="Component pins")


class Connection(BaseModel):
    """A connection between two pins"""
    from_component: str = Field(..., description="Source component ID")
    from_pin: str = Field(..., description="Source pin name")
    to_component: str = Field(..., description="Destination component ID")
    to_pin: str = Field(..., description="Destination pin name")
    type: str = Field(default="signal", description="Connection type (signal, power, ground)")
    description: Optional[str] = Field(default=None, description="Connection description")
    
    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        valid_types = ["signal", "power", "ground", "data", "clock"]
        if v not in valid_types:
            raise ValueError(f"type must be one of {valid_types}")
        return v


class Circuit(BaseModel):
    """Complete circuit schema"""
    name: str = Field(..., description="Circuit name")
    description: Optional[str] = Field(default=None, description="Circuit description")
    components: List[CircuitComponent] = Field(default_factory=list)
    connections: List[Connection] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "amplifier_circuit",
                "description": "Simple audio amplifier",
                "components": [
                    {
                        "id": "R1",
                        "type": "resistor",
                        "name": "Input Resistor",
                        "specs": {"value": "1k", "tolerance": "5%"},
                        "pins": [{"name": "pin1"}, {"name": "pin2"}]
                    }
                ],
                "connections": [
                    {
                        "from_component": "R1",
                        "from_pin": "pin1",
                        "to_component": "IC1",
                        "to_pin": "pin5",
                        "type": "signal"
                    }
                ]
            }
        }


class CircuitValidationError(BaseModel):
    """Error during circuit validation"""
    component_id: Optional[str] = None
    pin_name: Optional[str] = None
    message: str
    severity: str = "error"  # error, warning, info
