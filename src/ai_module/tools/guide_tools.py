"""
Guide Tools - Tools for generating markdown guides.
"""

from typing import Optional, Dict, Any
from pydantic import Field

from .base import AITool, ToolInput, ToolOutput
from ..circuit.schemas import Circuit


# ==================== Input Schemas ====================

class GenerateGuideInput(ToolInput):
    """Input for generating a guide"""
    circuit: Dict[str, Any] = Field(..., description="Circuit JSON to generate guide for")


# ==================== Output Schemas ====================

class GenerateGuideOutput(ToolOutput):
    """Output for guide generation"""
    guide_markdown: str = Field(default="", description="Generated markdown guide")


# ==================== Tool Implementations ====================

class GenerateGuideTool(AITool):
    """Generate a step-by-step markdown guide for a circuit"""
    
    name = "generate_guide"
    description = """
    Generate a comprehensive step-by-step markdown guide for a circuit.
    The guide includes: overview, components list, assembly steps, connections, design rationale, and testing instructions.
    
    Use this after building a circuit to provide the user with assembly instructions.
    """
    input_schema = GenerateGuideInput
    output_schema = GenerateGuideOutput
    
    def __init__(self, generator):
        self.generator = generator
    
    def execute(self, **kwargs) -> GenerateGuideOutput:
        """Execute guide generation"""
        try:
            input_data = GenerateGuideInput(**kwargs)
            
            # Convert dict to Circuit object
            circuit = Circuit(**input_data.circuit)
            
            guide = self.generator.generate(circuit)
            
            return GenerateGuideOutput(
                success=True,
                message="Guide generated successfully",
                data={"guide_markdown": guide}
            )
        except Exception as e:
            return GenerateGuideOutput(
                success=False,
                message="Error generating guide",
                error=str(e)
            )
