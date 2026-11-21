"""
Guide Generator - Creates step-by-step markdown guides for circuits.

Generates:
- Component placement instructions
- Wiring/connections guide
- Design rationale explanations
- Testing instructions
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..circuit.schemas import Circuit, CircuitComponent, Connection


@dataclass
class GuideSection:
    """A section in the guide"""
    title: str
    content: str
    order: int = 0


class GuideGenerator:
    """
    Generates markdown guides for circuits.
    
    Usage:
        generator = GuideGenerator()
        guide = generator.generate(circuit)
        print(guide)
    """
    
    def __init__(self):
        """Initialize the generator"""
        pass
    
    def generate(self, circuit: Circuit) -> str:
        """
        Generate a complete markdown guide for a circuit.
        
        Args:
            circuit: The Circuit object to generate a guide for
        
        Returns:
            Markdown string with the complete guide
        """
        sections = self._generate_sections(circuit)
        
        # Sort sections by order
        sections.sort(key=lambda s: s.order)
        
        # Build markdown
        markdown_parts = [
            f"# {circuit.name}",
            "",
            circuit.description or "",
            "",
            "---",
            ""
        ]
        
        for section in sections:
            markdown_parts.append(f"## {section.title}")
            markdown_parts.append("")
            markdown_parts.append(section.content)
            markdown_parts.append("")
        
        return "\n".join(markdown_parts)
    
    def _generate_sections(self, circuit: Circuit) -> List[GuideSection]:
        """Generate all guide sections"""
        sections: List[GuideSection] = []
        
        # 1. Overview
        sections.append(self._generate_overview(circuit))
        
        # 2. Components List
        sections.append(self._generate_components_list(circuit))
        
        # 3. Step-by-Step Assembly
        sections.append(self._generate_assembly_steps(circuit))
        
        # 4. Connections Guide
        sections.append(self._generate_connections_guide(circuit))
        
        # 5. Design Rationale
        sections.append(self._generate_design_rationale(circuit))
        
        # 6. Testing Instructions
        sections.append(self._generate_testing_instructions(circuit))
        
        return sections
    
    def _generate_overview(self, circuit: Circuit) -> GuideSection:
        """Generate the overview section"""
        content = f"""
This guide will walk you through building the **{circuit.name}** circuit.

**Circuit Type:** {circuit.metadata.get('circuit_type', 'Custom')}

**Difficulty:** Beginner

**Estimated Time:** {self._estimate_time(circuit)} minutes

**Components Needed:** {len(circuit.components)}

**Connections:** {len(circuit.connections)}
        """
        return GuideSection(
            title="📋 Overview",
            content=content.strip(),
            order=1
        )
    
    def _generate_components_list(self, circuit: Circuit) -> GuideSection:
        """Generate the components list section"""
        content_parts = ["Below are all the components required for this circuit:"]
        content_parts.append("")
        content_parts.append("| ID | Type | Name | Value | Description |")
        content_parts.append("|----|------|------|-------|-------------|")
        
        for comp in circuit.components:
            value = comp.specs.value or "-"
            description = comp.description or "-"
            content_parts.append(
                f"| {comp.id} | {comp.type} | {comp.name} | {value} | {description} |"
            )
        
        content_parts.append("")
        content_parts.append("**Total Components:** " + str(len(circuit.components)))
        
        return GuideSection(
            title="📦 Components List",
            content="\n".join(content_parts),
            order=2
        )
    
    def _generate_assembly_steps(self, circuit: Circuit) -> GuideSection:
        """Generate step-by-step assembly instructions"""
        content_parts = [
            "Follow these steps to assemble your circuit:",
            ""
        ]
        
        for i, comp in enumerate(circuit.components, 1):
            content_parts.append(f"### Step {i}: Place {comp.id} - {comp.name}")
            content_parts.append("")
            content_parts.append(f"- **Component:** {comp.type}")
            content_parts.append(f"- **Value:** {comp.specs.value or 'N/A'}")
            
            if comp.pins:
                content_parts.append(f"- **Pins:** {', '.join(p.name for p in comp.pins)}")
            
            if comp.description:
                content_parts.append(f"- **Purpose:** {comp.description}")
            
            content_parts.append("")
        
        return GuideSection(
            title="🔧 Step-by-Step Assembly",
            content="\n".join(content_parts),
            order=3
        )
    
    def _generate_connections_guide(self, circuit: Circuit) -> GuideSection:
        """Generate the connections guide"""
        if not circuit.connections:
            return GuideGenerator._generate_empty_section("🔗 Connections Guide", "No connections defined.")
        
        content_parts = [
            "Connect the components as follows:",
            ""
        ]
        
        for i, conn in enumerate(circuit.connections, 1):
            content_parts.append(f"### Connection {i}")
            content_parts.append("")
            content_parts.append(f"- **From:** {conn.from_component} - {conn.from_pin}")
            content_parts.append(f"- **To:** {conn.to_component} - {conn.to_pin}")
            content_parts.append(f"- **Type:** {conn.type}")
            
            if conn.description:
                content_parts.append(f"- **Purpose:** {conn.description}")
            
            content_parts.append("")
        
        content_parts.append("---")
        content_parts.append("")
        content_parts.append("**Connection Summary:**")
        content_parts.append("")
        
        # Create a connection table
        content_parts.append("| # | From | To | Type |")
        content_parts.append("|---|------|-----|------|")
        
        for i, conn in enumerate(circuit.connections, 1):
            content_parts.append(
                f"| {i} | {conn.from_component}.{conn.from_pin} | {conn.to_component}.{conn.to_pin} | {conn.type} |"
            )
        
        return GuideSection(
            title="🔗 Connections Guide",
            content="\n".join(content_parts),
            order=4
        )
    
    def _generate_design_rationale(self, circuit: Circuit) -> GuideSection:
        """Generate design rationale explanations"""
        content_parts = [
            "### Why This Design Works",
            "",
            "This circuit has been designed with the following principles:",
            ""
        ]
        
        # Add component-specific rationale
        for comp in circuit.components:
            rationale = self._get_component_rationale(comp)
            if rationale:
                content_parts.append(f"- **{comp.id} ({comp.name})**: {rationale}")
        
        content_parts.append("")
        
        # Add connection rationale
        if circuit.connections:
            content_parts.append("### Connection Logic")
            content_parts.append("")
            
            for conn in circuit.connections:
                rationale = self._get_connection_rationale(conn)
                if rationale:
                    content_parts.append(f"- **{conn.from_component}.{conn.from_pin} → {conn.to_component}.{conn.to_pin}**: {rationale}")
        
        return GuideSection(
            title="💡 Design Rationale",
            content="\n".join(content_parts),
            order=5
        )
    
    def _generate_testing_instructions(self, circuit: Circuit) -> GuideSection:
        """Generate testing instructions"""
        content = f"""
### Before Powering On
1. Double-check all connections against the connections guide
2. Verify component values match the components list
3. Ensure no short circuits exist

### Power-Up Procedure
1. Connect power supply to the circuit
2. Gradually increase voltage to operating level
3. Monitor for any smoke or unusual smells (IMMEDIATELY disconnect if detected)

### Testing {circuit.name}

**Expected Behavior:**
- [ ] Circuit powers on without issues
- [ ] All components receive correct voltage
- [ ] Output signals are as expected

**Troubleshooting:**
- If circuit doesn't work, re-check all connections
- Verify component orientations (diodes, ICs, etc.)
- Use a multimeter to test voltage at key points

**Safety First:**
- Always work in a well-ventilated area
- Use appropriate safety equipment
- Never work on live circuits
        """
        
        return GuideSection(
            title="⚡ Testing Instructions",
            content=content.strip(),
            order=6
        )
    
    def _estimate_time(self, circuit: Circuit) -> int:
        """Estimate assembly time in minutes"""
        # Base time: 5 minutes per component
        time = len(circuit.components) * 5
        
        # Add time for connections: 2 minutes per connection
        time += len(circuit.connections) * 2
        
        # Minimum 10 minutes
        return max(10, time)
    
    def _get_component_rationale(self, comp: CircuitComponent) -> str:
        """Get rationale for a component"""
        rationale_map = {
            "resistor": "Limits current flow to protect other components",
            "capacitor": "Stores and releases electrical energy, filters signals",
            "ic": "Integrated circuit providing specific functionality",
            "transistor": "Amplifies or switches electronic signals",
            "diode": "Allows current to flow in one direction only",
            "inductor": "Stores energy in a magnetic field",
            "battery": "Provides power to the circuit",
            "ground": "Reference point for voltage measurements"
        }
        
        return rationale_map.get(comp.type.lower(), "Essential for circuit operation")
    
    def _get_connection_rationale(self, conn: Connection) -> str:
        """Get rationale for a connection"""
        if conn.type == "power":
            return "Provides power to the component"
        elif conn.type == "ground":
            return "Connects to ground reference"
        elif conn.type == "signal":
            return "Carries signal between components"
        elif conn.type == "clock":
            return "Synchronizes component operations"
        else:
            return "Establishes electrical connection"
    
    @staticmethod
    def _generate_empty_section(title: str, content: str) -> GuideSection:
        """Generate an empty section"""
        return GuideSection(
            title=title,
            content=content,
            order=0
        )


# Singleton instance
_generator: Optional[GuideGenerator] = None


def get_guide_generator() -> GuideGenerator:
    """Get the singleton guide generator"""
    global _generator
    if _generator is None:
        _generator = GuideGenerator()
    return _generator


def reset_guide_generator() -> None:
    """Reset the singleton"""
    global _generator
    _generator = None
