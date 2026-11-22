"""
Agent Factory - Creates LLM agents with tool calling capabilities.

Uses LangChain for tool calling integration.
"""

from typing import Optional, Dict, Any, List, Callable
from enum import Enum
import os

try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.tools import tool
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.graph import Graph
    from langgraph.prebuilt import ToolNode
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from ..intent.schemas import IntentType
from ..routing.model_router import ModelType
from ..config import get_config


class AgentType(str, Enum):
    """Types of agents available"""
    GENERAL = "general"           # For general queries (uses Flash)
    DESIGN = "design"             # For design tasks (uses Neon 4 with tools)
    DEBUG = "debug"               # For debugging (uses Neon 4 with tools)


class AgentFactory:
    """
    Factory for creating LLM agents with tool calling.
    
    Usage:
        factory = AgentFactory()
        
        # Create a design agent with tools
        design_agent = factory.create_agent(
            agent_type=AgentType.DESIGN,
            tools=[search_tool, build_tool, guide_tool]
        )
        
        # Use the agent
        response = design_agent.invoke({"messages": [HumanMessage(content="Create an amplifier")]})
    """
    
    def __init__(self):
        """Initialize the factory"""
        self.config = get_config()
        self._agents: Dict[str, Any] = {}
    
    def create_agent(
        self,
        agent_type: AgentType,
        tools: Optional[List[Callable]] = None,
        system_prompt: Optional[str] = None
    ) -> Any:
        """
        Create an agent with tool calling.
        
        Args:
            agent_type: Type of agent (determines model and behavior)
            tools: List of tool functions
            system_prompt: Custom system prompt
        
        Returns:
            LangGraph agent with tool calling
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is required for agent functionality. "
                "Install with: pip install langchain langchain-google-genai langgraph"
            )
        
        # Select model based on agent type
        model = self._get_model(agent_type)
        
        # Get default system prompt
        if system_prompt is None:
            system_prompt = self._get_system_prompt(agent_type)
        
        # Create the agent
        if tools:
            return self._create_tool_calling_agent(model, tools, system_prompt)
        else:
            return self._create_basic_agent(model, system_prompt)
    
    def _get_model(self, agent_type: AgentType) -> ChatGoogleGenerativeAI:
        """Get the appropriate LLM model for the agent type"""
        
        model_config = {
            AgentType.GENERAL: ModelType.GEMINI_FLASH,
            AgentType.DESIGN: ModelType.GEMINI_NEON_4,
            AgentType.DEBUG: ModelType.GEMINI_NEON_4
        }
        
        model_type = model_config.get(agent_type, ModelType.GEMINI_FLASH)
        model_id = self.config.models.gemini_flash_id
        
        if model_type == ModelType.GEMINI_NEON_4:
            model_id = self.config.models.gemini_neon4_id
        
        return ChatGoogleGenerativeAI(
            model=model_id,
            google_api_key=self.config.models.google_api_key,
            temperature=self.config.models.temperature,
            max_tokens=self.config.models.max_tokens
        )
    
    def _get_system_prompt(self, agent_type: AgentType) -> str:
        """Get the system prompt for the agent type"""
        
        prompts = {
            AgentType.GENERAL: """
            You are a helpful AI assistant. Answer general questions directly and concisely.
            Do not use tools unless specifically asked to search for information.
            """,
            AgentType.DESIGN: """
            You are Cliff, a professional Embedded Systems Engineer and Circuit Design Expert.
            
            **Your Role:**
            You are an expert AI assistant specializing in embedded systems, circuit design, and PCB development.
            Your primary function is to assist engineers, hobbyists, and students in designing, analyzing,
            debugging, and optimizing electronic circuits and embedded systems.
            
            **Capabilities:**
            - Search and retrieve circuit components from the vector database
            - Design complete circuit schematics from requirements
            - Generate step-by-step assembly and wiring guides
            - Debug and troubleshoot circuit issues
            - Explain electronic concepts and component behavior
            - Provide best practices for PCB design and embedded systems
            
            **When the user asks to create, edit, or debug a circuit:**
            1. First, understand the requirements and constraints
            2. Search for relevant components using the search_components tool
            3. Design or modify the circuit using the build_circuit tool
            4. Generate a comprehensive assembly guide using the generate_guide tool
            5. Provide clear, actionable instructions
            
            **Design Guidelines:**
            - Always prioritize safety and reliability
            - Follow industry standards (IPC, IEEE, etc.)
            - Consider power consumption, thermal management, and EMI
            - Use appropriate component values and tolerances
            - Provide clear component placement and routing suggestions
            
            **Response Format:**
            - For circuit creation: Provide circuit JSON, markdown guide, and explanation
            - For circuit editing: Show what was changed and why
            - For debugging: Explain the issue and suggest solutions
            - For general questions: Provide clear, concise, and accurate answers
            
            **Safety First:**
            - Always warn about high voltage/current dangers
            - Advise proper safety equipment (ESD protection, insulated tools)
            - Never suggest unsafe practices
            - Remind users to double-check connections before powering on
            
            **Technical Specifications:**
            - Use standard electronic symbols and notation
            - Provide component values in standard units (kΩ, µF, mA, etc.)
            - Include tolerances and ratings when relevant
            - Reference datasheets when specific component behavior is discussed
            
            **Always use the available tools** to help with design, editing, and debugging tasks.
            Do not guess or hallucinate component specifications - always retrieve from the database.
            """,
            AgentType.DEBUG: """
            You are Cliff, a professional Embedded Systems Engineer specializing in Circuit Debugging.
            
            **Your Role:**
            You are an expert troubleshooter for electronic circuits and embedded systems.
            Your primary function is to help users identify, diagnose, and resolve issues in their circuits.
            
            **Capabilities:**
            - Analyze circuit designs for potential issues
            - Search for component specifications and behavior
            - Identify common failure modes and symptoms
            - Suggest systematic debugging approaches
            - Provide test procedures and expected results
            
            **Debugging Process:**
            1. Ask clarifying questions about symptoms and observations
            2. Search for relevant component information using search_components tool
            3. Analyze the circuit design for potential issues
            4. Suggest specific tests and measurements
            5. Provide step-by-step troubleshooting guide
            
            **Common Issues to Check:**
            - Power supply issues (voltage levels, ripple, noise)
            - Ground loops and improper grounding
            - Component orientation and polarity
            - Incorrect component values
            - Signal integrity issues
            - Thermal problems
            - ESD damage
            - Manufacturing defects
            
            **Testing Guidance:**
            - Always start with visual inspection
            - Check power rails first
            - Use multimeter for DC measurements
            - Use oscilloscope for signal analysis
            - Check for shorts and opens
            - Verify component placement
            
            **Safety Reminders:**
            - Always work on de-energized circuits when possible
            - Use proper safety equipment
            - Be cautious with high voltage circuits
            - Never work on live high-voltage circuits
            
            **Always use the available tools** to retrieve component information and circuit details.
            """
        }
        
        return prompts.get(agent_type, prompts[AgentType.GENERAL])
    
    def _create_tool_calling_agent(
        self,
        model: ChatGoogleGenerativeAI,
        tools: List[Callable],
        system_prompt: str
    ) -> Any:
        """Create an agent with tool calling using LangGraph"""
        
        # Bind tools to the model
        model_with_tools = model.bind_tools(tools)
        
        # Create the graph
        graph = Graph()
        
        # Add nodes
        graph.add_node("agent", self._create_agent_node(model_with_tools, system_prompt))
        graph.add_node("tools", ToolNode(tools))
        
        # Define edges
        graph.add_edge("agent", "tools")
        graph.add_edge("tools", "agent")
        
        # Set entry point
        graph.set_entry_point("agent")
        
        return graph.compile()
    
    def _create_agent_node(self, model: Any, system_prompt: str) -> Callable:
        """Create the agent node for the graph"""
        
        def agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
            messages = state.get("messages", [])
            
            # Add system prompt
            messages = [SystemMessage(content=system_prompt)] + messages
            
            # Get response from model
            response = model.invoke(messages)
            
            return {"messages": [response]}
        
        return agent_node
    
    def _create_basic_agent(
        self,
        model: ChatGoogleGenerativeAI,
        system_prompt: str
    ) -> Any:
        """Create a basic agent without tools"""
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        return prompt | model
    
    def create_intent_aware_agent(
        self,
        intent: IntentType,
        tools: Optional[List[Callable]] = None
    ) -> Any:
        """
        Create an agent based on intent.
        
        Args:
            intent: The classified intent
            tools: Optional tools to bind
        
        Returns:
            Agent configured for the intent
        """
        intent_to_agent = {
            "general_query": AgentType.GENERAL,
            "edit_schematic": AgentType.DESIGN,
            "create_design": AgentType.DESIGN,
            "debug_circuit": AgentType.DEBUG
        }
        
        agent_type = intent_to_agent.get(intent, AgentType.GENERAL)
        return self.create_agent(agent_type, tools)


# Singleton instance
_factory: Optional[AgentFactory] = None


def get_agent_factory() -> AgentFactory:
    """Get the singleton agent factory"""
    global _factory
    if _factory is None:
        _factory = AgentFactory()
    return _factory


def reset_agent_factory() -> None:
    """Reset the singleton"""
    global _factory
    _factory = None
