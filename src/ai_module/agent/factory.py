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
            You are an expert circuit design assistant. You can:
            - Search for circuit components
            - Build circuit designs
            - Generate assembly guides
            
            When the user asks to create, edit, or debug a circuit:
            1. First, search for relevant components
            2. Then build the circuit
            3. Finally, generate a guide
            
            Always use the available tools to help with design tasks.
            """,
            AgentType.DEBUG: """
            You are a circuit debugging expert. You can:
            - Search for components to understand their behavior
            - Analyze circuit designs
            - Suggest fixes for common issues
            
            When debugging, use tools to fetch component information and circuit details.
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
