from typing import Any, Dict
from deepagents import create_deep_agent
from src.core.container import ServiceContainer
from src.core.tool_registry import ToolRegistry
from src.core.config import Config
from src.plugins.tools.retriever import RetrieverTool


class AgentService:
    """Service for managing agent interactions"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._agent = None
        self._tools_initialized = False

    def _initialize_tools(self):
        """Initialize tools for the agent"""
        if not self._tools_initialized:
            # Register retriever tool with user_id
            retriever_tool = RetrieverTool()
            ToolRegistry.register(retriever_tool)
            self._tools_initialized = True

    @property
    def agent(self):
        """Lazy load agent"""
        if self._agent is None:
            self._initialize_tools()
            
            # Get all registered tools
            tools = ToolRegistry.get_all_tools()
            
            # Create agent
            self._agent = create_deep_agent(
                model=f"google_genai:{Config.GEMINI_MODEL}",
                tools=tools,
                system_prompt=Config.SYSTEM_PROMPT + f"\n\nYou are currently assisting User: {self.user_id}."
            )

        return self._agent

    def ask(self, question: str) -> str:
        """Ask the agent a question"""
        try:
            response = self.agent.invoke({
                "messages": [{"role": "user", "content": question}]
            })
            last_message = response["messages"][-1]
            content = last_message.content if hasattr(last_message, 'content') else str(last_message)
            return content
        except Exception as e:
            raise RuntimeError(f"Agent Error: {str(e)}")
