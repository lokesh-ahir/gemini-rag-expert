"""
AI Module - Modular RAG Chatbot with Intent Classification & Model Routing

This module is self-contained and can be extracted for use in other projects.
Contains:
- Intent classification (general_query, edit_schematic, create_design, debug_circuit)
- Model routing (Gemini Flash for general, Gemini Neon 4 for design tasks)
- Circuit building from Qdrant components
- Markdown guide generation
- PostgreSQL chat history (workspace_id, design_id based)
- RAG integration for circuit components
- LLM Agent with tool calling (LangChain/LangGraph)
- FastAPI endpoints for all functionality

Quick Start:
    from ai_module import AIModule, get_ai_module
    
    module = AIModule(
        db_url="postgresql://user:pass@localhost/db",
        qdrant_url="http://localhost:6333",
        google_api_key="your_key"
    )
    
    # Use LLM agent with tool calling
    response = module.chat_with_agent(
        workspace_id="...",
        design_id="...",
        query="Create an amplifier with LM386"
    )

To use the API endpoints:
    from ai_module.api.router import router as ai_router
    app.include_router(ai_router, prefix="/api/ai")
"""

__version__ = "1.0.0"

# Main module
from .main import AIModule, get_ai_module, reset_ai_module

# Sub-modules
from . import config
from . import intent
from . import routing
from . import rag
from . import circuit
from . import guide
from . import chat
from . import agent
from . import tools
from . import api

# Exports
__all__ = [
    "AIModule",
    "get_ai_module",
    "reset_ai_module",
    "config",
    "intent",
    "routing",
    "rag",
    "circuit",
    "guide",
    "chat",
    "agent",
    "tools",
    "api",
]
