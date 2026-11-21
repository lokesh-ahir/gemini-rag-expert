"""
AI Module - Modular RAG Chatbot with Intent Classification & Model Routing

This module is self-contained and can be extracted for use in other projects.

Features:
- Intent classification (general_query, edit_schematic, create_design, debug_circuit)
- Model routing (Gemini Flash for general, Gemini Neon 4 for design tasks)
- LLM Agent with tool calling (LangChain/LangGraph)
- Circuit building from Qdrant components (1800+ components)
- Markdown guide generation
- PostgreSQL chat history (workspace_id, design_id based)
- Cost tracking

API Endpoints (only 2):
- POST /ai/chat - Send message, get AI response with intent classification and tool calling
- GET /ai/chat/{design_id} - Retrieve chat messages (paginated, 25 per page)

Usage:
    from ai_module import AIModule, get_ai_module
    
    module = AIModule(
        db_url="postgresql://user:pass@localhost/db",
        qdrant_url="http://localhost:6333",
        google_api_key="your_key"
    )
    
    # Chat with LLM + tool calling
    response = module.chat_with_agent(
        workspace_id="...",
        design_id="...", 
        query="Create an amplifier with LM386"
    )
    
    # Get chat history
    messages, total, pages = module.get_chat_messages(design_id="...", page=1, limit=25)
"""

__version__ = "1.0.0"

# Main module
from .main import AIModule, get_ai_module, reset_ai_module

# Sub-modules (internal, not exposed via API)
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
]
