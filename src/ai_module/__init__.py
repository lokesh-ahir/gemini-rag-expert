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
"""

__version__ = "1.0.0"
