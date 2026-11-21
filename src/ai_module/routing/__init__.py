"""
Model Routing Module
Routes intents to appropriate models for cost optimization.
- general_query → Gemini Flash (cheaper)
- edit_schematic/create_design/debug_circuit → Gemini Neon 4 (more capable)
"""
