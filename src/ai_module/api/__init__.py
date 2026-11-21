"""
API Module - FastAPI endpoints for the AI Module.

Only 2 endpoints:
- POST /ai/chat - Send a chat message and get response (with intent classification, tool calling, etc.)
- GET /ai/chat/{design_id} - Retrieve chat messages for a design

To use:
    from fastapi import FastAPI
    from ai_module.api.router import router as ai_router
    
    app = FastAPI()
    app.include_router(ai_router, prefix="/api")
    
    # Now you have:
    # POST /api/ai/chat - Chat with AI
    # GET /api/ai/chat/{design_id} - Get chat history
"""
