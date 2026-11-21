import uvicorn
from fastapi import FastAPI
from src.api.router import router as rag_router
from src.core.config import Config
from src.core.container import ServiceContainer
from src.core.file_processor_registry import FileProcessorRegistry
from src.core.tool_registry import ToolRegistry
from src.plugins.tools.retriever import RetrieverTool

def create_app():
    # Validate configuration
    Config.validate()
    
    # Configure dependency injection container
    ServiceContainer.configure(Config.SERVICE_CONFIG)
    
    # Initialize registries
    FileProcessorRegistry.load_default_processors()
    
    # Register default tools
    ToolRegistry.register(RetrieverTool())
    
    # Create FastAPI app
    app = FastAPI(title="RAG Chatbot with Gemini & Qdrant", version="1.0.0")
    
    # Include routers
    app.include_router(rag_router, prefix="/api")
    
    # Mount AI Module endpoints (if available)
    try:
        from src.ai_module.api.router import router as ai_router
        app.include_router(ai_router, prefix="/api")
    except ImportError:
        # AI module not available, skip
        pass

    @app.get("/")
    async def root():
        return {"message": "RAG Chatbot API is running"}

    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=Config.PORT, reload=True)
