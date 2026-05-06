import uvicorn
from fastapi import FastAPI
from src.api.router import router as rag_router
from src.core.config import Config

def create_app():
    Config.validate()
    
    app = FastAPI(title="RAG Chatbot with Gemini & Qdrant", version="1.0.0")
    
    app.include_router(rag_router, prefix="/api")

    @app.get("/")
    async def root():
        return {"message": "RAG Chatbot API is running"}

    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=Config.PORT, reload=True)
