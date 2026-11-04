from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from pydantic import BaseModel
from src.core.container import ServiceContainer
from src.core.file_processor_registry import FileProcessorRegistry
from src.services.agent_service import AgentService

router = APIRouter()

class ChatRequest(BaseModel):
    query: str

class UserResponse(BaseModel):
    user_id: str
    message: str


@router.post("/create-user", response_model=UserResponse)
async def create_user(user_id: str):
    try:
        vector_store = ServiceContainer.get_service("vector_store")
        collection_name = f"user_{user_id}"
        vector_store.ensure_collection(collection_name)
        return {"user_id": user_id, "message": f"User {user_id} initialized successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-files/{user_id}")
async def upload_files(user_id: str, files: List[UploadFile] = File(...)):
    try:
        vector_store = ServiceContainer.get_service("vector_store")
        collection_name = f"user_{user_id}"
        
        texts = []
        metadatas = []
        
        for file in files:
            content = await file.read()
            processor = FileProcessorRegistry.get_processor(file.filename)
            text = processor.extract_text(content, file.filename)
            if text.strip():
                texts.append(text)
                metadatas.append({"filename": file.filename, "user_id": user_id})
        
        if texts:
            vector_store.add_documents(collection_name, texts, metadatas)
        
        return {"message": f"Successfully processed {len(files)} files for user {user_id}"}
    except Exception as e:
        print(f"Error uploading files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/{user_id}")
async def chat(user_id: str, request: ChatRequest):
    print(f"Received chat request for user: {user_id}")
    print(f"Query: {request.query}")
    try:
        agent_service = AgentService(user_id)
        answer = agent_service.ask(request.query)
        return {"answer": answer}
    except Exception as e:
        print(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health():
    return {"status": "healthy", "service": "rag-chatbot"}
