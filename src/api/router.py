from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from src.services.qdrant_service import QdrantService
from src.services.gemini_service import GeminiService
from src.utils.file_processor import FileProcessor
from pydantic import BaseModel

router = APIRouter()
qdrant_service = QdrantService()

class ChatRequest(BaseModel):
    query: str

class UserResponse(BaseModel):
    user_id: str
    message: str

@router.post("/create-user", response_model=UserResponse)
async def create_user(user_id: str):
    try:
        qdrant_service.ensure_collection(user_id)
        return {"user_id": user_id, "message": f"User {user_id} initialized successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-files/{user_id}")
async def upload_files(user_id: str, files: List[UploadFile] = File(...)):
    try:
        for file in files:
            content = await file.read()
            text = FileProcessor.extract_text(content, file.filename)
            if text.strip():
                qdrant_service.add_documents(user_id, text, {"filename": file.filename})
        
        return {"message": f"Successfully processed {len(files)} files for user {user_id}"}
    except Exception as e:
        print(f"Error uploading files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/{user_id}")
async def chat(user_id: str, request: ChatRequest):
    print(f"Received chat request for user: {user_id}")
    print(f"Query: {request.query}")
    try:
        gemini = GeminiService(user_id)
        answer = gemini.ask(request.query)
        return {"answer": answer}
    except Exception as e:
        print(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health():
    return {"status": "healthy", "service": "rag-chatbot"}
