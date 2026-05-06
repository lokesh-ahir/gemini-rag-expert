# 🚀 gemini-rag-expert (2026 Edition)

A high-performance, multi-user RAG (Retrieval-Augmented Generation) chatbot built with **FastAPI**, **Qdrant**, and **Google Gemini 2.0**. This system uses **Agentic Reasoning** to perform deep research across your documents.

## ✨ Key Features

- **Agentic Search**: Uses the `DeepAgents` harness to perform multi-step planning and research loops.
- **Gemini 2.0 Integration**: Powered by the latest `gemini-3.1-pro-preview` model.
- **High-Precision Embeddings**: Uses `gemini-embedding-2` with 3072-dimensional vectors for superior retrieval.
- **Multi-User Isolation**: Each user gets a dedicated, isolated Qdrant collection for privacy.
- **Smart Ingestion**: Automatic PDF/Docx/Text processing with reliable batch-ingestion logic.
- **Production Logs**: Detailed server-side tracking for document extraction and agent reasoning.

## 🛠️ Tech Stack

- **Framework**: FastAPI
- **LLM**: Google Gemini 3.1 Pro
- **Embeddings**: Google Gemini Embedding 2
- **Vector DB**: Qdrant
- **Agent Framework**: LangGraph / DeepAgents
- **Language**: Python 3.11+

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.11+
- Qdrant (Cloud or Local)
- Google AI Studio API Key

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/your-username/gemini-rag-expert.git
cd gemini-rag-expert

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_google_api_key
GEMINI_MODEL=gemini-3.1-pro-preview
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
SYSTEM_PROMPT="You are Cliff, a professional AI Research Assistant..."
```

### 4. Running the App
```bash
uvicorn src.main:app --reload --port 8005
```

## 📡 API Endpoints

- `POST /api/upload-files/{user_id}`: Upload PDFs/Docs for a specific user.
- `POST /api/chat/{user_id}`: Chat with the agent about uploaded documents.
- `GET /api/health`: Check service status.

## 📄 License
MIT License
