import os
# pyrefly: ignore [missing-import]
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.services.rag_service import RagService, ChatRequest, IngestRequest, ChatAssistantResponse

# Load environment variables
load_dotenv()

app = FastAPI(
    title="DineX RAG AI Service",
    description="Python FastAPI service for DineX AI Assistant RAG pipeline.",
    version="1.0.0"
)

# Enable CORS for gateway access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_service = RagService()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "gemini_enabled": os.getenv("GEMINI_API_KEY") is not None}

@app.post("/chat", response_model=ChatAssistantResponse)
async def chat(request: ChatRequest):
    try:
        response = await rag_service.generate_chat_response(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating chat response: {str(e)}")

@app.post("/ingest")
async def ingest(request: IngestRequest):
    success = await rag_service.ingest_documents(request.branch_id, request.documents)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to ingest documents.")
    return {"status": "success", "message": f"Successfully ingested {len(request.documents)} documents."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
