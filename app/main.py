"""
DineX-AI Service — Entry Point (Phase 2)
"""
import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers.ai import router as ai_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
)
logger = logging.getLogger("main")

app = FastAPI(
    title="DineX RAG AI Service",
    description="FastAPI microservice — Gemini RAG pipeline with pgvector + GraphRAG. Sits behind .NET 10 API Gateway.",
    version="2.0.0",
)

# ---------------------------------------------------------------------------
# CORS — allow Gateway and local dev origins
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # tighten to gateway domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(ai_router)

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    return {
        "status": "healthy",
        "version": "2.0.0",
        "gemini_enabled": bool(settings.GEMINI_API_KEY),
        "db_url_prefix": settings.DATABASE_URL[:35] + "...",
    }


# ---------------------------------------------------------------------------
# Local dev runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
