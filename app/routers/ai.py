"""
Step 5c — AI Router
Exposes:
  POST /api/ai/chat    → chat with Gemini RAG pipeline
  POST /api/ai/ingest  → embed MenuItems into pgvector
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, AliasChoices
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.schemas import ChatResponse
from app.services.chat_service import handle_chat
from app.services.ingest_service import ingest_menu_embeddings

logger = logging.getLogger("router.ai")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if settings.INTERNAL_API_KEY and api_key == settings.INTERNAL_API_KEY:
        return api_key
    raise HTTPException(
        status_code=403,
        detail="Truy cập bị từ chối: X-API-Key không hợp lệ!"
    )


router = APIRouter(prefix="/api/ai", tags=["AI"], dependencies=[Depends(verify_api_key)])


# ---------------------------------------------------------------------------
# Request models (internal to this router only)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Nội dung tin nhắn từ người dùng, giới hạn 2000 ký tự để chống DoS/Buffer Overflow."
    )
    session_id: Optional[str] = Field(None, validation_alias=AliasChoices("session_id", "sessionId"))
    branch_id: Optional[str] = Field(None, validation_alias=AliasChoices("branch_id", "branchId"))
    chat_history: list[dict] = Field([], validation_alias=AliasChoices("chat_history", "chatHistory"))
    chat_cart: list[dict] = Field([], validation_alias=AliasChoices("chat_cart", "chatCart"))


class IngestResponse(BaseModel):
    status: str
    processed_count: int


# ---------------------------------------------------------------------------
# POST /api/ai/chat
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse, summary="RAG Chat with Gemini")
async def chat_endpoint(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """
    Accepts a user message, runs the full RAG pipeline (pgvector search +
    Gemini tool-call loop), and returns a strict ChatResponse.
    """
    try:
        return await handle_chat(
            db=db,
            message=request.message,
            branch_id=request.branch_id,
            chat_history=request.chat_history,
            session_id=request.session_id,
            chat_cart=request.chat_cart,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unhandled error in /api/ai/chat")
        raise HTTPException(status_code=500, detail=f"Chat pipeline error: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /api/ai/ingest
# ---------------------------------------------------------------------------

@router.post("/ingest", response_model=IngestResponse, summary="Ingest MenuItems embeddings")
async def ingest_endpoint(
    db: AsyncSession = Depends(get_db),
) -> IngestResponse:
    """
    Queries all MenuItems with NULL Embedding, generates vectors via Gemini,
    and bulk-upserts them into the pgvector column.
    """
    try:
        processed_count = await ingest_menu_embeddings(db)
        return IngestResponse(status="success", processed_count=processed_count)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unhandled error in /api/ai/ingest")
        raise HTTPException(status_code=500, detail=f"Ingestion error: {exc}") from exc
