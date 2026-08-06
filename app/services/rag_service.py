"""
Step 4 — RAG & Semantic Search Service
Provides:
  - search_products: cosine-distance pgvector search
  - get_recommendations: GraphRAG cross-sell via ItemRelations table
"""
from __future__ import annotations

import logging
import asyncio
from uuid import UUID

import google.generativeai as genai
from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.schemas import ProductResponse

logger = logging.getLogger("rag_service")

_EMBED_MODEL = "models/text-embedding-004"

import os

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or settings.GEMINI_API_KEY
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is missing.")

genai.configure(api_key=api_key)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

async def _embed_query(query: str) -> str:
    """
    Generate a Gemini embedding for a single query string and return it
    as a pgvector-compatible literal string: '[0.1,0.2,...]'.
    Sử dụng asyncio.to_thread để ngăn chặn Blocking Event Loop của FastAPI.
    """
    try:
        response = await asyncio.to_thread(
            genai.embed_content,
            model=_EMBED_MODEL,
            content=query,
            task_type="retrieval_query",
            output_dimensionality=768,
        )
        vector: list[float] = response["embedding"]
        return "[" + ",".join(map(str, vector)) + "]"
    except Exception as exc:
        logger.error("[RAG Service] Lỗi khi tạo embedding: %s", exc)
        raise HTTPException(status_code=502, detail=f"Embedding generation failed: {exc}") from exc


def _row_to_product(row: dict) -> ProductResponse:
    return ProductResponse(
        id=UUID(row["id"]),
        name=row["name"],
        price=float(row["price"]),
        image_url=row.get("image_url"),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def search_products(
    db: AsyncSession,
    query: str,
    category_id: str | None = None,
    branch_id: str | None = None,
    limit: int = 3,
    distance_limit: float = 0.65,
) -> list[ProductResponse]:
    """
    Cosine-distance semantic search on MenuItems.Embedding (pgvector) with distance threshold.
    Supports pre-filtering by CategoryId and BranchId (via BranchMenuItems).
    Returns up to `limit` available, non-sold-out products ordered by relevance.
    """
    vector_literal = await _embed_query(query)

    sql = text(
        """
        SELECT m."Id"::text          AS id,
               m."Name"              AS name,
               m."BasePrice"::text   AS price,
               m."ImageUrl"          AS image_url
        FROM   "MenuItems" m
        WHERE  m."IsAvailable"  = true
          AND  m."Embedding"    IS NOT NULL
          AND  (:category_id IS NULL OR m."CategoryId" = CAST(:category_id AS uuid))
          AND  (:branch_id IS NULL OR EXISTS (
               SELECT 1 FROM "BranchMenuItems" bmi 
               WHERE bmi."MenuItemId" = m."Id" 
                 AND bmi."BranchId" = CAST(:branch_id AS uuid) 
                 AND bmi."IsActive" = true 
                 AND bmi."IsSoldOut" = false
          ))
          AND  (m."Embedding" <=> CAST(:vector AS vector)) <= :distance_limit
        ORDER BY m."Embedding" <=> CAST(:vector AS vector)
        LIMIT  :top_k
        """
    )

    try:
        result = await db.execute(
            sql,
            {
                "vector": vector_literal,
                "top_k": limit,
                "distance_limit": distance_limit,
                "category_id": category_id,
                "branch_id": branch_id,
            },
        )
        rows = result.mappings().all()
    except Exception as exc:
        logger.error("search_products DB error: %s", exc)
        raise HTTPException(status_code=500, detail="Product search failed.") from exc

    return [_row_to_product(dict(r)) for r in rows]


async def get_recommendations(
    db: AsyncSession,
    source_item_ids: list[str],
    limit: int = 2,
) -> list[ProductResponse]:
    """
    GraphRAG cross-sell/up-sell lookup via ItemRelations.
    Returns up to `limit` related available products ordered by relation Weight.
    """
    if not source_item_ids:
        return []

    sql = text(
        """
        SELECT m."Id"::text          AS id,
               m."Name"              AS name,
               m."BasePrice"::text   AS price,
               m."ImageUrl"          AS image_url
        FROM   "ItemRelations" r
        JOIN   "MenuItems"     m ON r."TargetItemId" = m."Id"
        WHERE  r."SourceItemId" = ANY(CAST(:source_ids AS uuid[]))
          AND  m."IsAvailable"  = true
        ORDER BY r."Weight" DESC
        LIMIT  :limit
        """
    )

    try:
        result = await db.execute(
            sql,
            {"source_ids": source_item_ids, "limit": limit},
        )
        rows = result.mappings().all()
    except Exception as exc:
        logger.error("get_recommendations DB error: %s", exc)
        raise HTTPException(status_code=500, detail="Recommendation lookup failed.") from exc

    return [_row_to_product(dict(r)) for r in rows]
