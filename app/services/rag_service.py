"""
Step 4 — RAG & Semantic Search Service
Provides:
  - search_products: cosine-distance pgvector search
  - get_recommendations: GraphRAG cross-sell via ItemRelations table
"""
from __future__ import annotations

import logging
from uuid import UUID

import google.generativeai as genai
from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.schemas import ProductResponse

logger = logging.getLogger("rag_service")

_EMBED_MODEL = "models/text-embedding-004"

genai.configure(api_key=settings.GEMINI_API_KEY)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

async def _embed_query(query: str) -> str:
    """
    Generate a Gemini embedding for a single query string and return it
    as a pgvector-compatible literal string: '[0.1,0.2,...]'.
    """
    try:
        response = genai.embed_content(
            model=_EMBED_MODEL,
            content=query,
            task_type="retrieval_query",
        )
        vector: list[float] = response["embedding"]
        return "[" + ",".join(map(str, vector)) + "]"
    except Exception as exc:
        logger.error("Gemini embedding error: %s", exc)
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
    limit: int = 3,
) -> list[ProductResponse]:
    """
    Cosine-distance semantic search on MenuItems.Embedding (pgvector).
    Returns up to `limit` available, non-sold-out products ordered by relevance.
    """
    vector_literal = await _embed_query(query)

    sql = text(
        """
        SELECT "Id"::text          AS id,
               "Name"              AS name,
               "PriceAmount"::text AS price,
               "ImageUrl"          AS image_url
        FROM   "MenuItems"
        WHERE  "IsAvailable"  = true
          AND  "IsSoldOut"    = false
          AND  "Embedding"    IS NOT NULL
        ORDER BY "Embedding" <=> :vector::vector
        LIMIT  :limit
        """
    )

    try:
        result = await db.execute(sql, {"vector": vector_literal, "limit": limit})
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
               m."PriceAmount"::text AS price,
               m."ImageUrl"          AS image_url
        FROM   "ItemRelations" r
        JOIN   "MenuItems"     m ON r."TargetItemId" = m."Id"
        WHERE  r."SourceItemId" = ANY(:source_ids::uuid[])
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
