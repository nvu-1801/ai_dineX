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

from app.services.key_manager import call_llm_api_with_fallback


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

async def _embed_query(query: str) -> str:
    """
    Generate a Gemini embedding for a single query string and return it
    as a pgvector-compatible literal string: '[0.1,0.2,...]'.
    Runs with automatic key fallback.
    """
    def _do_embed():
        return genai.embed_content(
            model=_EMBED_MODEL,
            content=query,
            task_type="retrieval_query",
            output_dimensionality=768,
        )

    try:
        response = await call_llm_api_with_fallback(_do_embed)
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


import re

def extract_food_query(text: str) -> str:
    """
    Strips conversational context prefixes (e.g. 'tôi muốn ăn', 'cho tôi món', 'tôi thèm')
    and suffix filler words to extract the core dish/food search query.
    """
    if not text:
        return ""

    cleaned = text.strip()

    prefix_patterns = [
        r"^(dạ\s+)?(cho\s+)?(tôi|mình|em|anh|chị)\s+hỏi\s*",
        r"^(bên\s+mình|bên\s+em|bên\s+anh|bên\s+quán|quán|cửa\s+hàng|nhà\s+hàng|ở\s+đây)\s*(có\s+bán|có|bán)?\s*",
        r"^(tôi|mình|em|anh|chị)\s+(muốn|thèm|cần)\s+(ăn|uống|gọi|mua|đặt|tìm)?\s*(món|đồ\s+ăn|đồ\s+uống)?\s*",
        r"^(tôi|mình|em|anh|chị)\s+(muốn|thèm|cần)\s*",
        r"^(cho\s+)?(tôi|mình|em|anh|chị)\s+(ăn|uống|gọi|xin|mua|đặt)?\s*(món|đồ\s+ăn|đồ\s+uống)?\s*",
        r"^(cho\s+xin|bán\s+cho|đặt\s+cho)\s+(tôi|mình|em|anh|chị)?\s*(món|đồ\s+ăn|đồ\s+uống)?\s*",
        r"^(muốn|thèm)\s+(ăn|uống|món)?\s*",
        r"^(tư\s+vấn|gợi\s+ý|tìm\s+kiếm|cần\s+tìm|tìm\s+món|tìm)\s+(giúp\s+)?(tôi|mình|em|anh|chị)?\s*(món|đồ\s+ăn|đồ\s+uống)?\s*",
        r"^(có\s+bán|có|bán)\s+(món)?\s*",
        r"^(cho\s+1|cho\s+2|cho\s+3|cho\s+\d+)\s*",
    ]

    changed = True
    while changed:
        changed = False
        for pat in prefix_patterns:
            subbed = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()
            if subbed != cleaned and subbed:
                cleaned = subbed
                changed = True

    suffix_patterns = [
        r"\s+(không\s+ạ|khong\s+a|ko\s+a|không|khong|ko|nhé|nhe|nha|ạ|a|với|voi|giúp tôi|giúp mình|giúp em|hỏi với|với ạ)\??$",
    ]
    for pat in suffix_patterns:
        subbed = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()
        if subbed:
            cleaned = subbed

    # Strip leading quantities e.g. "2 tô ", "1 suất "
    qty_sub = re.sub(r"^\d+\s*(tô|bát|dĩa|phần|suất|ly|cốc)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
    if qty_sub:
        cleaned = qty_sub

    return cleaned if cleaned else text.strip()


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
    clean_query = extract_food_query(query)
    logger.info("[search_products] Raw query: '%s' -> Extracted food query: '%s'", query, clean_query)
    
    vector_literal = await _embed_query(clean_query if clean_query else query)
    text_pattern = f"%{(clean_query if clean_query else query).lower()}%"

    sql = text(
        """
        SELECT m."Id"::text          AS id,
               m."Name"              AS name,
               m."BasePrice"::text   AS price,
               m."ImageUrl"          AS image_url
        FROM   "MenuItems" m
        WHERE  m."IsAvailable"  = true
          AND  (:category_id IS NULL OR m."CategoryId" = CAST(:category_id AS uuid))
          AND  (:branch_id IS NULL OR EXISTS (
               SELECT 1 FROM "BranchMenuItems" bmi 
               WHERE bmi."MenuItemId" = m."Id" 
                 AND bmi."BranchId" = CAST(:branch_id AS uuid) 
                 AND bmi."IsActive" = true 
                 AND bmi."IsSoldOut" = false
          ))
          AND  (
               (m."Embedding" IS NOT NULL AND (m."Embedding" <=> CAST(:vector AS vector)) <= :distance_limit)
               OR LOWER(m."Name") LIKE :text_pattern
          )
        ORDER BY 
          CASE WHEN LOWER(m."Name") LIKE :text_pattern THEN 0 ELSE 1 END,
          CASE WHEN m."Embedding" IS NOT NULL THEN (m."Embedding" <=> CAST(:vector AS vector)) ELSE 1.0 END
        LIMIT  :top_k
        """
    )

    try:
        result = await db.execute(
            sql,
            {
                "vector": vector_literal,
                "text_pattern": text_pattern,
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
    branch_id: str | None = None,
    limit: int = 2,
) -> list[ProductResponse]:
    """
    GraphRAG cross-sell/up-sell lookup via ItemRelations.
    Returns up to `limit` related available products ordered by relation Weight.
    Supports pre-filtering by BranchId if branch_id is provided.
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
          AND  (:branch_id IS NULL OR EXISTS (
               SELECT 1 FROM "BranchMenuItems" bmi 
               WHERE bmi."MenuItemId" = m."Id" 
                 AND bmi."BranchId" = CAST(:branch_id AS uuid) 
                 AND bmi."IsActive" = true 
                 AND bmi."IsSoldOut" = false
          ))
        ORDER BY r."Weight" DESC
        LIMIT  :limit
        """
    )

    try:
        result = await db.execute(
            sql,
            {
                "source_ids": source_item_ids,
                "branch_id": branch_id,
                "limit": limit,
            },
        )
        rows = result.mappings().all()
    except Exception as exc:
        logger.error("get_recommendations DB error: %s", exc)
        raise HTTPException(status_code=500, detail="Recommendation lookup failed.") from exc

    return [_row_to_product(dict(r)) for r in rows]

