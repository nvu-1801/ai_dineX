"""
Step 3 — Data Ingestion Service
Fetches MenuItems with NULL Embedding from PostgreSQL,
generates vectors via Gemini text-embedding-004, and bulk-updates pgvector.
"""
from __future__ import annotations

import logging
from typing import Any

import google.generativeai as genai
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings

logger = logging.getLogger("ingest_service")

# Gemini embedding model
_EMBED_MODEL = "models/text-embedding-004"
_BATCH_SIZE = 50  # stay within Gemini batch limits

# Configure Gemini once at module load
import os

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or settings.GEMINI_API_KEY
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is missing.")

genai.configure(api_key=api_key)


async def ingest_menu_embeddings(db: AsyncSession) -> int:
    """
    Embed all MenuItems where Embedding IS NULL.

    Returns:
        Number of rows updated.
    """
    # 1. Fetch items that need embeddings
    fetch_sql = text(
        """
        SELECT "Id"::text AS id,
               "Name"       AS name,
               "Description" AS description
        FROM   "MenuItems"
        WHERE  "Embedding" IS NULL
        LIMIT  1000
        """
    )
    result = await db.execute(fetch_sql)
    rows: list[Any] = result.mappings().all()

    if not rows:
        logger.info("No MenuItems require embedding — nothing to do.")
        return 0

    logger.info("Found %d items requiring embeddings.", len(rows))

    processed = 0

    # 2. Batch-embed & update
    for batch_start in range(0, len(rows), _BATCH_SIZE):
        batch = rows[batch_start : batch_start + _BATCH_SIZE]

        texts = [
            f"{r['name']}. {r['description'] or ''}".strip() for r in batch
        ]

        try:
            response = genai.embed_content(
                model=_EMBED_MODEL,
                content=texts,
                task_type="retrieval_document",
                output_dimensionality=768,
            )
            embeddings: list[list[float]] = response["embedding"]
        except Exception as exc:
            logger.error("Gemini embedding error on batch starting %d: %s", batch_start, exc)
            raise

        # 3. Bulk-update each row — pgvector accepts cast from text literal
        for row, vector in zip(batch, embeddings):
            vector_literal = "[" + ",".join(map(str, vector)) + "]"
            update_sql = text(
                """
                UPDATE "MenuItems"
                SET    "Embedding" = CAST(:vec AS vector)
                WHERE  "Id" = CAST(:id AS uuid)
                """
            )
            await db.execute(update_sql, {"vec": vector_literal, "id": row["id"]})

        await db.commit()
        processed += len(batch)
        logger.info("Committed batch — cumulative processed: %d", processed)

    return processed
