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
_EMBED_MODEL = "models/gemini-embedding-001"
_BATCH_SIZE = 50  # stay within Gemini batch limits

# Configure Gemini once at module load
from app.services.key_manager import call_llm_api_with_fallback


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

        def _embed_batch(batch_texts=texts):
            return genai.embed_content(
                model=_EMBED_MODEL,
                content=batch_texts,
                task_type="retrieval_document",
                output_dimensionality=768,
            )

        try:
            response = await call_llm_api_with_fallback(_embed_batch)
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


import asyncio
from app.database import AsyncSessionLocal


async def auto_ingestion_worker():
    """
    Background worker:
    1. Runs immediately on application startup to ingest any missing embeddings.
    2. Repeats periodically every 48 hours (2 days).
    """
    CHECK_INTERVAL_SECONDS = 48 * 60 * 60  # 48 hours

    logger.info("[Auto-Ingestion Worker] Initialized. Starting startup scan...")

    while True:
        try:
            async with AsyncSessionLocal() as session:
                count = await ingest_menu_embeddings(session)
                if count > 0:
                    logger.info("[Auto-Ingestion Worker] Successfully generated embeddings for %d new items.", count)
                else:
                    logger.info("[Auto-Ingestion Worker] Menu vector database is up-to-date (0 items pending).")
        except Exception as e:
            logger.error("[Auto-Ingestion Worker] Error during scheduled ingest: %s", e)

        logger.info("[Auto-Ingestion Worker] Sleeping for %d hours until next scan...", CHECK_INTERVAL_SECONDS // 3600)
        try:
            await asyncio.sleep(CHECK_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            logger.info("[Auto-Ingestion Worker] Worker task cancelled. Shutting down gracefully.")
            break

