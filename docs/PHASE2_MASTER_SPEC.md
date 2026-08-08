# MASTER TASK SPECIFICATION: PHASE 2 (STEPS 3, 4 & 5)

@BackendDev @RAGEngineer
Read `.cursorrules` and `docs/ARCHITECTURE.md` carefully before writing code. 
Implement Steps 3, 4, and 5 for the DineX-AI service in one cohesive execution.

---

## TASK 1: STEP 3 - DATA INGESTION (`app/services/ingest_service.py` & `/api/ai/ingest`)

1. Create `app/services/ingest_service.py`:
   - Write an async function `ingest_menu_embeddings(db: AsyncSession)`:
     - Query all rows from `"MenuItems"` where `"Embedding"` IS NULL.
     - Batch process texts (`Name` + `Description`) using `google.generativeai.embed_content` (`models/text-embedding-004`).
     - Execute raw SQL bulk updates to save the vector array directly into the `"Embedding"` column (`pgvector`).
2. Expose a `POST /api/ai/ingest` endpoint in `app/routers/ai.py` that triggers this function and returns `{"status": "success", "processed_count": int}`.

---

## TASK 2: STEP 4 - RAG & SEARCH LOGIC (`app/services/rag_service.py`)

Create `app/services/rag_service.py` with the following async functions using raw SQL (`asyncpg` / `SQLAlchemy text()`):

1. `search_products(db: AsyncSession, query: str, limit: int = 3) -> list[ProductResponse]`:
   - Generate embedding for `query`.
   - Perform Cosine Distance search (`<=>`) on `"MenuItems"`:
     ```sql
     SELECT "Id"::text as id, "Name" as name, "PriceAmount"::text as price, "ImageUrl" as image_url
     FROM "MenuItems"
     WHERE "IsAvailable" = true AND "IsSoldOut" = false
     ORDER BY "Embedding" <=> :vector::vector
     LIMIT :limit;
     ```
   - Return mapped `ProductResponse` schemas.

2. `get_recommendations(db: AsyncSession, source_item_ids: list[str], limit: int = 2) -> list[ProductResponse]`:
   - Perform GraphRAG lookup via `"ItemRelations"` table to fetch cross-sell/up-sell targets:
     ```sql
     SELECT m."Id"::text as id, m."Name" as name, m."PriceAmount"::text as price, m."ImageUrl" as image_url
     FROM "ItemRelations" r
     JOIN "MenuItems" m ON r."TargetItemId" = m."Id"
     WHERE r."SourceItemId" ANY(:source_ids) AND m."IsAvailable" = true
     ORDER BY r."Weight" DESC
     LIMIT :limit;
     ```

---

## TASK 3: STEP 5 - BUSINESS FUNCTIONS & GEMINI CALLING (`app/services/tools_service.py` & `app/services/chat_service.py`)

1. Create `app/services/tools_service.py`:
   - `calculate_total_price(items: list[dict]) -> dict`: Sums prices and returns total breakdown.
   - `generate_payment_qr(amount: float, order_info: str) -> dict`: Calls .NET 10 Gateway via `httpx.AsyncClient` to fetch VietQR payload from `MAIN_BACKEND_URL`.
   - `submit_order(order_data: dict) -> dict`: Posts final order payload to .NET 10 Gateway endpoint `/api/orders`.

2. Create `app/services/chat_service.py`:
   - Initialize `genai.GenerativeModel('gemini-1.5-flash')` with the tools above and System Instructions.
   - Handle incoming user message:
     - Run `search_products` to pull context.
     - Run Gemini `generate_content_async` with tool declarations.
     - Process tool calls if Gemini requests them.
     - Format and strictly return a valid `ChatResponse` (as defined in `app/schemas.py`).

---

## TASK 4: ROUTER & ENTRY POINT INTEGRATION (`app/routers/ai.py` & `main.py`)

1. Create `app/routers/ai.py`:
   - `POST /api/ai/chat`: Accepts `{"message": str, "branch_id": Optional[str]}`. Injects `AsyncSession` via `get_db()`. Returns strict `ChatResponse`.
   - `POST /api/ai/ingest`: Triggers data ingestion.
2. Update `main.py`:
   - Register `app/routers/ai.py` router with prefix `/api/ai`.
   - Ensure CORS middleware is enabled.

---

## CONSTRAINTS
- Strict `async`/`await` for all I/O, Database, and HTTP operations.
- Do NOT alter property names in `ProductResponse` or `ChatResponse` (`app/schemas.py`).
- Output full, production-ready Python files without placeholders or truncated code.