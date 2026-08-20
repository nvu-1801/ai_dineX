"""
Step 4 — RAG & Semantic Search Service
Provides:
  - search_products: cosine-distance pgvector search
  - get_recommendations: GraphRAG cross-sell via ItemRelations table
"""
from __future__ import annotations

import logging
import asyncio
import re
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


def extract_price_info(text: str) -> tuple[float | None, float | None, str]:
    """
    Extracts price constraints (min_price, max_price) from Vietnamese natural language food queries.
    Supports units: k, kđ, nghìn, ngàn, triệu, tr, vnd, đ.
    Supports price thresholds: 50k, 100k, 200k, 300k, 400k, 500k, 1tr, etc.
    """
    if not text:
        return None, None, ""

    text_lower = text.lower()
    min_price: float | None = None
    max_price: float | None = None

    # Pre-normalize slashes, underscores, and extra whitespace in input
    text_lower = re.sub(r'[\/\\\_]+', ' ', text_lower)
    text_lower = re.sub(r'\s+', ' ', text_lower).strip()

    def _parse_num(val_str: str, unit_str: str | None) -> float:
        val = float(val_str.replace('.', '').replace(',', ''))
        if not unit_str:
            return val * 1000 if val < 1000 else val
        unit = unit_str.strip().lower()
        if unit.startswith(('tr', 'triệu')):
            return val * 1000000
        if unit.startswith(('k', 'nghìn', 'ngàn', 'kđ')):
            return val * 1000
        return val * 1000 if val < 1000 else val

    # 1. Range pattern: e.g. "từ 30k đến 50k", "30-50k", "từ 100 nghìn tới 200k"
    range_match = re.search(
        r'(\d+(?:[\.,]\d+)?)\s*(k|kđ|nghìn|ngàn|tr|triệu)?\s*(đến|-|tới)\s*(\d+(?:[\.,]\d+)?)\s*(k|kđ|nghìn|ngàn|tr|triệu)',
        text_lower
    )
    if range_match:
        val1 = _parse_num(range_match.group(1), range_match.group(2) or range_match.group(5))
        val2 = _parse_num(range_match.group(4), range_match.group(5))
        min_price, max_price = min(val1, val2), max(val1, val2)
        text_lower = text_lower.replace(range_match.group(0), '')

    # 2. Max price pattern: e.g. "dưới 50k", "dưới 100k", "dưới 200k", "dưới 300k", "dưới 400k", "dưới 500k", "dưới 1tr", "tối đa 200k"
    if not max_price:
        max_match = re.search(
            r'(dưới|nhỏ hơn|ít hơn|max|tối đa)\s*(\d+(?:[\.,]\d+)?)\s*(k|kđ|nghìn|ngàn|tr|triệu|vnd|đ)?',
            text_lower
        )
        if max_match:
            max_price = _parse_num(max_match.group(2), max_match.group(3))
            text_lower = text_lower.replace(max_match.group(0), '')

    # 3. Min price pattern: e.g. "trên 30k", "trên 100k", "trên 200k", "từ 50k", "tối thiểu 100k"
    if not min_price:
        min_match = re.search(
            r'(trên|từ|lớn hơn|min|tối thiểu|hơn)\s*(\d+(?:[\.,]\d+)?)\s*(k|kđ|nghìn|ngàn|tr|triệu|vnd|đ)?',
            text_lower
        )
        if min_match:
            min_price = _parse_num(min_match.group(2), min_match.group(3))
            text_lower = text_lower.replace(min_match.group(0), '')

    # 4. Exact / Estimated price pattern: e.g. "tầm 45k", "khoảng 100k", "giá 200k", "100k", "200k", "300k", "400k", "500k"
    if not min_price and not max_price:
        exact_match = re.search(
            r'(tầm|khoảng|giá)?\s*(\d+(?:[\.,]\d+)?)\s*(k|kđ|nghìn|ngàn|tr|triệu)\b',
            text_lower
        )
        if exact_match:
            target_price = _parse_num(exact_match.group(2), exact_match.group(3))
            tolerance = 15000 if target_price <= 100000 else target_price * 0.15
            min_price = max(0, target_price - tolerance)
            max_price = target_price + tolerance
            text_lower = text_lower.replace(exact_match.group(0), '')

    # Clean remaining price filler words & digits from clean_query
    clean_query = re.sub(r'\b(dưới|nhỏ hơn|ít hơn|tối đa|trên|lớn hơn|hơn|tối thiểu|giá|tầm|khoảng|nghìn|ngàn|triệu|k|kđ|vnd|đ|đồng)\b', '', text_lower, flags=re.IGNORECASE)
    clean_query = re.sub(r'\d+', '', clean_query)
    clean_query = re.sub(r'\s+', ' ', clean_query).strip()

    return min_price, max_price, clean_query


def normalize_vietnamese_food_typos(text: str) -> str:
    """Normalize common Vietnamese food typos e.g. lẫu -> lẩu, hủ tếu -> hủ tiếu."""
    if not text:
        return ""
    text_lower = text.lower()
    typo_map = {
        r'\blẫu\b': 'lẩu',
        r'\blễu\b': 'lẩu',
        r'\bhủ tếu\b': 'hủ tiếu',
        r'\bhu tieu\b': 'hủ tiếu',
        r'\bpho bo\b': 'phở bò',
        r'\bpho ga\b': 'phở gà',
        r'\bcom tam\b': 'cơm tấm',
        r'\btra dao\b': 'trà đào',
        r'\btra sua\b': 'trà sữa',
    }
    for pattern, replacement in typo_map.items():
        text_lower = re.sub(pattern, replacement, text_lower)
    return text_lower


def extract_food_query(text: str) -> str:
    """
    Strips conversational context prefixes (e.g. 'tôi muốn ăn', 'cho tôi món', 'tôi thèm')
    and suffix filler words to extract the core dish/food search query.
    """
    if not text:
        return ""

    cleaned = normalize_vietnamese_food_typos(text.strip())

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
        r"^(top|bán chạy|hot|ngon|món ngon|nổi tiếng)\s+(các\s+món|món\s+ăn|món)?\s*",
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
        r"\s+(gia đình|nổi tiếng|bán chạy|ngon|hot|top)$",
    ]
    for pat in suffix_patterns:
        subbed = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()
        if subbed:
            cleaned = subbed

    # Strip leading quantities e.g. "2 tô ", "1 suất "
    qty_sub = re.sub(r"^\d+\s*(tô|bát|dĩa|phần|suất|ly|cốc)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
    if qty_sub:
        cleaned = qty_sub

    # Strip discovery filler words e.g. "top", "món ăn", "bán chạy", "ngon", "các món", "hot"
    discovery_clean = re.sub(r'\b(top|bán chạy|hot|ngon|món ngon|các món|gợi ý|nổi tiếng|món ăn|món|gia đình)\b', '', cleaned, flags=re.IGNORECASE)
    discovery_clean = re.sub(r'\s+', ' ', discovery_clean).strip()
    if not discovery_clean:
        return ""

    return discovery_clean


import unicodedata


def remove_vietnamese_accent(text: str) -> str:
    """Chuyển đổi chuỗi tiếng Việt có dấu thành không dấu."""
    if not text:
        return ""
    text = re.sub(r'[đĐ]', 'd', text)
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    return text.lower().strip()


def extract_search_tokens(query: str) -> list[str]:
    """Tách các từ khóa (tokens) từ chuỗi tìm kiếm, bỏ qua ký tự đặc biệt."""
    if not query:
        return []
    clean_text = re.sub(r'[^\w\s]', ' ', query.lower())
    tokens = [t.strip() for t in clean_text.split() if len(t.strip()) > 0]
    return tokens


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def search_products(
    db: AsyncSession,
    query: str,
    category_id: str | None = None,
    branch_id: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    limit: int = 3,
    distance_limit: float = 0.65,
    user_lat: float | None = None,
    user_lng: float | None = None,
    max_radius_km: float = 15.0,
) -> list[ProductResponse]:
    """
    Hybrid Multi-Tier Search (Exact -> Multi-Token AND LIKE -> Vector pgvector).
    Handles 3-4-5 word exact dish names regardless of word order or punctuation.
    Enforces 15km user location radius filtering (matching Nearby Stores & Hot Deals).
    """
    extracted_min, extracted_max, clean_q = extract_price_info(query)
    final_min_price = min_price if min_price is not None else extracted_min
    final_max_price = max_price if max_price is not None else extracted_max

    clean_query = extract_food_query(clean_q if clean_q else query)
    search_text = clean_query if clean_query else ""
    is_price_only = not search_text.strip() or search_text.strip().lower() in [
        "món", "đồ ăn", "thực đơn", "các món", "bán", "quán", "đồ uống", "top", "ngon", "bán chạy", "hot", "gợi ý", "nổi tiếng", "top món", "top món ăn", "top món ngon"
    ]

    effective_lat = user_lat if user_lat is not None else 10.776889
    effective_lng = user_lng if user_lng is not None else 106.700806

    logger.info(
        "[search_products] Raw query: '%s' -> Clean query: '%s' | Price range: [%s, %s] | User Pos: (%s, %s) | Max Radius: %s km",
        query,
        search_text,
        final_min_price,
        final_max_price,
        effective_lat,
        effective_lng,
        max_radius_km
    )

    if is_price_only:
        # Nhánh 1: Truy vấn thuần túy theo giá với 15km Radius Filter
        sql_str = """
            SELECT m."Id"::text          AS id,
                   m."Name"              AS name,
                   m."BasePrice"::text   AS price,
                   m."ImageUrl"          AS image_url,
                   0.0                   AS distance,
                   0                     AS match_rank
            FROM "MenuItems" m
            WHERE m."IsAvailable" = true
              AND (:min_price IS NULL OR m."BasePrice" >= :min_price)
              AND (:max_price IS NULL OR m."BasePrice" <= :max_price)
              AND (:category_id IS NULL OR m."CategoryId" = CAST(:category_id AS uuid))
              AND (:branch_id IS NULL OR EXISTS (
                   SELECT 1 FROM "BranchMenuItems" bmi 
                   JOIN "Branches" b ON b."Id" = bmi."BranchId"
                   WHERE bmi."MenuItemId" = m."Id" 
                     AND bmi."BranchId" = CAST(:branch_id AS uuid) 
                     AND bmi."IsActive" = true 
                     AND bmi."IsSoldOut" = false
                     AND (
                       b."Latitude" IS NULL OR b."Longitude" IS NULL OR
                       (6371 * acos(LEAST(1.0, cos(radians(:user_lat)) * cos(radians(b."Latitude")) * cos(radians(b."Longitude") - radians(:user_lng)) + sin(radians(:user_lat)) * sin(radians(b."Latitude"))))) <= :max_radius_km
                     )
              ))
              AND (
                :branch_id IS NOT NULL OR EXISTS (
                   SELECT 1 FROM "BranchMenuItems" bmi 
                   JOIN "Branches" b ON b."Id" = bmi."BranchId"
                   WHERE bmi."MenuItemId" = m."Id" 
                     AND bmi."IsActive" = true 
                     AND bmi."IsSoldOut" = false
                     AND (
                       b."Latitude" IS NULL OR b."Longitude" IS NULL OR
                       (6371 * acos(LEAST(1.0, cos(radians(:user_lat)) * cos(radians(b."Latitude")) * cos(radians(b."Longitude") - radians(:user_lng)) + sin(radians(:user_lat)) * sin(radians(b."Latitude"))))) <= :max_radius_km
                     )
                )
              )
            ORDER BY m."BasePrice" ASC
            LIMIT :top_k
        """
        params = {
            "top_k": limit,
            "category_id": category_id,
            "branch_id": branch_id,
            "min_price": final_min_price,
            "max_price": final_max_price,
            "user_lat": effective_lat,
            "user_lng": effective_lng,
            "max_radius_km": max_radius_km,
        }
        try:
            result = await db.execute(text(sql_str), params)
            rows = result.mappings().all()
            return [_row_to_product(dict(r)) for r in rows]
        except Exception as exc:
            logger.error("search_products Price-Only DB error: %s", exc)
            raise HTTPException(status_code=500, detail="Product search failed.") from exc

    # Nhánh 2: Hybrid Search với 15km Radius Filter
    vector_literal = await _embed_query(search_text)
    tokens = extract_search_tokens(search_text)

    params: dict[str, Any] = {
        "vector": vector_literal,
        "exact_pattern": f"%{search_text.lower()}%",
        "top_k": limit,
        "distance_limit": distance_limit,
        "category_id": category_id,
        "branch_id": branch_id,
        "min_price": final_min_price,
        "max_price": final_max_price,
        "user_lat": effective_lat,
        "user_lng": effective_lng,
        "max_radius_km": max_radius_km,
    }

    token_conditions = []
    for i, token in enumerate(tokens):
        param_name = f"token_{i}"
        unaccent_param = f"unaccent_{i}"
        unaccent_val = remove_vietnamese_accent(token)
        token_conditions.append(f'(LOWER(m."Name") LIKE :{param_name} OR LOWER(m."Name") LIKE :{unaccent_param})')
        params[param_name] = f"%{token}%"
        params[unaccent_param] = f"%{unaccent_val}%"

    token_sql = " OR ".join(token_conditions) if token_conditions else "FALSE"

    sql_str = f"""
        WITH RankedItems AS (
            SELECT m."Id"::text          AS id,
                   m."Name"              AS name,
                   m."BasePrice"::text   AS price,
                   m."ImageUrl"          AS image_url,
                   CASE WHEN m."Embedding" IS NOT NULL THEN (m."Embedding" <=> CAST(:vector AS vector)) ELSE 1.0 END AS distance,
                   CASE 
                       -- Rank 0: Match chính xác cụm từ liên tiếp
                       WHEN LOWER(m."Name") LIKE :exact_pattern THEN 0
                       
                       -- Rank 1: Match toàn bộ các từ (Tokens) không phân biệt thứ tự (3-4-5 từ)
                       WHEN {token_sql} THEN 1
                       
                       -- Rank 3: Cosine Similarity Vector Search
                       ELSE 3
                   END AS match_rank
            FROM "MenuItems" m
            WHERE m."IsAvailable" = true
              AND (:min_price IS NULL OR m."BasePrice" >= :min_price)
              AND (:max_price IS NULL OR m."BasePrice" <= :max_price)
              AND (:category_id IS NULL OR m."CategoryId" = CAST(:category_id AS uuid))
              AND (:branch_id IS NULL OR EXISTS (
                   SELECT 1 FROM "BranchMenuItems" bmi 
                   JOIN "Branches" b ON b."Id" = bmi."BranchId"
                   WHERE bmi."MenuItemId" = m."Id" 
                     AND bmi."BranchId" = CAST(:branch_id AS uuid) 
                     AND bmi."IsActive" = true 
                     AND bmi."IsSoldOut" = false
                     AND (
                       b."Latitude" IS NULL OR b."Longitude" IS NULL OR
                       (6371 * acos(LEAST(1.0, cos(radians(:user_lat)) * cos(radians(b."Latitude")) * cos(radians(b."Longitude") - radians(:user_lng)) + sin(radians(:user_lat)) * sin(radians(b."Latitude"))))) <= :max_radius_km
                     )
              ))
              AND (
                :branch_id IS NOT NULL OR EXISTS (
                   SELECT 1 FROM "BranchMenuItems" bmi 
                   JOIN "Branches" b ON b."Id" = bmi."BranchId"
                   WHERE bmi."MenuItemId" = m."Id" 
                     AND bmi."IsActive" = true 
                     AND bmi."IsSoldOut" = false
                     AND (
                       b."Latitude" IS NULL OR b."Longitude" IS NULL OR
                       (6371 * acos(LEAST(1.0, cos(radians(:user_lat)) * cos(radians(b."Latitude")) * cos(radians(b."Longitude") - radians(:user_lng)) + sin(radians(:user_lat)) * sin(radians(b."Latitude"))))) <= :max_radius_km
                     )
                )
              )
              AND (
                  (m."Embedding" IS NOT NULL AND (m."Embedding" <=> CAST(:vector AS vector)) <= :distance_limit)
                  OR LOWER(m."Name") LIKE :exact_pattern
                  OR ({token_sql})
              )
        )
        SELECT id, name, price, image_url, distance, match_rank
        FROM RankedItems
        ORDER BY match_rank ASC, distance ASC
        LIMIT :top_k
    """

    sql = text(sql_str)

    try:
        result = await db.execute(sql, params)
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


async def get_user_personalized_recommendations(
    db: AsyncSession,
    user_id: str,
    branch_id: str | None = None,
    limit: int = 9,
) -> list[dict]:
    """
    Generate AI personalized product recommendations for a specific user using:
    1. Order History (OrderItems & Orders)
    2. Search / Research History (UserSearchHistories)
    3. Gemini Vector Embedding + pgvector Cosine Search
    """
    logger.info("[Personalized Recs] Generating recommendations for UserId: %s (BranchId: %s)", user_id, branch_id)

    # 1. Fetch user's recent ordered items
    recent_orders_sql = text(
        """
        SELECT DISTINCT oi."ProductName"
        FROM "Orders" o
        JOIN "OrderItems" oi ON o."Id" = oi."OrderId"
        WHERE o."CustomerId" = CAST(:user_id AS uuid)
          AND o."OrderStatus" NOT IN ('Cancelled')
        ORDER BY oi."ProductName"
        LIMIT 5
        """
    )
    
    # 2. Fetch user's recent search queries
    recent_searches_sql = text(
        """
        SELECT DISTINCT "Query"
        FROM "UserSearchHistories"
        WHERE "UserId" = CAST(:user_id AS uuid)
          AND "IsDelete" = false
        ORDER BY "Query"
        LIMIT 5
        """
    )

    ordered_names: list[str] = []
    search_queries: list[str] = []

    try:
        res_orders = await db.execute(recent_orders_sql, {"user_id": user_id})
        ordered_names = [r[0] for r in res_orders.fetchall() if r[0]]

        res_searches = await db.execute(recent_searches_sql, {"user_id": user_id})
        search_queries = [r[0] for r in res_searches.fetchall() if r[0]]
    except Exception as exc:
        logger.warning("[Personalized Recs] History fetch warning: %s", exc)

    # 3. Build User Preference Context
    pref_parts = []
    if ordered_names:
        pref_parts.append(f"Món ăn đã từng đặt: {', '.join(ordered_names)}")
    if search_queries:
        pref_parts.append(f"Từ khóa vừa tìm kiếm: {', '.join(search_queries)}")

    if pref_parts:
        preference_summary = "Sở thích khẩu vị người dùng: " + ". ".join(pref_parts)
        reason = "Gợi ý dựa trên món bạn từng mua & từ khóa tìm kiếm gần đây"
    else:
        preference_summary = "Món ăn ngon được yêu thích hàng đầu, dễ dùng, phục vụ nhanh"
        reason = "Gợi ý món ngon được yêu thích hàng đầu"

    logger.info("[Personalized Recs] Preference summary: '%s'", preference_summary)

    # 4. Generate Preference Vector & Perform pgvector Cosine Distance Search
    vector_literal = await _embed_query(preference_summary)

    sql = text(
        """
        WITH CandidateRecs AS (
            SELECT m."Id"::text          AS id,
                   m."Name"              AS name,
                   m."BasePrice"::text   AS price,
                   m."ImageUrl"          AS image_url,
                   CASE WHEN m."Embedding" IS NOT NULL 
                        THEN (m."Embedding" <=> CAST(:vector AS vector)) 
                        ELSE 1.0 
                   END AS distance,
                   COALESCE((
                       SELECT COUNT(oi."Id") 
                       FROM "OrderItems" oi 
                       WHERE oi."MenuItemId" = m."Id" OR oi."ProductName" = m."Name"
                   ), 0) AS order_count
            FROM   "MenuItems" m
            WHERE  m."IsAvailable"  = true
              AND  (:branch_id IS NULL OR EXISTS (
                   SELECT 1 FROM "BranchMenuItems" bmi 
                   WHERE bmi."MenuItemId" = m."Id" 
                     AND bmi."BranchId" = CAST(:branch_id AS uuid) 
                     AND bmi."IsActive" = true 
                     AND bmi."IsSoldOut" = false
              ))
            ORDER BY 
              CASE WHEN m."Embedding" IS NOT NULL THEN (m."Embedding" <=> CAST(:vector AS vector)) ELSE 1.0 END ASC
            LIMIT 30
        )
        SELECT id, name, price, image_url
        FROM CandidateRecs
        ORDER BY order_count DESC, distance ASC
        LIMIT  :top_k
        """
    )

    try:
        result = await db.execute(
            sql,
            {
                "vector": vector_literal,
                "branch_id": branch_id,
                "top_k": limit,
            },
        )
        rows = result.mappings().all()
    except Exception as exc:
        logger.error("[Personalized Recs] DB Query error: %s", exc)
        raise HTTPException(status_code=500, detail="Personalized recommendation query failed.") from exc

    return [
        {
            "id": UUID(r["id"]),
            "name": r["name"],
            "price": float(r["price"]),
            "image_url": r.get("image_url"),
            "reason": reason,
        }
        for r in rows
    ]


