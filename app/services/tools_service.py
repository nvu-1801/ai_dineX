"""
Step 5a — Business Function Tools
Provides typed tool implementations called by Gemini Function Calling:
  - calculate_total_price
  - generate_payment_qr   (calls .NET 10 Gateway)
  - submit_order          (calls .NET 10 Gateway)
"""
from __future__ import annotations

import logging

import httpx
from fastapi import HTTPException

from app.config import settings

logger = logging.getLogger("tools_service")

# Shared async HTTP client — reused across calls (connection pooling)
_http_client = httpx.AsyncClient(
    base_url=settings.MAIN_BACKEND_URL.rstrip("/"),
    timeout=httpx.Timeout(10.0),
    headers={"Content-Type": "application/json"},
)


# ---------------------------------------------------------------------------
# Tool 1: calculate_total_price
# ---------------------------------------------------------------------------

def calculate_total_price(items: list[dict]) -> dict:
    """
    Compute order subtotal from a list of line items.

    Each item dict must contain:
      - name       (str)
      - quantity   (int)
      - unit_price (float, VND)

    Returns:
      {
        "breakdown": [{"name": str, "quantity": int, "subtotal": float}, ...],
        "total":     float
      }
    """
    breakdown: list[dict] = []
    total = 0.0

    for item in items:
        name: str = item.get("name", "Unknown")
        quantity: int = int(item.get("quantity", 1))
        unit_price: float = float(item.get("unit_price", 0.0))
        subtotal = round(quantity * unit_price, 2)
        breakdown.append({"name": name, "quantity": quantity, "subtotal": subtotal})
        total += subtotal

    return {"breakdown": breakdown, "total": round(total, 2)}


# ---------------------------------------------------------------------------
# Tool 2: generate_payment_qr
# ---------------------------------------------------------------------------

async def generate_payment_qr(amount: float, order_info: str) -> dict:
    """
    Request a VietQR payload from the .NET 10 Gateway.

    Args:
        amount:     Total amount in VND (must be > 0).
        order_info: Human-readable order reference string.

    Returns gateway JSON: { "qr_data_url": str, "transaction_ref": str, ... }
    """
    if amount <= 0:
        raise HTTPException(status_code=400, detail="Payment amount must be greater than 0.")

    payload = {"amount": amount, "orderInfo": order_info}

    try:
        response = await _http_client.post("/api/payments/vietqr", json=payload)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        logger.error("Gateway VietQR error %s: %s", exc.response.status_code, exc.response.text)
        raise HTTPException(
            status_code=502,
            detail=f"Payment gateway error: {exc.response.status_code}",
        ) from exc
    except httpx.RequestError as exc:
        logger.error("Gateway connection error: %s", exc)
        raise HTTPException(status_code=503, detail="Payment gateway unreachable.") from exc


# ---------------------------------------------------------------------------
# Tool 3: submit_order
# ---------------------------------------------------------------------------

async def submit_order(order_data: dict) -> dict:
    """
    Post a confirmed order to the .NET 10 Gateway /api/orders endpoint.

    Args:
        order_data: Fully validated order payload dict.

    Returns gateway JSON: { "order_id": str, "status": str, ... }
    """
    try:
        response = await _http_client.post("/api/orders", json=order_data)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        logger.error("Gateway submit_order error %s: %s", exc.response.status_code, exc.response.text)
        raise HTTPException(
            status_code=502,
            detail=f"Order submission failed: {exc.response.status_code}",
        ) from exc
    except httpx.RequestError as exc:
        logger.error("Gateway connection error on submit_order: %s", exc)
        raise HTTPException(status_code=503, detail="Order gateway unreachable.") from exc
