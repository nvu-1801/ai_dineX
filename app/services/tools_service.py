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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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


# ---------------------------------------------------------------------------
# Tool 4: check_order_status
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=5),
    retry=retry_if_exception_type(httpx.RequestError),
    reraise=True,
)
async def _check_order_status_with_retry(order_id: str) -> dict:
    headers = {"X-Internal-Api-Key": settings.INTERNAL_API_KEY}
    response = await _http_client.get(f"/api/orders/{order_id}/status", headers=headers)
    
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 401:
        logger.error("Lỗi xác thực API Key nội bộ với .NET Gateway.")
        return {"error": "Hệ thống tra cứu đang được bảo trì (Lỗi xác thực nội bộ)."}
    elif response.status_code == 404:
        return {"error": f"Không tìm thấy đơn hàng với mã {order_id}."}
    else:
        logger.error("Gateway status check error %s: %s", response.status_code, response.text)
        return {"error": "Hệ thống tra cứu đang bận, vui lòng thử lại sau."}

async def execute_check_order_status(order_id: str) -> dict:
    """
    Query the latest order status and progress from the .NET 10 Gateway with retries.
    """
    try:
        return await _check_order_status_with_retry(order_id)
    except Exception as exc:
        logger.error("Failed to query order status after all retry attempts: %s", exc)
        return {"error": "Không thể kết nối với hệ thống tra cứu đơn hàng lúc này."}


# ---------------------------------------------------------------------------
# Tool 5: cancel_order
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=5),
    retry=retry_if_exception_type(httpx.RequestError),
    reraise=True,
)
async def _cancel_order_with_retry(order_id: str) -> dict:
    headers = {"X-Internal-Api-Key": settings.INTERNAL_API_KEY}
    payload = {"reason": "Cancelled by user via AI Chat Agent"}
    response = await _http_client.post(f"/api/ai/chat/orders/{order_id}/cancel", json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 400:
        try:
            err_data = response.json()
            err_msg = err_data.get("detail") or err_data.get("message") or "Đơn hàng đã thanh toán hoặc đã được chuẩn bị, không thể hủy."
            return {"error": err_msg}
        except Exception:
            return {"error": "Điều kiện hủy đơn hàng không hợp lệ (đơn đã thanh toán hoặc đang chuẩn bị)."}
    elif response.status_code == 401:
        logger.error("Lỗi xác thực API Key nội bộ với .NET Gateway.")
        return {"error": "Hệ thống hủy đơn đang được bảo trì (Lỗi xác thực nội bộ)."}
    elif response.status_code == 404:
        return {"error": f"Không tìm thấy đơn hàng với mã {order_id}."}
    else:
        logger.error("Gateway cancel order error %s: %s", response.status_code, response.text)
        return {"error": "Hệ thống hủy đơn đang bận, vui lòng thử lại sau."}

async def execute_cancel_order(order_id: str) -> dict:
    """
    Cancel an order on behalf of the customer via .NET 10 Gateway.
    """
    try:
        return await _cancel_order_with_retry(order_id)
    except Exception as exc:
        logger.error("Failed to cancel order after all retry attempts: %s", exc)
        return {"error": "Không thể kết nối với hệ thống hủy đơn hàng lúc này."}
