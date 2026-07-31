from __future__ import annotations

from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

class OrderItemDraft(BaseModel):
    """A single line-item the user intends to order."""

    menu_item_name: str = Field(description="Exact product name as it appears in the menu")
    quantity: int = Field(default=1, ge=1, description="Number of units requested")
    size_text: Optional[str] = Field(
        default=None,
        description="Requested size variant: 'small' | 'medium' | 'large'",
    )
    toppings: list[str] = Field(
        default_factory=list,
        description="List of requested topping names",
    )
    note: Optional[str] = Field(
        default=None,
        description="Free-form special instructions for this item",
    )


class OrderDraft(BaseModel):
    """Aggregated order constructed from the user's message."""

    items: list[OrderItemDraft] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Task 2 — RAG Schema Setup
# ---------------------------------------------------------------------------

class ProductResponse(BaseModel):
    """
    Represents a single product surfaced by the RAG semantic search.
    Property names are frozen — Flutter UI depends on the exact JSON contract.
    """

    id: UUID = Field(description="Stable product UUID from PostgreSQL")
    name: str = Field(description="Display name of the product")
    price: float = Field(ge=0, description="Unit price in VND")
    image_url: Optional[HttpUrl] = Field(
        default=None,
        description="CDN URL for the product image; null when no image is available",
    )


class ChatResponse(BaseModel):
    """
    Top-level response envelope returned to the .NET 10 Gateway → Flutter UI.
    Property names are frozen — never rename without coordinating with Flutter team.
    """

    reply: str = Field(
        description="Natural-language assistant reply in Vietnamese"
    )
    order_draft: Optional[OrderDraft] = Field(
        default=None,
        description="Structured order extracted from the user's message; null when no order intent detected",
    )
    recommendations: list[ProductResponse] = Field(
        default_factory=list,
        description="Ranked list of products surfaced by pgvector semantic search",
    )
