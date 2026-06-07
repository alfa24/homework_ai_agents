"""Снимок состояния RAG-индекса."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class IndexState:
    """Иммутабельный снимок: сколько документов и когда индексировались."""

    document_count: int
    last_indexed_at: datetime
