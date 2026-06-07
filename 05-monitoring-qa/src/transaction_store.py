"""In-memory хранилище транзакций по chat_id."""
from __future__ import annotations

from typing import Iterable

from models import Transaction


class TransactionStore:
    def __init__(self) -> None:
        self._data: dict[int, list[Transaction]] = {}

    def get(self, chat_id: int) -> list[Transaction]:
        return list(self._data.get(chat_id, []))

    def extend(self, chat_id: int, transactions: Iterable[Transaction]) -> None:
        bucket = self._data.setdefault(chat_id, [])
        bucket.extend(transactions)

    def clear(self, chat_id: int) -> None:
        self._data.pop(chat_id, None)
