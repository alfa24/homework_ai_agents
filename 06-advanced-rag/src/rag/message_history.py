"""Протокол хранилища истории RAG-диалога."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from langchain_core.messages import BaseMessage


@runtime_checkable
class MessageHistoryStore(Protocol):
    """Контракт хранилища истории сообщений диалога по chat_id."""

    def get(self, chat_id: int) -> list[BaseMessage]:
        """Возвращает копию истории сообщений для чата."""

    def append(self, chat_id: int, question: str, answer: str) -> None:
        """Добавляет пару сообщений (вопрос пользователя + ответ ассистента)."""

    def clear(self, chat_id: int) -> None:
        """Очищает историю диалога для чата."""
