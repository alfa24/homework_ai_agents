"""In-memory реализация `MessageHistoryStore` для RAG-ассистента."""
from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class InMemoryMessageHistory:
    """Хранит историю RAG-диалога для каждого chat_id с ограничением длины.

    Буфер ограничен по числу сообщений (Human + AI считаются раздельно).
    Системный промпт в историю не входит — его подставляет цепочка ответа.
    """

    def __init__(self, max_messages: int = 10) -> None:
        self._max_messages = max_messages
        self._data: dict[int, list[BaseMessage]] = {}

    def get(self, chat_id: int) -> list[BaseMessage]:
        return list(self._data.get(chat_id, []))

    def append(self, chat_id: int, question: str, answer: str) -> None:
        history = self._data.setdefault(chat_id, [])
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))
        overflow = len(history) - self._max_messages
        if overflow > 0:
            del history[:overflow]

    def clear(self, chat_id: int) -> None:
        self._data.pop(chat_id, None)
