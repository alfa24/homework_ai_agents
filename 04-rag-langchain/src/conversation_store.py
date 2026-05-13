"""In-memory хранилище истории диалога по chat_id."""
from __future__ import annotations


class ConversationStore:
    """Хранит сообщения диалога с ограничением по длине.

    Системный промпт в историю не входит — его подставляет LLMClient.
    """

    def __init__(self, max_messages: int = 20) -> None:
        self._max_messages = max_messages
        self._data: dict[int, list[dict]] = {}

    def get(self, chat_id: int) -> list[dict]:
        return list(self._data.get(chat_id, []))

    def append(self, chat_id: int, role: str, content: str) -> None:
        history = self._data.setdefault(chat_id, [])
        history.append({"role": role, "content": content})
        overflow = len(history) - self._max_messages
        if overflow > 0:
            del history[:overflow]

    def clear(self, chat_id: int) -> None:
        self._data.pop(chat_id, None)
