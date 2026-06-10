"""Переписывание запроса пользователя с учётом истории диалога."""
from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from exceptions import RagError

logger = logging.getLogger(__name__)


class QueryRewriter:
    """Трансформирует последнее сообщение пользователя в поисковый запрос."""

    def __init__(self, llm: BaseChatModel, instruction: str) -> None:
        self._chain = (
            ChatPromptTemplate.from_messages(
                [
                    MessagesPlaceholder(variable_name="messages"),
                    ("user", instruction),
                ]
            )
            | llm
            | StrOutputParser()
        )

    def rewrite(self, messages: list[BaseMessage]) -> str:
        try:
            query = self._chain.invoke({"messages": messages})
        except Exception as exc:  # noqa: BLE001 — любой сбой LLM
            logger.exception("QueryRewriter failed")
            raise RagError(f"Не удалось переписать запрос: {exc}") from exc
        logger.info("QueryRewriter: rewritten=%r", query[:200])
        return query
