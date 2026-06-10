"""Генерация финального ответа по контексту и истории диалога."""
from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from exceptions import RagError

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Строит ответ LLM на основе истории сообщений и извлечённого контекста."""

    def __init__(self, llm: BaseChatModel, system_template: str) -> None:
        self._chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", system_template),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            | llm
            | StrOutputParser()
        )

    def generate(self, messages: list[BaseMessage], context: str) -> str:
        try:
            answer = self._chain.invoke({"messages": messages, "context": context})
        except Exception as exc:  # noqa: BLE001 — любой сбой LLM
            logger.exception("AnswerGenerator failed")
            raise RagError(f"Не удалось сгенерировать ответ: {exc}") from exc
        logger.info("AnswerGenerator: answer_len=%d", len(answer))
        return answer
