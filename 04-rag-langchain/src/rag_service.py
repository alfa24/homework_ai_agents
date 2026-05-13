"""RAG-сервис: тонкий оркестратор шагов пайплайна и истории диалога."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from langchain_core.messages import HumanMessage

from rag.answer_generator import AnswerGenerator
from rag.context_retriever import ContextRetriever
from rag.corpus_indexer import CorpusIndexer
from rag.message_history import MessageHistoryStore
from rag.query_rewriter import QueryRewriter

logger = logging.getLogger(__name__)


class RagService:
    """Координирует индексацию, шаги пайплайна и историю диалога."""

    def __init__(
        self,
        indexer: CorpusIndexer,
        rewriter: QueryRewriter,
        retriever: ContextRetriever,
        generator: AnswerGenerator,
        conversations: MessageHistoryStore,
    ) -> None:
        self._indexer = indexer
        self._rewriter = rewriter
        self._retriever = retriever
        self._generator = generator
        self._conversations = conversations

    @property
    def is_ready(self) -> bool:
        return self._indexer.is_ready

    @property
    def document_count(self) -> int:
        state = self._indexer.state
        return state.document_count if state else 0

    @property
    def last_indexed_at(self) -> Optional[datetime]:
        state = self._indexer.state
        return state.last_indexed_at if state else None

    def reindex(self) -> int:
        """Перестраивает векторное хранилище из всех источников."""
        state = self._indexer.reindex()
        return state.document_count

    def answer(self, chat_id: int, question: str) -> str:
        """Диалоговый ответ по корпусу с query transformation и историей."""
        history = self._conversations.get(chat_id)
        messages = history + [HumanMessage(content=question)]
        logger.info(
            "RAG answer: chat=%s, history_len=%d, question=%r",
            chat_id, len(history), question[:120],
        )

        rewritten = self._rewriter.rewrite(messages)
        chunks = self._retriever.retrieve(rewritten)
        response = self._generator.generate(
            messages=messages,
            context=ContextRetriever.format(chunks),
        )

        self._conversations.append(chat_id, question, response)
        logger.info(
            "RAG answer done: chat=%s, answer_len=%d", chat_id, len(response)
        )
        return response

    def reset(self, chat_id: int) -> None:
        """Очищает историю RAG-диалога для чата."""
        self._conversations.clear(chat_id)
