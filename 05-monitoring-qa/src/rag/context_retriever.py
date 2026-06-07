"""Извлечение релевантного контекста из корпуса для заданного запроса."""
from __future__ import annotations

import logging

from langchain_core.documents import Document

from exceptions import RagError
from rag.corpus_indexer import CorpusIndexer

logger = logging.getLogger(__name__)


class ContextRetriever:
    """Поиск top-K документов в векторном хранилище."""

    def __init__(self, indexer: CorpusIndexer, k: int) -> None:
        self._indexer = indexer
        self._k = k

    def retrieve(self, query: str) -> list[Document]:
        retriever = self._indexer.as_retriever(k=self._k)
        try:
            chunks = retriever.invoke(query)
        except Exception as exc:  # noqa: BLE001 — любой сбой retriever
            logger.exception("ContextRetriever failed")
            raise RagError(f"Не удалось получить контекст: {exc}") from exc
        logger.info("ContextRetriever: docs=%d", len(chunks))
        return chunks

    @staticmethod
    def format(chunks: list[Document]) -> str:
        return "\n\n".join(chunk.page_content for chunk in chunks)
