"""Индексатор корпуса: агрегирует `DocumentSource` в `InMemoryVectorStore`."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Sequence

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStoreRetriever

from exceptions import RagError
from rag.document_source import DocumentSource
from rag.index_state import IndexState

logger = logging.getLogger(__name__)


class CorpusIndexer:
    """Собирает документы из источников и строит векторное хранилище."""

    def __init__(
        self,
        sources: Sequence[DocumentSource],
        embeddings: Embeddings,
    ) -> None:
        self._sources = sources
        self._embeddings = embeddings
        self._vector_store: InMemoryVectorStore | None = None
        self._state: IndexState | None = None

    @property
    def state(self) -> IndexState | None:
        return self._state

    @property
    def is_ready(self) -> bool:
        return self._vector_store is not None

    def reindex(self) -> IndexState:
        documents = self._collect_documents()
        if not documents:
            raise RagError("Не найдено документов для индексации")

        logger.info("Индексирую %d документов", len(documents))
        self._vector_store = InMemoryVectorStore.from_documents(
            documents, embedding=self._embeddings
        )
        self._state = IndexState(
            document_count=len(documents),
            last_indexed_at=datetime.now(),
        )
        logger.info("Индексация завершена: %d документов", len(documents))
        return self._state

    def as_retriever(self, k: int) -> VectorStoreRetriever:
        if self._vector_store is None:
            raise RagError("Индекс не построен. Сначала выполните reindex().")
        return self._vector_store.as_retriever(search_kwargs={"k": k})

    def _collect_documents(self) -> list:
        documents: list = []
        for source in self._sources:
            documents.extend(source.load())
        return documents
