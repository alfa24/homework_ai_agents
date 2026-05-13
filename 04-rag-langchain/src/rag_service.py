"""RAG-сервис: индексация корпуса и ответы с retrieval (ответы — в итерации 2)."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from langchain_core.vectorstores import InMemoryVectorStore, VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import Settings
from document_loader import load_pdfs, load_sberbank_json
from exceptions import RagError

logger = logging.getLogger(__name__)


class RagService:
    """Управляет векторным хранилищем и генерацией ответов по корпусу."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._embeddings = OpenAIEmbeddings(
            model=settings.model_embeddings,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self._llm = ChatOpenAI(
            model=settings.model_chat_rag,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
        )
        self._vector_store: Optional[InMemoryVectorStore] = None
        self._document_count: int = 0
        self._last_indexed_at: Optional[datetime] = None

    @property
    def is_ready(self) -> bool:
        return self._vector_store is not None and self._document_count > 0

    @property
    def document_count(self) -> int:
        return self._document_count

    @property
    def last_indexed_at(self) -> Optional[datetime]:
        return self._last_indexed_at

    def reindex(self) -> int:
        """Перестраивает векторное хранилище из всех источников в data_dir."""
        data_dir = self._settings.data_dir
        if not data_dir.exists():
            raise RagError(f"Каталог с данными не найден: {data_dir}")

        pdf_docs = load_pdfs(
            data_dir,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )
        json_docs = load_sberbank_json(data_dir / "sberbank_help_documents.json")
        documents = pdf_docs + json_docs

        if not documents:
            raise RagError(f"Не найдено документов для индексации в {data_dir}")

        logger.info("Индексирую %d документов (PDF=%d, JSON=%d)",
                    len(documents), len(pdf_docs), len(json_docs))
        self._vector_store = InMemoryVectorStore.from_documents(
            documents, embedding=self._embeddings
        )
        self._document_count = len(documents)
        self._last_indexed_at = datetime.now()
        logger.info("Индексация завершена: %d документов", self._document_count)
        return self._document_count

    def _retriever(self) -> VectorStoreRetriever:
        if self._vector_store is None:
            raise RagError("Индекс не построен. Сначала выполните reindex().")
        return self._vector_store.as_retriever(
            search_kwargs={"k": self._settings.retriever_k}
        )

    def answer(self, question: str, history: list[dict] | None = None) -> str:
        """Генерирует ответ по корпусу. Реализация — в итерации 2."""
        raise NotImplementedError("answer() будет реализован в итерации 2")
