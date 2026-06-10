"""Извлечение релевантного контекста: semantic, hybrid, hybrid_rerank."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever

from exceptions import RagError
from rag.corpus_indexer import CorpusIndexer

logger = logging.getLogger(__name__)

SEMANTIC_MODE = "semantic"
HYBRID_MODE = "hybrid"
HYBRID_RERANK_MODE = "hybrid_rerank"


@dataclass(frozen=True, slots=True)
class RetrieverConfig:
    """Настройки retrieval из Settings."""
    mode: str
    semantic_k: int
    bm25_k: int
    hybrid_k: int
    semantic_weight: float
    bm25_weight: float


class ContextRetriever:
    """Поиск документов: semantic-only или hybrid (semantic + BM25)."""

    def __init__(self, indexer: CorpusIndexer, config: RetrieverConfig) -> None:
        self._indexer = indexer
        self._config = config

    def retrieve(self, query: str) -> list[Document]:
        try:
            if self._config.mode == SEMANTIC_MODE:
                chunks = self._semantic_retrieve(query)
            else:
                # hybrid и hybrid_rerank используют EnsembleRetriever
                chunks = self._hybrid_retrieve(query)
        except RagError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("ContextRetriever failed")
            raise RagError(f"Не удалось получить контекст: {exc}") from exc
        logger.info(
            "ContextRetriever[%s]: docs=%d", self._config.mode, len(chunks),
        )
        return chunks

    def _semantic_retrieve(self, query: str) -> list[Document]:
        retriever = self._indexer.as_retriever(k=self._config.semantic_k)
        return retriever.invoke(query)

    def _hybrid_retrieve(self, query: str) -> list[Document]:
        semantic = self._indexer.as_retriever(k=self._config.semantic_k)

        documents = self._indexer.documents
        if not documents:
            raise RagError("Индекс не построен. Сначала выполните reindex().")

        bm25 = BM25Retriever.from_documents(documents)
        bm25.k = self._config.bm25_k

        ensemble = EnsembleRetriever(
            retrievers=[semantic, bm25],
            weights=[self._config.semantic_weight, self._config.bm25_weight],
        )
        results = ensemble.invoke(query)
        # Ограничиваем финальный список hybrid_k документами
        return results[: self._config.hybrid_k]

    @staticmethod
    def format(chunks: list[Document]) -> str:
        return "\n\n".join(chunk.page_content for chunk in chunks)
