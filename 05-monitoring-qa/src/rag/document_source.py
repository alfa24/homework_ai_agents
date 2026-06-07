"""Источники документов для RAG-индексации.

Каждый источник инкапсулирует свою стратегию загрузки и подготовки документов.
Добавление нового источника = новый класс, реализующий `DocumentSource`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from exceptions import RagError

logger = logging.getLogger(__name__)


class DocumentSource(Protocol):
    """Источник документов для индексации."""

    def load(self) -> list[Document]:
        ...


class PdfDocumentSource:
    """Все PDF-файлы в каталоге, разрезанные на чанки."""

    def __init__(self, data_dir: Path, chunk_size: int, chunk_overlap: int) -> None:
        self._data_dir = data_dir
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def load(self) -> list[Document]:
        pdf_paths = sorted(self._data_dir.glob("*.pdf"))
        if not pdf_paths:
            logger.warning("PDF-файлы не найдены в %s", self._data_dir)
            return []

        raw_docs = self._load_raw(pdf_paths)
        chunks = self._split(raw_docs)
        logger.info("PDF разрезаны на %d чанков", len(chunks))
        return chunks

    def _load_raw(self, pdf_paths: list[Path]) -> list[Document]:
        raw_docs: list[Document] = []
        for path in pdf_paths:
            loaded = PyPDFLoader(str(path)).load()
            for doc in loaded:
                doc.metadata["source"] = path.name
            raw_docs.extend(loaded)
            logger.info("Загружен PDF %s: %d страниц", path.name, len(loaded))
        return raw_docs

    def _split(self, raw_docs: list[Document]) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        return splitter.split_documents(raw_docs)


class SberbankJsonDocumentSource:
    """Q&A из JSON-справки СберБанка: один item = один атомарный документ."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> list[Document]:
        if not self._path.exists():
            logger.warning("JSON с Q&A не найден: %s", self._path)
            return []

        items = self._read_items()
        documents = [self._to_document(item) for item in items if item.get("full_text")]
        logger.info(
            "Загружено %d Q&A документов из %s", len(documents), self._path.name
        )
        return documents

    def _read_items(self) -> list[dict]:
        raw = self._path.read_text(encoding="utf-8")
        try:
            items = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RagError(f"Невалидный JSON в {self._path}: {exc}") from exc
        if not isinstance(items, list):
            raise RagError(
                f"Ожидался массив в {self._path}, получено: {type(items).__name__}"
            )
        return items

    def _to_document(self, item: dict) -> Document:
        return Document(
            page_content=item["full_text"],
            metadata={
                "source": self._path.name,
                "url": item.get("url", ""),
                "category": item.get("category", ""),
                "type": item.get("type", ""),
                "question": item.get("question", ""),
            },
        )
