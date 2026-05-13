"""Загрузка документов для RAG: PDF (с чанкованием) и JSON Q&A (атомарно)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from exceptions import RagError

logger = logging.getLogger(__name__)


def load_pdfs(data_dir: Path, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Загружает все PDF из каталога и режет на чанки.

    Метаданные чанков: {source: <имя файла>, page: <номер страницы>}.
    """
    pdf_paths = sorted(data_dir.glob("*.pdf"))
    if not pdf_paths:
        logger.warning("PDF-файлы не найдены в %s", data_dir)
        return []

    raw_docs: list[Document] = []
    for path in pdf_paths:
        loader = PyPDFLoader(str(path))
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source"] = path.name
        raw_docs.extend(loaded)
        logger.info("Загружен PDF %s: %d страниц", path.name, len(loaded))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(raw_docs)
    logger.info("PDF разрезаны на %d чанков", len(chunks))
    return chunks


def load_sberbank_json(path: Path) -> list[Document]:
    """Загружает JSON Q&A как атомарные документы (один item = один Document)."""
    if not path.exists():
        logger.warning("JSON с Q&A не найден: %s", path)
        return []

    try:
        items = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RagError(f"Невалидный JSON в {path}: {exc}") from exc

    if not isinstance(items, list):
        raise RagError(f"Ожидался массив в {path}, получено: {type(items).__name__}")

    documents: list[Document] = []
    for idx, item in enumerate(items):
        full_text = item.get("full_text")
        if not full_text:
            logger.debug("Пропущен item #%d без full_text", idx)
            continue
        documents.append(
            Document(
                page_content=full_text,
                metadata={
                    "source": path.name,
                    "url": item.get("url", ""),
                    "category": item.get("category", ""),
                    "type": item.get("type", ""),
                    "question": item.get("question", ""),
                },
            )
        )

    logger.info("Загружено %d Q&A документов из %s", len(documents), path.name)
    return documents
