"""Хендлеры RAG-ассистента: /index, /index_status, /ask, /ask_reset, /evaluate_dataset."""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from langchain_core.documents import Document

from config import Settings
from evaluation import RagEvaluator
from exceptions import RagError
from handlers.errors import reply_error
from handlers.texts import (
    ASK_NO_INDEX_HINT,
    ASK_RESET_DONE,
    ASK_USAGE_HINT,
    GENERIC_LLM_ERROR,
)
from rag_service import RagService

logger = logging.getLogger(__name__)


def _format_sources(chunks: list[Document]) -> str:
    """Форматирует список чанков в строку вида '📚 Источники: file.pdf (стр. 1, 3)'."""
    pages_by_source: dict[str, set[int]] = defaultdict(set)
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page")
        if page is not None:
            pages_by_source[source].add(int(page) + 1)  # PyPDF даёт 0-based
        else:
            pages_by_source[source]  # просто регистрируем источник
    parts = []
    for source, pages in sorted(pages_by_source.items()):
        if pages:
            page_list = ", ".join(str(p) for p in sorted(pages))
            parts.append(f"{source} (стр. {page_list})")
        else:
            parts.append(source)
    return "📚 Источники: " + "; ".join(parts)


def build_rag_router(
    rag_service: RagService,
    settings: Settings,
    *,
    show_sources: bool = False,
) -> Router:
    router = Router()

    @router.message(Command("index"))
    async def cmd_index(message: Message) -> None:
        chat_id = message.chat.id
        logger.info("Reindex requested by %s", chat_id)
        progress = await message.answer("⏳ Индексирую документы…")
        started = time.monotonic()
        try:
            count = await asyncio.to_thread(rag_service.reindex)
        except RagError as exc:
            await reply_error(
                progress, f"❌ Ошибка индексации: {exc}",
                log_message="Reindex failed", chat_id=chat_id, edit=True,
            )
            return
        except Exception as exc:  # noqa: BLE001 — сеть/LLM/прочее
            await reply_error(
                progress, f"❌ Ошибка индексации: {exc}",
                log_message="Unexpected reindex error", chat_id=chat_id, edit=True,
            )
            return
        duration = time.monotonic() - started
        logger.info(
            "Reindex done for chat=%s: docs=%d, duration=%.1fs",
            chat_id, count, duration,
        )
        await progress.edit_text(
            f"✅ Готово: {count} документов (за {duration:.1f} с)"
        )

    @router.message(Command("ask"))
    async def cmd_ask(message: Message) -> None:
        chat_id = message.chat.id
        parts = (message.text or "").split(maxsplit=1)
        question = parts[1].strip() if len(parts) > 1 else ""
        if not question:
            await message.answer(ASK_USAGE_HINT)
            return
        if not rag_service.is_ready:
            await message.answer(ASK_NO_INDEX_HINT)
            return
        logger.info("RAG ask from %s: %s...", chat_id, question[:100])
        progress = await message.answer("⏳ Ищу ответ в документах…")
        try:
            answer, chunks = await asyncio.to_thread(rag_service.answer, chat_id, question)
        except RagError as exc:
            await reply_error(
                progress, f"❌ {exc}",
                log_message="RAG error on ask", chat_id=chat_id, edit=True,
            )
            return
        except Exception:  # noqa: BLE001 — неизвестные сбои возвращаем пользователю мягко
            await reply_error(
                progress, GENERIC_LLM_ERROR,
                log_message="Unexpected RAG error", chat_id=chat_id, edit=True,
            )
            return
        if show_sources and chunks:
            answer = f"{answer}\n\n{_format_sources(chunks)}"
        await progress.edit_text(answer)

    @router.message(Command("ask_reset"))
    async def cmd_ask_reset(message: Message) -> None:
        chat_id = message.chat.id
        logger.info("RAG reset by %s", chat_id)
        rag_service.reset(chat_id)
        await message.answer(ASK_RESET_DONE)

    @router.message(Command("index_status"))
    async def cmd_index_status(message: Message) -> None:
        chat_id = message.chat.id
        logger.info("Index status requested by %s", chat_id)
        if not rag_service.is_ready:
            await message.answer("ℹ️ Индекс не построен. Выполните /index.")
            return
        last = rag_service.last_indexed_at
        last_str = last.strftime("%Y-%m-%d %H:%M:%S") if last else "—"
        await message.answer(
            "📊 Статус индекса:\n"
            f"• Готов: да\n"
            f"• Режим retrieval: {rag_service.retrieval_mode}\n"
            f"• Документов: {rag_service.document_count}\n"
            f"• Последняя индексация: {last_str}"
        )

    @router.message(Command("evaluate_dataset"))
    async def cmd_evaluate_dataset(message: Message) -> None:
        chat_id = message.chat.id
        logger.info("RAGAS evaluation requested by %s", chat_id)
        if not rag_service.is_ready:
            await message.answer("ℹ️ Индекс не построен. Сначала выполните /index.")
            return
        progress = await message.answer("⏳ Запускаю RAGAS evaluation… Это может занять несколько минут.")
        evaluator = RagEvaluator(settings, rag_service)
        try:
            avg_scores = await asyncio.to_thread(evaluator.evaluate)
        except FileNotFoundError as exc:
            await reply_error(
                progress, f"❌ {exc}",
                log_message="Evaluation dataset not found", chat_id=chat_id, edit=True,
            )
            return
        except Exception:  # noqa: BLE001
            await reply_error(
                progress, "❌ Ошибка evaluation. Подробности в логах.",
                log_message="RAGAS evaluation failed", chat_id=chat_id, edit=True,
            )
            return
        await progress.edit_text(RagEvaluator.format_results(avg_scores))

    return router
