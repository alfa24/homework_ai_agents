"""Хендлеры RAG-ассистента: /index, /index_status, /ask, /ask_reset."""
from __future__ import annotations

import asyncio
import logging
import time

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

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


def build_rag_router(rag_service: RagService) -> Router:
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
            answer = await asyncio.to_thread(rag_service.answer, chat_id, question)
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
            f"• Документов: {rag_service.document_count}\n"
            f"• Последняя индексация: {last_str}"
        )

    return router
