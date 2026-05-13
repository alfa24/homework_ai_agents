"""Тонкие Telegram-хендлеры. Вся бизнес-логика вынесена в FinanceService/RagService."""
from __future__ import annotations

import asyncio
import base64
import logging
import time

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import Message

from exceptions import (
    AudioConversionError,
    LLMAudioUnsupportedError,
    LLMError,
    LLMImageUnsupportedError,
    RagError,
)
from finance_service import FinanceService
from rag_service import RagService
from report_formatter import ReportFormatter

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 4000

WELCOME_MESSAGE = (
    "Привет! Я персональный финансовый советник.\n\n"
    "Я могу:\n"
    "• Извлекать транзакции из ваших сообщений\n"
    "• Вести учет доходов и расходов\n"
    "• Предоставлять советы по управлению финансами\n\n"
    "Используйте /start для начала нового диалога и очистки истории."
)
GENERIC_LLM_ERROR = (
    "Извините, произошла ошибка на стороне провайдера LLM. "
    "Пожалуйста, попробуйте еще раз через несколько секунд."
)
GENERIC_PROCESSING_ERROR = (
    "Произошла ошибка при обработке вашего сообщения. "
    "Попробуйте еще раз или используйте /start для начала нового диалога."
)
IMAGE_MODEL_HINT = (
    "Извините, используемая модель не поддерживает обработку изображений.\n\n"
    "Для работы с изображениями необходимо использовать vision-модель, например:\n"
    "• meta-llama/llama-3.2-11b-vision-instruct (OpenRouter)\n"
    "• llama3.2-vision (Ollama)\n\n"
    "Измените MODEL в файле .env на одну из этих моделей."
)
VOICE_UNSUPPORTED_HINT = (
    "Извините, текущий провайдер LLM не поддерживает транскрибацию аудио.\n\n"
    "Голосовые сообщения работают только через OpenRouter/OpenAI с моделью `openai/whisper-1`. "
    "Ollama не поддерживает audio transcriptions эндпоинт."
)
VOICE_EMPTY_RECOGNITION = (
    "Не удалось распознать речь в голосовом сообщении. Попробуйте записать ещё раз."
)
AUDIO_CONVERSION_ERROR = (
    "Произошла ошибка при обработке голосового сообщения. "
    "Попробуйте ещё раз или используйте /start для начала нового диалога."
)
ASK_USAGE_HINT = (
    "Используйте: /ask <ваш вопрос>\n"
    "Например: /ask Какие условия по потребительскому кредиту?"
)
ASK_NO_INDEX_HINT = "Индекс не построен. Сначала выполните /index."
ASK_RESET_DONE = "🧹 История диалога с RAG-ассистентом очищена."


def _image_filter(message: Message) -> bool:
    if message.photo:
        return True
    doc = message.document
    return bool(doc and doc.mime_type and doc.mime_type.startswith("image/"))


def build_router(
    service: FinanceService,
    formatter: ReportFormatter,
    rag_service: RagService,
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
            logger.exception("Reindex failed for chat=%s", chat_id)
            await progress.edit_text(f"❌ Ошибка индексации: {exc}")
            return
        except Exception as exc:  # noqa: BLE001 — сеть/LLM/прочее
            logger.exception("Unexpected reindex error for chat=%s", chat_id)
            await progress.edit_text(f"❌ Ошибка индексации: {exc}")
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
            logger.exception("RAG error on ask from %s", chat_id)
            await progress.edit_text(f"❌ {exc}")
            return
        except Exception:  # noqa: BLE001 — неизвестные сбои возвращаем пользователю мягко
            logger.exception("Unexpected RAG error for chat=%s", chat_id)
            await progress.edit_text(GENERIC_LLM_ERROR)
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

    @router.message(Command("start"))
    async def cmd_start(message: Message) -> None:
        service.reset(message.chat.id)
        logger.info("User %s started the bot", message.chat.id)
        await message.answer(WELCOME_MESSAGE)

    @router.message(Command("balance"))
    async def cmd_balance(message: Message) -> None:
        chat_id = message.chat.id
        logger.info("Balance requested by %s", chat_id)
        text = formatter.format_balance_report(service.transactions(chat_id))
        await message.answer(text)

    @router.message(Command("transactions"))
    async def cmd_transactions(message: Message) -> None:
        chat_id = message.chat.id
        logger.info("Transactions list requested by %s", chat_id)
        parts = formatter.format_transaction_list(service.transactions(chat_id))
        for part in parts:
            await message.answer(part)

    @router.message(_image_filter)
    async def on_image(message: Message) -> None:
        chat_id = message.chat.id
        logger.info("Image received from %s", chat_id)
        image_b64, mime = await _download_image(message)
        try:
            response = await service.handle_image(chat_id, image_b64, mime)
        except LLMImageUnsupportedError:
            await message.answer(IMAGE_MODEL_HINT)
            return
        except LLMError as exc:
            logger.exception("LLM error on image from %s: %s", chat_id, exc)
            await message.answer(GENERIC_LLM_ERROR)
            return

        text = formatter.format_answer(response, service.transactions(chat_id))
        await message.answer(text)

    @router.message(F.voice)
    async def on_voice(message: Message) -> None:
        chat_id = message.chat.id
        voice = message.voice
        logger.info(
            "Voice from %s: duration=%ss, mime=%s, size=%s",
            chat_id, voice.duration, voice.mime_type, voice.file_size,
        )
        ogg_bytes = await _download_bytes(message, voice.file_id)
        try:
            recognized, response = await service.handle_voice(chat_id, ogg_bytes)
        except LLMAudioUnsupportedError:
            await message.answer(VOICE_UNSUPPORTED_HINT)
            return
        except AudioConversionError:
            logger.exception("ffmpeg conversion failed for chat=%s", chat_id)
            await message.answer(AUDIO_CONVERSION_ERROR)
            return
        except LLMError as exc:
            logger.exception("LLM error on voice from %s: %s", chat_id, exc)
            await message.answer(GENERIC_LLM_ERROR)
            return

        if response is None:
            await message.answer(VOICE_EMPTY_RECOGNITION)
            return

        await message.answer(f"🎤 Распознано: {recognized}")
        text = formatter.format_answer(response, service.transactions(chat_id))
        await message.answer(text)

    @router.message()
    async def on_text(message: Message) -> None:
        if not message.text:
            await message.answer("Извините, я работаю только с текстовыми сообщениями.")
            return
        if len(message.text) > MAX_MESSAGE_LENGTH:
            await message.answer(
                f"Извините, ваше сообщение слишком длинное ({len(message.text)} символов). "
                f"Максимальная длина: {MAX_MESSAGE_LENGTH} символов."
            )
            return

        chat_id = message.chat.id
        logger.info("Text from %s: %s...", chat_id, message.text[:100])
        try:
            response = await service.handle_text(chat_id, message.text)
        except LLMError as exc:
            logger.exception("LLM error on text from %s: %s", chat_id, exc)
            await message.answer(GENERIC_LLM_ERROR)
            return
        except Exception:  # noqa: BLE001 — неизвестные сбои возвращаем пользователю мягко
            logger.exception("Unexpected error on text from %s", chat_id)
            await message.answer(GENERIC_PROCESSING_ERROR)
            return

        text = formatter.format_answer(response, service.transactions(chat_id))
        await message.answer(text)

    return router


async def _download_bytes(message: Message, file_id: str) -> bytes:
    info = await message.bot.get_file(file_id)
    buffer = await message.bot.download_file(info.file_path)
    return buffer.getvalue()


async def _download_image(message: Message) -> tuple[str, str]:
    if message.photo:
        file_id = message.photo[-1].file_id
        mime = "image/jpeg"
    else:
        file_id = message.document.file_id
        mime = message.document.mime_type or "image/jpeg"
    raw = await _download_bytes(message, file_id)
    return base64.b64encode(raw).decode("utf-8"), mime
