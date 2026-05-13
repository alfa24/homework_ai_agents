"""Хендлеры финансового сервиса: /start, /balance, /transactions, медиа и текст."""
from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import Message

from exceptions import (
    AudioConversionError,
    LLMAudioUnsupportedError,
    LLMError,
    LLMImageUnsupportedError,
)
from finance_service import FinanceService
from handlers.errors import reply_error
from handlers.media import download_bytes, download_image
from handlers.texts import (
    AUDIO_CONVERSION_ERROR,
    GENERIC_LLM_ERROR,
    GENERIC_PROCESSING_ERROR,
    IMAGE_MODEL_HINT,
    MAX_MESSAGE_LENGTH,
    TEXT_ONLY_HINT,
    VOICE_EMPTY_RECOGNITION,
    VOICE_UNSUPPORTED_HINT,
    WELCOME_MESSAGE,
    too_long_message,
)
from report_formatter import ReportFormatter

logger = logging.getLogger(__name__)


def _image_filter(message: Message) -> bool:
    if message.photo:
        return True
    doc = message.document
    return bool(doc and doc.mime_type and doc.mime_type.startswith("image/"))


def build_finance_router(service: FinanceService, formatter: ReportFormatter) -> Router:
    router = Router()

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
        image_b64, mime = await download_image(message)
        try:
            response = await service.handle_image(chat_id, image_b64, mime)
        except LLMImageUnsupportedError:
            await message.answer(IMAGE_MODEL_HINT)
            return
        except LLMError:
            await reply_error(
                message, GENERIC_LLM_ERROR,
                log_message="LLM error on image", chat_id=chat_id,
            )
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
        ogg_bytes = await download_bytes(message, voice.file_id)
        try:
            recognized, response = await service.handle_voice(chat_id, ogg_bytes)
        except LLMAudioUnsupportedError:
            await message.answer(VOICE_UNSUPPORTED_HINT)
            return
        except AudioConversionError:
            await reply_error(
                message, AUDIO_CONVERSION_ERROR,
                log_message="ffmpeg conversion failed", chat_id=chat_id,
            )
            return
        except LLMError:
            await reply_error(
                message, GENERIC_LLM_ERROR,
                log_message="LLM error on voice", chat_id=chat_id,
            )
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
            await message.answer(TEXT_ONLY_HINT)
            return
        if len(message.text) > MAX_MESSAGE_LENGTH:
            await message.answer(too_long_message(len(message.text)))
            return

        chat_id = message.chat.id
        logger.info("Text from %s: %s...", chat_id, message.text[:100])
        try:
            response = await service.handle_text(chat_id, message.text)
        except LLMError:
            await reply_error(
                message, GENERIC_LLM_ERROR,
                log_message="LLM error on text", chat_id=chat_id,
            )
            return
        except Exception:  # noqa: BLE001 — неизвестные сбои возвращаем пользователю мягко
            await reply_error(
                message, GENERIC_PROCESSING_ERROR,
                log_message="Unexpected error on text", chat_id=chat_id,
            )
            return

        text = formatter.format_answer(response, service.transactions(chat_id))
        await message.answer(text)

    return router
