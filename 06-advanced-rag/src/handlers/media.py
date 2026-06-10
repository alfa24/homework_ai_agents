"""Утилиты загрузки медиа из Telegram."""
from __future__ import annotations

import base64

from aiogram.types import Message


async def download_bytes(message: Message, file_id: str) -> bytes:
    info = await message.bot.get_file(file_id)
    buffer = await message.bot.download_file(info.file_path)
    return buffer.getvalue()


async def download_image(message: Message) -> tuple[str, str]:
    if message.photo:
        file_id = message.photo[-1].file_id
        mime = "image/jpeg"
    else:
        file_id = message.document.file_id
        mime = message.document.mime_type or "image/jpeg"
    raw = await download_bytes(message, file_id)
    return base64.b64encode(raw).decode("utf-8"), mime
