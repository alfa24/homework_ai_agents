"""Единый помощник для логирования и уведомления пользователя об ошибке."""
from __future__ import annotations

import logging

from aiogram.types import Message

logger = logging.getLogger(__name__)


async def reply_error(
    target: Message,
    text: str,
    *,
    log_message: str,
    chat_id: int,
    edit: bool = False,
) -> None:
    """Логирует текущее исключение и уведомляет пользователя.

    edit=True — редактирует переданное прогресс-сообщение, иначе отправляет новое.
    """
    logger.exception("%s: chat=%s", log_message, chat_id)
    if edit:
        await target.edit_text(text)
    else:
        await target.answer(text)
