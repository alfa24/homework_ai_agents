"""Ядро бизнес-логики: связывает LLMClient, хранилища и конвертацию аудио."""
from __future__ import annotations

import logging

from audio_converter import AudioConverter
from conversation_store import ConversationStore
from llm_client import LLMClient
from models import Transaction, TransactionResponse
from transaction_store import TransactionStore

logger = logging.getLogger(__name__)

IMAGE_HISTORY_PLACEHOLDER = "[Изображение: чек/скриншот]"


class FinanceService:
    """Координирует извлечение транзакций и обновление пользовательского состояния."""

    def __init__(
        self,
        llm: LLMClient,
        conversations: ConversationStore,
        transactions: TransactionStore,
        audio: AudioConverter,
    ) -> None:
        self._llm = llm
        self._conversations = conversations
        self._transactions = transactions
        self._audio = audio

    async def handle_text(self, chat_id: int, text: str) -> TransactionResponse:
        history = self._conversations.get(chat_id)
        response = await self._llm.extract_from_text(history, text)
        self._record(chat_id, user_content=text, response=response)
        return response

    async def handle_image(
        self, chat_id: int, image_base64: str, mime: str
    ) -> TransactionResponse:
        history = self._conversations.get(chat_id)
        response = await self._llm.extract_from_image(history, image_base64, mime)
        self._record(chat_id, user_content=IMAGE_HISTORY_PLACEHOLDER, response=response)
        return response

    async def handle_voice(
        self, chat_id: int, ogg_bytes: bytes
    ) -> tuple[str, TransactionResponse | None]:
        """Транскрибация + текстовый пайплайн. Возвращает (распознанный_текст, response|None)."""
        wav_bytes = await self._audio.ogg_to_wav(ogg_bytes)
        recognized = (await self._llm.transcribe_audio(wav_bytes)).strip()
        if not recognized:
            return "", None
        response = await self.handle_text(chat_id, recognized)
        return recognized, response

    def reset(self, chat_id: int) -> None:
        self._conversations.clear(chat_id)
        self._transactions.clear(chat_id)

    def transactions(self, chat_id: int) -> list[Transaction]:
        return self._transactions.get(chat_id)

    # ---------- Internals ----------

    def _record(
        self, chat_id: int, user_content: str, response: TransactionResponse
    ) -> None:
        if response.transactions:
            self._transactions.extend(chat_id, response.transactions)
            logger.info(
                "Extracted %d transactions for chat=%s",
                len(response.transactions), chat_id,
            )
        else:
            logger.info("No transactions extracted for chat=%s", chat_id)
        self._conversations.append(chat_id, "user", user_content)
        self._conversations.append(chat_id, "assistant", response.answer)
