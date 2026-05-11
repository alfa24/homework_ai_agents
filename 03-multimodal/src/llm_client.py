"""Унифицированный клиент LLM: извлечение транзакций (текст/изображение) и транскрибация."""
from __future__ import annotations

import base64
import json
import logging
from typing import Sequence

from openai import APIError, AsyncOpenAI

from config import Settings
from exceptions import (
    LLMAudioUnsupportedError,
    LLMError,
    LLMImageUnsupportedError,
    LLMParseError,
)
from models import TransactionResponse

logger = logging.getLogger(__name__)

HISTORY_WINDOW = 10
DEFAULT_ANSWER_TEXT = "Обработал ваше сообщение."
DEFAULT_ANSWER_IMAGE = "Обработал изображение."


class LLMClient:
    """Один клиент для всех модальностей: text, image, audio."""

    def __init__(self, settings: Settings, client: AsyncOpenAI) -> None:
        self._settings = settings
        self._client = client

    # ---------- Public API ----------

    async def extract_from_text(
        self, history: Sequence[dict], user_text: str
    ) -> TransactionResponse:
        messages = [
            {"role": "system", "content": self._settings.system_prompt_text},
            *list(history)[-HISTORY_WINDOW:],
            {"role": "user", "content": user_text},
        ]
        raw = await self._call_structured(self._settings.model_text, messages)
        return self._parse(raw, default_answer=DEFAULT_ANSWER_TEXT)

    async def extract_from_image(
        self,
        history: Sequence[dict],
        image_base64: str,
        mime: str = "image/jpeg",
    ) -> TransactionResponse:
        model = self._settings.model_image or self._settings.model_text
        messages = [
            {"role": "system", "content": self._settings.system_prompt_image},
            *list(history)[-HISTORY_WINDOW:],
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image_base64}"},
                    },
                    {"type": "text", "text": "Извлеки транзакции из этого изображения"},
                ],
            },
        ]
        raw = await self._call_structured(model, messages)
        return self._parse(raw, default_answer=DEFAULT_ANSWER_IMAGE)

    async def transcribe_audio(self, wav_bytes: bytes, filename: str = "voice.wav") -> str:
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Расшифруй этот аудиофайл. Верни только текст расшифровки без дополнительных комментариев.",
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"},
                    },
                ],
            }
        ]
        logger.info(
            "Transcribing audio: filename=%s, size=%d bytes, model=%s",
            filename, len(wav_bytes), self._settings.model_audio,
        )
        try:
            response = await self._client.chat.completions.create(
                model=self._settings.model_audio,
                messages=messages,
            )
        except APIError as exc:
            raise self._classify_audio(exc) from exc

        text = response.choices[0].message.content or ""
        logger.info("Transcription result (length %d): %s", len(text), text[:500])
        return text

    # ---------- Internals ----------

    async def _call_structured(self, model: str, messages: list[dict]) -> str:
        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "transaction_response",
                "schema": TransactionResponse.model_json_schema(),
                "strict": True,
            },
        }
        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=schema,
            )
        except APIError as exc:
            raise self._classify_chat(exc) from exc

        raw = response.choices[0].message.content or ""
        logger.info(
            "Raw LLM response (length=%d, model=%s): %s",
            len(raw), model, raw[:1000] or "EMPTY",
        )
        if not raw.strip():
            raise LLMParseError("LLM returned empty response")
        return raw

    @staticmethod
    def _parse(raw: str, default_answer: str) -> TransactionResponse:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse LLM JSON: %s; content=%r", exc, raw[:500])
            raise LLMParseError("Invalid JSON in LLM response") from exc

        payload.setdefault("transactions", [])
        payload.setdefault("answer", default_answer)
        try:
            return TransactionResponse.model_validate(payload)
        except Exception as exc:  # pydantic validation error
            logger.error("Failed to validate TransactionResponse: %s", exc)
            raise LLMParseError("Invalid TransactionResponse payload") from exc

    @staticmethod
    def _classify_chat(error: APIError) -> LLMError:
        text = str(error).lower()
        if "image input" in text:
            return LLMImageUnsupportedError(str(error))
        return LLMError(str(error))

    @staticmethod
    def _classify_audio(error: APIError) -> LLMError:
        text = str(error).lower()
        if "404" in text or "not found" in text or "audio" in text:
            return LLMAudioUnsupportedError(str(error))
        return LLMError(str(error))
