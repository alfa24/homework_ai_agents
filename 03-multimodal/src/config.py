"""Конфигурация приложения: иммутабельный Settings c fail-fast валидацией."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from exceptions import ConfigError

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent


def _load_prompt(file_path: str, env_var: str | None = None) -> str:
    """Возвращает текст промпта: сначала из переменной окружения, затем из файла."""
    if env_var:
        value = os.getenv(env_var)
        if value:
            return value
    path = Path(file_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ConfigError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    telegram_token: str
    openai_api_key: str
    openai_base_url: str
    model_text: str
    model_image: str | None
    model_audio: str
    system_prompt: str
    proxy_url: str | None

    @classmethod
    def load(cls) -> "Settings":
        model_text = os.getenv("MODEL_TEXT") or os.getenv("MODEL")
        if not model_text:
            raise ConfigError("Missing required environment variable: MODEL_TEXT")

        return cls(
            telegram_token=_require("TELEGRAM_TOKEN"),
            openai_api_key=_require("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
            model_text=model_text,
            model_image=os.getenv("MODEL_IMAGE"),
            model_audio=os.getenv("MODEL_AUDIO", "openai/whisper-1"),
            system_prompt=_load_prompt(
                os.getenv("SYSTEM_PROMPT_PATH", "prompts/system_prompt.txt"),
                "SYSTEM_PROMPT",
            ),
            proxy_url=os.getenv("PROXY_URL"),
        )
