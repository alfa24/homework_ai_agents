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


def _require_prompt(file_path: str, env_var: str) -> str:
    """Как `_load_prompt`, но валится при пустом значении (fail-fast)."""
    text = _load_prompt(file_path, env_var)
    if not text:
        raise ConfigError(
            f"Empty prompt: set {env_var} or provide non-empty file at {file_path}"
        )
    return text


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ConfigError(f"Missing required environment variable: {name}")
    return value


def _require_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"Invalid integer for {name}: {raw!r}") from exc


def _require_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"Invalid float for {name}: {raw!r}") from exc


def _require_positive_int(
    name: str, default: int, legacy_name: str | None = None
) -> int:
    value = _require_int(name, default)
    if legacy_name and (os.getenv(name) is None or os.getenv(name) == ""):
        value = _require_int(legacy_name, default)
    if value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return value


def _require_positive_float(name: str, default: float) -> float:
    value = _require_float(name, default)
    if value <= 0:
        raise ConfigError(f"{name} must be a positive float")
    return value


def _require_choice(name: str, default: str, choices: set[str]) -> str:
    value = os.getenv(name, default)
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ConfigError(f"Invalid value for {name}: {value!r}. Allowed: {allowed}")
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
    # RAG (Sprint 2)
    model_chat_rag: str
    model_embeddings: str
    data_dir: Path
    rag_retrieval_mode: str
    semantic_retriever_k: int
    bm25_retriever_k: int
    hybrid_semantic_weight: float
    hybrid_bm25_weight: float
    reranker_model: str
    reranker_top_n: int
    chunk_size: int
    chunk_overlap: int
    query_transform_prompt: str
    answer_system_prompt: str
    show_sources: bool
    # RAGAS (Sprint 3)
    ragas_llm_model: str
    ragas_embedding_model: str
    # Langfuse
    langfuse_secret_key: str
    langfuse_public_key: str
    langfuse_base_url: str

    @classmethod
    def load(cls) -> "Settings":
        model_text = os.getenv("MODEL_TEXT") or os.getenv("MODEL")
        if not model_text:
            raise ConfigError("Missing required environment variable: MODEL_TEXT")

        data_dir_raw = os.getenv("DATA_DIR", "data")
        data_dir = Path(data_dir_raw)
        if not data_dir.is_absolute():
            data_dir = PROJECT_ROOT / data_dir

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
            model_chat_rag=_require("MODEL_CHAT_RAG"),
            model_embeddings=_require("MODEL_EMBEDDINGS"),
            data_dir=data_dir,
            rag_retrieval_mode=_require_choice(
                "RAG_RETRIEVAL_MODE",
                "semantic",
                {"semantic", "hybrid", "hybrid_rerank"},
            ),
            semantic_retriever_k=_require_positive_int(
                "SEMANTIC_RETRIEVER_K", 4, legacy_name="RETRIEVER_K"
            ),
            bm25_retriever_k=_require_positive_int("BM25_RETRIEVER_K", 4),
            hybrid_semantic_weight=_require_positive_float(
                "HYBRID_SEMANTIC_WEIGHT", 0.5
            ),
            hybrid_bm25_weight=_require_positive_float("HYBRID_BM25_WEIGHT", 0.5),
            reranker_model=os.getenv(
                "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            reranker_top_n=_require_positive_int("RERANKER_TOP_N", 4),
            chunk_size=_require_int("CHUNK_SIZE", 1000),
            chunk_overlap=_require_int("CHUNK_OVERLAP", 200),
            query_transform_prompt=_require_prompt(
                os.getenv(
                    "RAG_QUERY_TRANSFORM_PROMPT_PATH",
                    "prompts/rag_query_transform.txt",
                ),
                "RAG_QUERY_TRANSFORM_PROMPT",
            ),
            answer_system_prompt=_require_prompt(
                os.getenv(
                    "RAG_ANSWER_SYSTEM_PROMPT_PATH",
                    "prompts/rag_answer_system.txt",
                ),
                "RAG_ANSWER_SYSTEM_PROMPT",
            ),
            show_sources=os.getenv("SHOW_SOURCES", "false").lower() in ("true", "1", "yes"),
            ragas_llm_model=os.getenv("MODEL_RAGAS_LLM", "openai/gpt-4o-mini"),
            ragas_embedding_model=os.getenv(
                "MODEL_RAGAS_EMBEDDINGS", "openai/text-embedding-3-small"
            ),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            langfuse_base_url=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
        )
