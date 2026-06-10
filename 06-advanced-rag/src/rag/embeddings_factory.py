"""Фабрика embeddings: создаёт провайдер по имени из конфига."""
from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings

from exceptions import ConfigError

logger = logging.getLogger(__name__)


def create_embeddings(
    provider: str,
    model: str,
    *,
    api_key: str = "",
    base_url: str = "",
) -> Embeddings:
    """Возвращает Embeddings-объект по имени провайдера.

    Args:
        provider: ``"openai"`` или ``"huggingface"``.
        model: имя модели (передаётся провайдеру как есть).
        api_key: API-ключ (только для ``openai``).
        base_url: базовый URL (только для ``openai``).
    """
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        logger.info("Embeddings provider=openai, model=%s", model)
        return OpenAIEmbeddings(model=model, api_key=api_key, base_url=base_url)

    if provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        logger.info("Embeddings provider=huggingface, model=%s", model)
        return HuggingFaceEmbeddings(model_name=model)

    raise ConfigError(f"Unknown embeddings provider: {provider!r}")
