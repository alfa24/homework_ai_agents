"""Тесты fail-fast загрузки конфигурации Advanced RAG."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from config import Settings  # noqa: E402
from exceptions import ConfigError  # noqa: E402


class SettingsAdvancedRagTest(unittest.TestCase):
    def _base_env(self) -> dict[str, str]:
        return {
            "TELEGRAM_TOKEN": "telegram-token",
            "OPENAI_API_KEY": "openai-key",
            "MODEL_TEXT": "openai/gpt-4o-mini",
            "MODEL_CHAT_RAG": "openai/gpt-4o-mini",
            "MODEL_EMBEDDINGS": "openai/text-embedding-3-small",
            "RAG_QUERY_TRANSFORM_PROMPT": "rewrite question",
            "RAG_ANSWER_SYSTEM_PROMPT": "answer with context",
        }

    def _load(self, updates: dict[str, str] | None = None) -> Settings:
        env = self._base_env()
        if updates:
            env.update(updates)
        with patch.dict(os.environ, env, clear=True):
            return Settings.load()

    def test_defaults_are_loaded(self) -> None:
        settings = self._load()

        self.assertEqual(settings.rag_retrieval_mode, "semantic")
        self.assertEqual(settings.embeddings_provider, "openai")
        self.assertEqual(settings.semantic_retriever_k, 4)
        self.assertEqual(settings.bm25_retriever_k, 8)
        self.assertEqual(settings.hybrid_retriever_k, 8)
        self.assertEqual(settings.hybrid_semantic_weight, 0.5)
        self.assertEqual(settings.hybrid_bm25_weight, 0.5)
        self.assertEqual(
            settings.cross_encoder_model,
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        )
        self.assertEqual(settings.reranker_top_k, 4)
        self.assertEqual(settings.ragas_embeddings_provider, "openai")

    def test_valid_non_default_retrieval_mode_is_loaded(self) -> None:
        settings = self._load({"RAG_RETRIEVAL_MODE": "hybrid_rerank"})

        self.assertEqual(settings.rag_retrieval_mode, "hybrid_rerank")

    def test_invalid_retrieval_mode_raises_config_error(self) -> None:
        with self.assertRaises(ConfigError):
            self._load({"RAG_RETRIEVAL_MODE": "invalid"})

    def test_non_positive_top_k_and_top_n_raise_config_error(self) -> None:
        invalid_values = {
            "SEMANTIC_RETRIEVER_K": "0",
            "BM25_RETRIEVER_K": "-1",
            "RERANKER_TOP_K": "0",
        }

        for name, value in invalid_values.items():
            with self.subTest(name=name), self.assertRaises(ConfigError):
                self._load({name: value})

    def test_non_positive_hybrid_weights_raise_config_error(self) -> None:
        invalid_values = {
            "HYBRID_SEMANTIC_WEIGHT": "0",
            "HYBRID_BM25_WEIGHT": "-0.1",
        }

        for name, value in invalid_values.items():
            with self.subTest(name=name), self.assertRaises(ConfigError):
                self._load({name: value})

    def test_new_ragas_env_names_are_read(self) -> None:
        settings = self._load(
            {
                "MODEL_RAGAS_LLM": "openai/ragas-llm",
                "MODEL_RAGAS_EMBEDDINGS": "openai/ragas-embeddings",
                "RAGAS_LLM_MODEL": "old-llm-name",
                "RAGAS_EMBEDDING_MODEL": "old-embedding-name",
            }
        )

        self.assertEqual(settings.ragas_llm_model, "openai/ragas-llm")
        self.assertEqual(settings.ragas_embeddings_model, "openai/ragas-embeddings")


if __name__ == "__main__":
    unittest.main()
