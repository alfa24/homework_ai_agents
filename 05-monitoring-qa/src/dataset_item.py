"""Модель одной QA-пары синтетического датасета."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

QUESTION_KEY = "question"
ANSWER_KEY = "answer"
CONTEXT_KEY = "context"
SOURCE_KEY = "source"
PAGE_KEY = "page"
CATEGORY_KEY = "category"
URL_KEY = "url"


@dataclass(frozen=True, slots=True)
class DatasetItem:
    """Одна QA-пара для LangSmith/RAGAS."""

    question: str
    answer: str
    context: str
    source: str
    page: int | None = None
    category: str = ""
    url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            QUESTION_KEY: self.question,
            ANSWER_KEY: self.answer,
            CONTEXT_KEY: self.context,
            SOURCE_KEY: self.source,
            PAGE_KEY: self.page,
            CATEGORY_KEY: self.category,
            URL_KEY: self.url,
        }
