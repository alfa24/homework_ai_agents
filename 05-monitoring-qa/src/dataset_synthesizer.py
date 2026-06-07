"""CLI для синтеза и загрузки QA-датасета RAG.

Команды:
- generate: собрать датасет из PDF-чанков и готовых JSON Q&A;
- upload: загрузить сохранённый датасет в Langfuse без дублей по question.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langfuse import Langfuse

from config import PROJECT_ROOT, Settings
from dataset_item import (
    ANSWER_KEY,
    CATEGORY_KEY,
    CONTEXT_KEY,
    PAGE_KEY,
    QUESTION_KEY,
    SOURCE_KEY,
    URL_KEY,
    DatasetItem,
)

DATASET_DIR = PROJECT_ROOT / "datasets"
DATASET_PATH = DATASET_DIR / "05-rag-qa-dataset.json"
LANGFUSE_DATASET_NAME = "05-rag-qa-dataset"
PDF_SAMPLE_SIZE = 2
DEFAULT_ENCODING = "utf-8"
JSON_PATTERN = "*.json"
PDF_PATTERN = "*.pdf"
PROMPT_TEMPLATE = """Сгенерируй одну проверочную QA-пару по фрагменту документа.

Требования:
- question: самостоятельный вопрос на русском языке;
- answer: краткий эталонный ответ только по фактам из фрагмента;
- не добавляй факты, которых нет в тексте;
- верни только JSON без Markdown.

Фрагмент:
{context}

Формат JSON:
{{"question": "...", "answer": "..."}}
"""

logger = logging.getLogger(__name__)


class DatasetSynthesizer:
    """Генерирует QA-датасет из корпуса документов."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm = ChatOpenAI(
            model=settings.model_chat_rag,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
        )

    def generate(self) -> list[DatasetItem]:
        items = [*self._generate_from_pdfs(), *self._load_json_qa()]
        unique_items = self._deduplicate(items)
        self._save(unique_items)
        return unique_items

    def upload(self) -> int:
        items = self._load_dataset()
        langfuse = Langfuse()
        self._ensure_langfuse_dataset(langfuse)
        existing_questions = self._load_langfuse_questions(langfuse)

        uploaded_count = 0
        for item in items:
            question = item.question.strip()
            if not question or question in existing_questions:
                continue
            langfuse.create_dataset_item(
                dataset_name=LANGFUSE_DATASET_NAME,
                input={QUESTION_KEY: question},
                expected_output={ANSWER_KEY: item.answer},
                metadata={
                    SOURCE_KEY: item.source,
                    PAGE_KEY: item.page,
                    CATEGORY_KEY: item.category,
                    URL_KEY: item.url,
                    CONTEXT_KEY: item.context,
                },
            )
            existing_questions.add(question)
            uploaded_count += 1

        langfuse.flush()
        logger.info("Загружено новых примеров в Langfuse: %d", uploaded_count)
        return uploaded_count

    def _generate_from_pdfs(self) -> list[DatasetItem]:
        pdf_paths = sorted(self._settings.data_dir.glob(PDF_PATTERN))
        if not pdf_paths:
            logger.warning("PDF-файлы не найдены в %s", self._settings.data_dir)
            return []

        items: list[DatasetItem] = []
        for pdf_path in pdf_paths:
            chunks = self._load_pdf_chunks(pdf_path)[:PDF_SAMPLE_SIZE]
            for chunk in chunks:
                item = self._generate_from_chunk(chunk)
                if item is not None:
                    items.append(item)
        return items

    def _load_pdf_chunks(self, pdf_path: Path) -> list[Document]:
        raw_docs = PyPDFLoader(str(pdf_path)).load()
        for doc in raw_docs:
            doc.metadata[SOURCE_KEY] = pdf_path.name

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )
        chunks = splitter.split_documents(raw_docs)
        logger.info("PDF %s: выбрано %d чанков", pdf_path.name, min(len(chunks), PDF_SAMPLE_SIZE))
        return chunks

    def _generate_from_chunk(self, chunk: Document) -> DatasetItem | None:
        context = chunk.page_content.strip()
        if not context:
            return None

        response = self._llm.invoke(PROMPT_TEMPLATE.format(context=context))
        payload = self._parse_llm_json(str(response.content))
        question = str(payload.get(QUESTION_KEY, "")).strip()
        answer = str(payload.get(ANSWER_KEY, "")).strip()
        if not question or not answer:
            logger.warning("LLM вернула неполную QA-пару для %s", chunk.metadata.get(SOURCE_KEY, ""))
            return None

        return DatasetItem(
            question=question,
            answer=answer,
            context=context,
            source=str(chunk.metadata.get(SOURCE_KEY, "")),
            page=self._normalize_page(chunk.metadata.get(PAGE_KEY)),
        )

    def _load_json_qa(self) -> list[DatasetItem]:
        items: list[DatasetItem] = []
        for json_path in sorted(self._settings.data_dir.glob(JSON_PATTERN)):
            raw_items = self._read_json_items(json_path)
            for raw_item in raw_items:
                item = self._json_item_to_dataset_item(raw_item, json_path.name)
                if item is not None:
                    items.append(item)
        logger.info("Загружено готовых JSON Q&A: %d", len(items))
        return items

    def _read_json_items(self, json_path: Path) -> list[dict[str, Any]]:
        try:
            data = json.loads(json_path.read_text(encoding=DEFAULT_ENCODING))
        except json.JSONDecodeError:
            logger.exception("Пропускаю невалидный JSON: %s", json_path)
            return []
        if not isinstance(data, list):
            logger.warning("Пропускаю JSON не-массив: %s", json_path)
            return []
        return [item for item in data if isinstance(item, dict)]

    def _json_item_to_dataset_item(
        self,
        raw_item: dict[str, Any],
        source: str,
    ) -> DatasetItem | None:
        question = str(raw_item.get(QUESTION_KEY, "")).strip()
        answer = str(raw_item.get(ANSWER_KEY, "")).strip()
        context = str(raw_item.get("full_text") or answer).strip()
        if not question or not answer:
            return None
        return DatasetItem(
            question=question,
            answer=answer,
            context=context,
            source=source,
            category=str(raw_item.get(CATEGORY_KEY, "")).strip(),
            url=str(raw_item.get(URL_KEY, "")).strip(),
        )

    def _load_dataset(self) -> list[DatasetItem]:
        if not DATASET_PATH.exists():
            raise FileNotFoundError(
                f"Датасет не найден: {DATASET_PATH}. Сначала выполните make dataset."
            )

        data = json.loads(DATASET_PATH.read_text(encoding=DEFAULT_ENCODING))
        if not isinstance(data, list):
            raise ValueError(f"Ожидался JSON-массив в {DATASET_PATH}")
        return [self._dict_to_dataset_item(item) for item in data if isinstance(item, dict)]

    def _dict_to_dataset_item(self, item: dict[str, Any]) -> DatasetItem:
        return DatasetItem(
            question=str(item.get(QUESTION_KEY, "")).strip(),
            answer=str(item.get(ANSWER_KEY, "")).strip(),
            context=str(item.get(CONTEXT_KEY, "")).strip(),
            source=str(item.get(SOURCE_KEY, "")).strip(),
            page=self._parse_page(item.get(PAGE_KEY)),
            category=str(item.get(CATEGORY_KEY, "")).strip(),
            url=str(item.get(URL_KEY, "")).strip(),
        )

    def _save(self, items: list[DatasetItem]) -> None:
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        payload = [item.to_dict() for item in items]
        DATASET_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding=DEFAULT_ENCODING,
        )
        logger.info("Датасет сохранён: %s (%d примеров)", DATASET_PATH, len(items))

    def _ensure_langfuse_dataset(self, langfuse: Langfuse) -> None:
        try:
            langfuse.create_dataset(
                name=LANGFUSE_DATASET_NAME,
                description="QA-датасет для оценки RAG-пайплайна проекта 05-monitoring-qa",
            )
            logger.info("Создан датасет Langfuse: %s", LANGFUSE_DATASET_NAME)
        except Exception:
            logger.debug("Датасет Langfuse уже существует: %s", LANGFUSE_DATASET_NAME)

    def _load_langfuse_questions(self, langfuse: Langfuse) -> set[str]:
        questions: set[str] = set()
        try:
            dataset = langfuse.get_dataset(name=LANGFUSE_DATASET_NAME)
            for item in dataset.items:
                question = str(item.input.get(QUESTION_KEY, "")).strip()
                if question:
                    questions.add(question)
        except Exception:
            logger.debug("Не удалось загрузить items из Langfuse — дедупликация пропущена")
        return questions

    @staticmethod
    def _deduplicate(items: list[DatasetItem]) -> list[DatasetItem]:
        unique_items: list[DatasetItem] = []
        seen_questions: set[str] = set()
        for item in items:
            question = item.question.strip()
            if not question or question in seen_questions:
                continue
            seen_questions.add(question)
            unique_items.append(item)
        return unique_items[:5]

    @staticmethod
    def _parse_llm_json(text: str) -> dict[str, Any]:
        normalized = text.strip()
        if normalized.startswith("```"):
            normalized = normalized.strip("`").removeprefix("json").strip()
        try:
            payload = json.loads(normalized)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM вернула невалидный JSON: {text[:200]}") from exc
        if not isinstance(payload, dict):
            raise ValueError("LLM должна вернуть JSON-объект")
        return payload

    @staticmethod
    def _normalize_page(value: Any) -> int | None:
        page = DatasetSynthesizer._parse_page(value)
        if page is None:
            return None
        return page + 1

    @staticmethod
    def _parse_page(value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Синтез QA-датасета для RAG")
    parser.add_argument(
        "command",
        choices=("generate", "upload"),
        help="generate — сохранить JSON; upload — загрузить JSON в Langfuse",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = _parse_args()
    synthesizer = DatasetSynthesizer(Settings.load())

    if args.command == "generate":
        items = synthesizer.generate()
        print(f"Dataset saved to {DATASET_PATH}: {len(items)} examples")
    else:
        uploaded_count = synthesizer.upload()
        print(
            f"Uploaded {uploaded_count} new examples to Langfuse dataset "
            f"{LANGFUSE_DATASET_NAME}"
        )


if __name__ == "__main__":
    main()
