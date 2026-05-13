"""RAG-сервис: индексация корпуса и диалоговые ответы с retrieval."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import InMemoryVectorStore, VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import Settings
from document_loader import load_pdfs, load_sberbank_json
from exceptions import RagError
from rag_conversation_store import RagConversationStore

logger = logging.getLogger(__name__)

_QUERY_TRANSFORM_INSTRUCTION = (
    "Transform last user message to a search query in Russian language "
    "according to the whole conversation history above to further retrieve "
    "the information relevant to the conversation. Try to thoroughly analyze "
    "all messages to generate the most relevant query. The longer result "
    "better than short. Let it be better more abstract than specific. "
    "Only respond with the query, nothing else."
)

_ANSWER_SYSTEM_TEMPLATE = (
    "You are an assistant for question-answering tasks. Answer the user's "
    "questions based on the conversation history and below context retrieved "
    "for the last question. Answer 'Я не нашёл ответа на ваш вопрос.' if you "
    "don't find any information in the context. Use three sentences maximum "
    "and keep the answer concise. Respond in Russian.\n\n"
    "Context retrieved for the last question:\n\n{context}"
)


def _format_chunks(chunks: list[Document]) -> str:
    return "\n\n".join(chunk.page_content for chunk in chunks)


class RagService:
    """Управляет векторным хранилищем и генерацией ответов по корпусу."""

    def __init__(
        self,
        settings: Settings,
        conversations: RagConversationStore,
    ) -> None:
        self._settings = settings
        self._conversations = conversations
        self._embeddings = OpenAIEmbeddings(
            model=settings.model_embeddings,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self._llm = ChatOpenAI(
            model=settings.model_chat_rag,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
        )
        self._vector_store: Optional[InMemoryVectorStore] = None
        self._document_count: int = 0
        self._last_indexed_at: Optional[datetime] = None

    @property
    def is_ready(self) -> bool:
        return self._vector_store is not None and self._document_count > 0

    @property
    def document_count(self) -> int:
        return self._document_count

    @property
    def last_indexed_at(self) -> Optional[datetime]:
        return self._last_indexed_at

    def reindex(self) -> int:
        """Перестраивает векторное хранилище из всех источников в data_dir."""
        data_dir = self._settings.data_dir
        if not data_dir.exists():
            raise RagError(f"Каталог с данными не найден: {data_dir}")

        pdf_docs = load_pdfs(
            data_dir,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )
        json_docs = load_sberbank_json(data_dir / "sberbank_help_documents.json")
        documents = pdf_docs + json_docs

        if not documents:
            raise RagError(f"Не найдено документов для индексации в {data_dir}")

        logger.info("Индексирую %d документов (PDF=%d, JSON=%d)",
                    len(documents), len(pdf_docs), len(json_docs))
        self._vector_store = InMemoryVectorStore.from_documents(
            documents, embedding=self._embeddings
        )
        self._document_count = len(documents)
        self._last_indexed_at = datetime.now()
        logger.info("Индексация завершена: %d документов", self._document_count)
        return self._document_count

    def answer(self, chat_id: int, question: str) -> str:
        """Диалоговый ответ по корпусу с query transformation и историей."""
        if self._vector_store is None:
            raise RagError("Индекс не построен. Сначала выполните /index.")

        history = self._conversations.get(chat_id)
        messages = history + [HumanMessage(content=question)]
        logger.info(
            "RAG answer: chat=%s, history_len=%d, question=%r",
            chat_id, len(history), question[:120],
        )

        try:
            rewritten = self._build_query_transform_chain().invoke(
                {"messages": messages}
            )
            logger.info(
                "RAG rewritten query: chat=%s, query=%r", chat_id, rewritten[:200]
            )

            chunks = self._retriever().invoke(rewritten)
            logger.info(
                "RAG retrieved: chat=%s, docs=%d", chat_id, len(chunks)
            )

            response = self._build_answer_chain().invoke(
                {
                    "messages": messages,
                    "context": _format_chunks(chunks),
                }
            )
        except Exception as exc:  # noqa: BLE001 — любой сбой LLM/retriever
            logger.exception("RAG answer failed for chat=%s", chat_id)
            raise RagError(f"Ошибка генерации ответа: {exc}") from exc

        self._conversations.append(chat_id, question, response)
        logger.info(
            "RAG answer done: chat=%s, answer_len=%d", chat_id, len(response)
        )
        return response

    def reset(self, chat_id: int) -> None:
        """Очищает историю RAG-диалога для чата."""
        self._conversations.clear(chat_id)

    def _retriever(self) -> VectorStoreRetriever:
        if self._vector_store is None:
            raise RagError("Индекс не построен. Сначала выполните reindex().")
        return self._vector_store.as_retriever(
            search_kwargs={"k": self._settings.retriever_k}
        )

    def _build_query_transform_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="messages"),
                ("user", _QUERY_TRANSFORM_INSTRUCTION),
            ]
        )
        return prompt | self._llm | StrOutputParser()

    def _build_answer_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _ANSWER_SYSTEM_TEMPLATE),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        return prompt | self._llm | StrOutputParser()
