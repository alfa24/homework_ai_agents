"""Точка входа: сборка зависимостей, запуск polling."""
from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiohttp import ClientSession
from aiohttp_socks import ProxyConnector
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

from audio_converter import AudioConverter
from config import Settings
from conversation_store import ConversationStore
from finance_service import FinanceService
from handlers import build_router
from llm_client import LLMClient
from rag.answer_generator import AnswerGenerator
from rag.context_retriever import ContextRetriever, RetrieverConfig
from rag.corpus_indexer import CorpusIndexer
from rag.document_source import (
    PdfDocumentSource,
    SberbankJsonDocumentSource,
)
from rag.embeddings_factory import create_embeddings
from rag.in_memory_history import InMemoryMessageHistory
from rag.query_rewriter import QueryRewriter
from rag_service import RagService
from report_formatter import ReportFormatter
from transaction_store import TransactionStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SocksSession(AiohttpSession):
    def __init__(self, proxy_url: str) -> None:
        super().__init__()
        self._proxy_url = proxy_url

    async def create_session(self) -> ClientSession:
        if self._session is None or self._session.closed:
            self._session = ClientSession(
                connector=ProxyConnector.from_url(self._proxy_url)
            )
        return self._session


def _build_telegram_session(proxy_url: str | None) -> AiohttpSession | None:
    if not proxy_url:
        return None
    if proxy_url.startswith(("socks4://", "socks5://", "socks5h://")):
        return SocksSession(proxy_url)
    return AiohttpSession(proxy=proxy_url)


def _build_bot(settings: Settings) -> Bot:
    session = _build_telegram_session(settings.proxy_url)
    if session is not None:
        logger.info("Using proxy: %s", settings.proxy_url)
        return Bot(token=settings.telegram_token, session=session)
    return Bot(token=settings.telegram_token)


def _build_service(settings: Settings) -> FinanceService:
    openai_client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    llm = LLMClient(settings, openai_client)
    return FinanceService(
        llm=llm,
        conversations=ConversationStore(),
        transactions=TransactionStore(),
        audio=AudioConverter(),
    )


def _build_rag_service(settings: Settings) -> RagService:
    embeddings = create_embeddings(
        settings.embeddings_provider,
        settings.model_embeddings,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    llm = ChatOpenAI(
        model=settings.model_chat_rag,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=0,
    )
    sources = [
        PdfDocumentSource(
            data_dir=settings.data_dir,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
        SberbankJsonDocumentSource(
            path=settings.data_dir / "sberbank_help_documents.json"
        ),
    ]
    indexer = CorpusIndexer(sources=sources, embeddings=embeddings)
    return RagService(
        indexer=indexer,
        rewriter=QueryRewriter(
            llm=llm, instruction=settings.query_transform_prompt
        ),
        retrieval_mode=settings.rag_retrieval_mode,
        retriever=ContextRetriever(
            indexer=indexer,
            config=RetrieverConfig(
                mode=settings.rag_retrieval_mode,
                semantic_k=settings.semantic_retriever_k,
                bm25_k=settings.bm25_retriever_k,
                hybrid_k=settings.hybrid_retriever_k,
                semantic_weight=settings.hybrid_semantic_weight,
                bm25_weight=settings.hybrid_bm25_weight,
                cross_encoder_model=settings.cross_encoder_model,
                reranker_top_k=settings.reranker_top_k,
            ),
        ),
        generator=AnswerGenerator(
            llm=llm, system_template=settings.answer_system_prompt
        ),
        conversations=InMemoryMessageHistory(),
    )


async def main() -> None:
    settings = Settings.load()
    bot = _build_bot(settings)
    service = _build_service(settings)
    formatter = ReportFormatter()
    rag_service = _build_rag_service(settings)

    dp = Dispatcher()
    dp.include_router(build_router(service, formatter, rag_service, settings, show_sources=settings.show_sources))

    try:
        logger.info("Auto-indexing RAG corpus at startup...")
        await asyncio.to_thread(rag_service.reindex)
        logger.info("RAG index ready: %d documents", rag_service.document_count)
    except Exception:
        logger.exception(
            "Auto-indexing failed; /ask will be unavailable until /index is called"
        )

    logger.info("Starting bot...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
