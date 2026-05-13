"""Точка входа: сборка зависимостей, запуск polling."""
from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiohttp import ClientSession
from aiohttp_socks import ProxyConnector
from openai import AsyncOpenAI

from audio_converter import AudioConverter
from config import Settings
from conversation_store import ConversationStore
from finance_service import FinanceService
from handlers import build_router
from llm_client import LLMClient
from rag_conversation_store import RagConversationStore
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


async def main() -> None:
    settings = Settings.load()
    bot = _build_bot(settings)
    service = _build_service(settings)
    formatter = ReportFormatter()
    rag_conversations = RagConversationStore()
    rag_service = RagService(settings, rag_conversations)

    dp = Dispatcher()
    dp.include_router(build_router(service, formatter, rag_service))

    logger.info("Starting bot...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
