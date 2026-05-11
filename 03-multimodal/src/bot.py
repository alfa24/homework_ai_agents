import asyncio
import logging
from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiohttp import ClientSession
from aiohttp_socks import ProxyConnector
from handlers import router
from config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SocksSession(AiohttpSession):
    def __init__(self, proxy_url: str):
        super().__init__()
        self._proxy_url = proxy_url

    async def create_session(self) -> ClientSession:
        if self._session is None or self._session.closed:
            self._session = ClientSession(connector=ProxyConnector.from_url(self._proxy_url))
        return self._session


def _build_session(proxy_url: str | None) -> AiohttpSession | None:
    if not proxy_url:
        return None
    if proxy_url.startswith(("socks4://", "socks5://", "socks5h://")):
        return SocksSession(proxy_url)
    return AiohttpSession(proxy=proxy_url)


async def main():
    session = _build_session(config.PROXY_URL)
    if session is not None:
        logger.info(f"Using proxy: {config.PROXY_URL}")

    bot = Bot(token=config.TELEGRAM_TOKEN, session=session) if session else Bot(token=config.TELEGRAM_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)

    logger.info("Starting bot...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

