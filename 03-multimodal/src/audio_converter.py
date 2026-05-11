"""Конвертация аудио ogg → wav через ffmpeg."""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile

from exceptions import AudioConversionError

logger = logging.getLogger(__name__)

FFMPEG_TIMEOUT_SEC = 30


class AudioConverter:
    """Обёртка над ffmpeg для приведения аудио к формату, который понимает провайдер LLM.

    RouterAI принимает только 'wav' и 'mp3', Telegram присылает 'ogg'.
    """

    async def ogg_to_wav(self, ogg_bytes: bytes) -> bytes:
        src_path = self._write_temp(ogg_bytes, suffix=".ogg")
        wav_path = src_path[:-4] + ".wav"
        try:
            await self._run_ffmpeg(src_path, wav_path)
            with open(wav_path, "rb") as f:
                return f.read()
        finally:
            self._cleanup(src_path, wav_path)

    @staticmethod
    def _write_temp(data: bytes, suffix: str) -> str:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            return tmp.name

    @staticmethod
    async def _run_ffmpeg(src: str, dst: str) -> None:
        process = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", src,
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            dst,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(process.communicate(), FFMPEG_TIMEOUT_SEC)
        except asyncio.TimeoutError as exc:
            process.kill()
            raise AudioConversionError("ffmpeg timeout") from exc

        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            logger.error("ffmpeg failed: %s", detail)
            raise AudioConversionError(f"ffmpeg exit {process.returncode}")

    @staticmethod
    def _cleanup(*paths: str) -> None:
        for path in paths:
            if os.path.exists(path):
                os.unlink(path)
