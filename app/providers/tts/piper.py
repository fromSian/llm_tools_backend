from __future__ import annotations

import asyncio
import io
import logging
import subprocess
import wave
from functools import partial

from app.providers.tts.base import TTSProvider

logger = logging.getLogger(__name__)


class PiperTTSError(Exception):
    pass


class PiperProvider(TTSProvider):
    """TTS using Piper via subprocess."""

    def __init__(self, model: str = "", voice: str = "") -> None:
        self._default_model = model
        self._default_voice = voice

    def _synthesize_sync(self, text: str, voice: str | None, model: str | None) -> bytes:
        resolved_model = model or self._default_model
        resolved_voice = voice or self._default_voice

        cmd = ["piper", "--output_raw"]
        if resolved_model:
            cmd += ["--model", resolved_model]
        if resolved_voice:
            cmd += ["--voice", resolved_voice]

        try:
            result = subprocess.run(
                cmd,
                input=text.encode(),
                capture_output=True,
                timeout=60,
            )
        except FileNotFoundError as exc:
            raise PiperTTSError("piper executable not found.") from exc
        except subprocess.TimeoutExpired as exc:
            raise PiperTTSError("Piper synthesis timed out.") from exc

        if result.returncode != 0:
            raise PiperTTSError(f"Piper failed (code {result.returncode}): {result.stderr.decode()[:200]}")

        # Wrap raw PCM (16-bit, 22050 Hz, mono) in a WAV container
        raw_pcm = result.stdout
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(raw_pcm)
        return buf.getvalue()

    async def synthesize(self, text: str, voice: str | None = None, model: str | None = None) -> bytes:
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, partial(self._synthesize_sync, text, voice, model))
        except PiperTTSError:
            raise
        except Exception as exc:
            raise PiperTTSError(f"Synthesis failed: {exc}") from exc
