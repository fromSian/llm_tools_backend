from __future__ import annotations

import logging

from app.config import Settings
from app.providers.tts.base import TTSProvider
from app.providers.tts.piper import PiperTTSError

logger = logging.getLogger(__name__)


class TTSServiceError(Exception):
    def __init__(self, message: str, code: str = "TTS_ERROR") -> None:
        super().__init__(message)
        self.code = code


class TTSService:
    def __init__(self, provider: TTSProvider, settings: Settings) -> None:
        self._provider = provider
        self._settings = settings

    def _resolve_voice(self, voice: str) -> str | None:
        if voice == "default":
            return self._settings.tts_voice or None
        return voice

    def _resolve_model(self, model: str) -> str | None:
        if model == "default":
            return self._settings.tts_model or None
        return model

    async def synthesize(self, text: str, voice: str = "default", model: str = "default") -> bytes:
        if not text or not text.strip():
            raise TTSServiceError("Text cannot be empty.", "INVALID_REQUEST")
        if len(text) > self._settings.max_synthesis_text_length:
            raise TTSServiceError(
                f"Text exceeds maximum length of {self._settings.max_synthesis_text_length} characters.",
                "INVALID_REQUEST",
            )
        resolved_voice = self._resolve_voice(voice)
        resolved_model = self._resolve_model(model)
        try:
            return await self._provider.synthesize(text, voice=resolved_voice, model=resolved_model)
        except PiperTTSError as exc:
            raise TTSServiceError(str(exc), "TTS_ERROR") from exc
        except Exception as exc:
            logger.exception("Unexpected TTS error")
            raise TTSServiceError("Synthesis failed.", "INTERNAL_ERROR") from exc
