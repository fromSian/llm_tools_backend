from __future__ import annotations

import asyncio
import io
import logging
from functools import partial

from app.providers.stt.base import STTProvider

logger = logging.getLogger(__name__)


class WhisperSTTError(Exception):
    pass


class WhisperProvider(STTProvider):
    def __init__(self, model: str = "small", device: str = "auto", compute_type: str = "auto") -> None:
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise WhisperSTTError("faster-whisper is not installed.") from exc

        device = self._device if self._device != "auto" else "cpu"
        compute_type = self._compute_type if self._compute_type != "auto" else "int8"
        logger.info("Loading Whisper model: %s (device=%s, compute_type=%s)", self._model_name, device, compute_type)
        self._model = WhisperModel(self._model_name, device=device, compute_type=compute_type)

    def _transcribe_sync(self, audio_bytes: bytes, language: str | None) -> tuple[str, str | None]:
        self._load_model()
        audio_io = io.BytesIO(audio_bytes)
        segments, info = self._model.transcribe(audio_io, language=language)
        text = "".join(segment.text for segment in segments).strip()
        detected = info.language if hasattr(info, "language") else None
        return text, detected

    async def transcribe(self, audio_bytes: bytes, language: str | None = None) -> tuple[str, str | None]:
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, partial(self._transcribe_sync, audio_bytes, language))
        except WhisperSTTError:
            raise
        except Exception as exc:
            raise WhisperSTTError(f"Transcription failed: {exc}") from exc
