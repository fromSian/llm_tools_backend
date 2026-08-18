from __future__ import annotations

import logging

from app.config import Settings
from app.providers.stt.base import STTProvider
from app.providers.stt.whisper import WhisperSTTError

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_TYPES = {
    "audio/wav",
    "audio/wave",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/ogg",
    "audio/flac",
    "audio/webm",
    "audio/x-m4a",
    "application/octet-stream",
}

# Magic bytes for common audio formats
AUDIO_MAGIC = [
    (b"RIFF", "wav"),
    (b"\xff\xfb", "mp3"),
    (b"\xff\xf3", "mp3"),
    (b"\xff\xf2", "mp3"),
    (b"ID3", "mp3"),
    (b"OggS", "ogg"),
    (b"fLaC", "flac"),
    (b"\x1aE\xdf\xa3", "webm"),
]


class STTServiceError(Exception):
    def __init__(self, message: str, code: str = "STT_ERROR") -> None:
        super().__init__(message)
        self.code = code


def detect_audio_format(data: bytes) -> str | None:
    for magic, fmt in AUDIO_MAGIC:
        if data.startswith(magic):
            return fmt
    # MP4/M4A
    if len(data) >= 8 and data[4:8] in (b"ftyp", b"mdat", b"moov"):
        return "mp4"
    return None


class STTService:
    def __init__(self, provider: STTProvider, settings: Settings) -> None:
        self._provider = provider
        self._settings = settings

    def _validate_audio(self, data: bytes, content_type: str | None) -> None:
        max_bytes = self._settings.max_audio_size_mb * 1024 * 1024
        if not data:
            raise STTServiceError("Audio file is empty.", "INVALID_REQUEST")
        if len(data) > max_bytes:
            raise STTServiceError(
                f"Audio file exceeds maximum size of {self._settings.max_audio_size_mb} MB.",
                "PAYLOAD_TOO_LARGE",
            )
        fmt = detect_audio_format(data)
        if fmt is None:
            if content_type and content_type.split(";")[0].strip() not in SUPPORTED_AUDIO_TYPES:
                raise STTServiceError("Unsupported audio format.", "UNSUPPORTED_MEDIA_TYPE")
        return

    async def transcribe(self, audio_bytes: bytes, content_type: str | None = None, language: str | None = None) -> tuple[str, str | None]:
        self._validate_audio(audio_bytes, content_type)
        try:
            return await self._provider.transcribe(audio_bytes, language)
        except WhisperSTTError as exc:
            raise STTServiceError(str(exc), "STT_ERROR") from exc
        except Exception as exc:
            logger.exception("Unexpected STT error")
            raise STTServiceError("Transcription failed.", "INTERNAL_ERROR") from exc
