from __future__ import annotations

from abc import ABC, abstractmethod


class STTProvider(ABC):
    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, language: str | None = None) -> tuple[str, str | None]:
        """Return (text, detected_language)."""
        ...
