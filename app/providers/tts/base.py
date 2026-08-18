from __future__ import annotations

from abc import ABC, abstractmethod


class TTSProvider(ABC):
    @abstractmethod
    async def synthesize(self, text: str, voice: str | None = None, model: str | None = None) -> bytes:
        """Return WAV audio bytes."""
        ...
