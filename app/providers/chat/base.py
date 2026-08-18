from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.chat import ChatMessage


class ChatProvider(ABC):
    @abstractmethod
    async def generate(self, messages: list[ChatMessage], model: str) -> str:
        ...
