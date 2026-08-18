from __future__ import annotations

import logging

from app.config import Settings
from app.providers.chat.base import ChatProvider
from app.providers.chat.ollama import (
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from app.schemas.chat import ChatMessage, ChatRequest

logger = logging.getLogger(__name__)


class ChatServiceError(Exception):
    code: str = "CHAT_ERROR"

    def __init__(self, message: str, code: str = "CHAT_ERROR") -> None:
        super().__init__(message)
        self.code = code


class ChatService:
    def __init__(self, provider: ChatProvider, settings: Settings) -> None:
        self._provider = provider
        self._settings = settings

    def _resolve_model(self, model: str) -> str:
        if model == "default":
            return self._settings.ollama_model
        return model

    def _build_messages(self, request: ChatRequest) -> list[ChatMessage]:
        messages: list[ChatMessage] = []

        if request.system_prompt:
            # system_prompt takes precedence; strip any system messages from messages
            messages.append(ChatMessage(role="system", content=request.system_prompt))
            messages += [m for m in request.messages if m.role != "system"]
        else:
            messages = list(request.messages)

        return messages

    def _validate_limits(self, request: ChatRequest) -> None:
        s = self._settings
        if len(request.messages) > s.max_messages:
            raise ChatServiceError(
                f"Too many messages. Maximum is {s.max_messages}.",
                "INVALID_REQUEST",
            )
        for msg in request.messages:
            if len(msg.content) > s.max_message_length:
                raise ChatServiceError(
                    f"Message content exceeds maximum length of {s.max_message_length} characters.",
                    "INVALID_REQUEST",
                )
        total = sum(len(m.content) for m in request.messages)
        if total > s.max_conversation_length:
            raise ChatServiceError(
                f"Total conversation length exceeds maximum of {s.max_conversation_length} characters.",
                "INVALID_REQUEST",
            )

    async def chat(self, request: ChatRequest) -> str:
        self._validate_limits(request)
        model = self._resolve_model(request.model)
        messages = self._build_messages(request)

        try:
            return await self._provider.generate(messages, model)
        except OllamaModelNotFoundError as exc:
            raise ChatServiceError(str(exc), "MODEL_NOT_FOUND") from exc
        except OllamaTimeoutError as exc:
            raise ChatServiceError(str(exc), "GATEWAY_TIMEOUT") from exc
        except OllamaUnavailableError as exc:
            raise ChatServiceError(str(exc), "OLLAMA_UNAVAILABLE") from exc
        except Exception as exc:
            logger.exception("Unexpected error from chat provider")
            raise ChatServiceError("An unexpected error occurred.", "INTERNAL_ERROR") from exc
