from __future__ import annotations

import logging

import httpx

from app.providers.chat.base import ChatProvider
from app.schemas.chat import ChatMessage

logger = logging.getLogger(__name__)

OLLAMA_CHAT_ENDPOINT = "/api/chat"


class OllamaError(Exception):
    pass


class OllamaUnavailableError(OllamaError):
    pass


class OllamaTimeoutError(OllamaError):
    pass


class OllamaModelNotFoundError(OllamaError):
    pass


class OllamaProvider(ChatProvider):
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def generate(self, messages: list[ChatMessage], model: str) -> str:
        payload = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
                response = await client.post(OLLAMA_CHAT_ENDPOINT, json=payload)
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError("Ollama request timed out.") from exc
        except httpx.ConnectError as exc:
            raise OllamaUnavailableError("Cannot connect to Ollama.") from exc
        except httpx.RequestError as exc:
            raise OllamaUnavailableError(f"Ollama request failed: {exc}") from exc

        if response.status_code == 404:
            raise OllamaModelNotFoundError(f"Model not found: {model}")
        if response.status_code != 200:
            raise OllamaError(f"Ollama returned status {response.status_code}: {response.text[:200]}")

        try:
            data = response.json()
            return data["message"]["content"]
        except (KeyError, ValueError) as exc:
            raise OllamaError(f"Unexpected Ollama response: {exc}") from exc
