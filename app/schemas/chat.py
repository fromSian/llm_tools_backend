from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator, model_validator


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty or whitespace-only.")
        return v


class ChatRequest(BaseModel):
    model: str = "default"
    system_prompt: str | None = None
    messages: list[ChatMessage]

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        if not v:
            raise ValueError("messages must contain at least one message.")
        return v

    @field_validator("system_prompt")
    @classmethod
    def system_prompt_not_whitespace(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("system_prompt cannot be whitespace-only.")
        return v


class ChatResponse(BaseModel):
    reply: str


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
