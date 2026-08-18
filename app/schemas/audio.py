from __future__ import annotations

from pydantic import BaseModel, field_validator


class TranscriptionResponse(BaseModel):
    text: str
    language: str | None = None


class SynthesisRequest(BaseModel):
    text: str
    voice: str = "default"
    model: str = "default"

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text cannot be empty.")
        return v


class VoiceChatTextResponse(BaseModel):
    transcription: str
    reply: str
