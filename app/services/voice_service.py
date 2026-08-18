from __future__ import annotations

import logging

from app.schemas.chat import ChatMessage, ChatRequest
from app.services.chat_service import ChatService, ChatServiceError
from app.services.stt_service import STTService, STTServiceError
from app.services.tts_service import TTSService, TTSServiceError

logger = logging.getLogger(__name__)


class VoiceServiceError(Exception):
    def __init__(self, message: str, code: str = "VOICE_ERROR") -> None:
        super().__init__(message)
        self.code = code


class VoiceService:
    def __init__(self, stt: STTService, chat: ChatService, tts: TTSService) -> None:
        self._stt = stt
        self._chat = chat
        self._tts = tts

    async def voice_chat(
        self,
        audio_bytes: bytes,
        content_type: str | None = None,
        model: str = "default",
        system_prompt: str | None = None,
        voice: str = "default",
    ) -> tuple[str, str, bytes]:
        """Returns (transcription, reply_text, audio_bytes)."""
        try:
            transcription, _ = await self._stt.transcribe(audio_bytes, content_type)
        except STTServiceError as exc:
            raise VoiceServiceError(str(exc), exc.code) from exc

        if not transcription:
            raise VoiceServiceError("Could not transcribe audio. Please speak clearly.", "STT_EMPTY")

        chat_request = ChatRequest(
            model=model,
            system_prompt=system_prompt,
            messages=[ChatMessage(role="user", content=transcription)],
        )

        try:
            reply_text = await self._chat.chat(chat_request)
        except ChatServiceError as exc:
            raise VoiceServiceError(str(exc), exc.code) from exc

        try:
            audio = await self._tts.synthesize(reply_text, voice=voice)
        except TTSServiceError as exc:
            raise VoiceServiceError(str(exc), exc.code) from exc

        return transcription, reply_text, audio
