from __future__ import annotations

from functools import lru_cache

from app.config import Settings, get_settings
from app.providers.chat.ollama import OllamaProvider
from app.providers.stt.whisper import WhisperProvider
from app.providers.tts.piper import PiperProvider
from app.services.chat_service import ChatService
from app.services.stt_service import STTService
from app.services.tts_service import TTSService
from app.services.voice_service import VoiceService


@lru_cache
def get_chat_service() -> ChatService:
    settings = get_settings()
    provider = OllamaProvider(base_url=settings.ollama_base_url)
    return ChatService(provider=provider, settings=settings)


@lru_cache
def get_stt_service() -> STTService:
    settings = get_settings()
    provider = WhisperProvider(
        model=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
    return STTService(provider=provider, settings=settings)


@lru_cache
def get_tts_service() -> TTSService:
    settings = get_settings()
    provider = PiperProvider(model=settings.tts_model, voice=settings.tts_voice)
    return TTSService(provider=provider, settings=settings)


@lru_cache
def get_voice_service() -> VoiceService:
    return VoiceService(
        stt=get_stt_service(),
        chat=get_chat_service(),
        tts=get_tts_service(),
    )
