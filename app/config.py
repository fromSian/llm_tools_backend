from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Application
    app_env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # STT
    stt_provider: str = "whisper"
    whisper_model: str = "small"
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"

    # TTS
    tts_provider: str = "piper"
    tts_model: str = ""
    tts_voice: str = ""

    # Security
    request_signing_secret: str = ""
    request_timestamp_tolerance: int = 60

    # Redis
    redis_url: str = ""

    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 30
    rate_limit_window_seconds: int = 60

    # Validation
    max_audio_size_mb: int = 25
    max_messages: int = 50
    max_message_length: int = 10000
    max_conversation_length: int = 50000
    max_synthesis_text_length: int = 5000

    # CORS
    cors_origins: str = ""

    @property
    def cors_origins_list(self) -> list[str]:
        if not self.cors_origins:
            return []
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def signing_enabled(self) -> bool:
        return bool(self.request_signing_secret)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
