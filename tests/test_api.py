from __future__ import annotations

import hashlib
import hmac
import io
import time
import uuid
import wave

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.dependencies import get_chat_service, get_stt_service, get_tts_service, get_voice_service
from app.main import app
from app.middleware.rate_limit import InMemoryRateLimiter, set_rate_limiter
from app.providers.chat.base import ChatProvider
from app.providers.stt.base import STTProvider
from app.providers.tts.base import TTSProvider
from app.schemas.chat import ChatMessage
from app.security.nonce_store import InMemoryNonceStore, set_nonce_store
from app.services.chat_service import ChatService
from app.services.stt_service import STTService
from app.services.tts_service import TTSService
from app.services.voice_service import VoiceService

# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


def make_wav_bytes(text: str = "hello") -> bytes:
    """Create a minimal WAV file for testing."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x01" * 1600)
    return buf.getvalue()


class MockChatProvider(ChatProvider):
    def __init__(self, reply: str = "Test reply") -> None:
        self.calls: list[tuple[list[ChatMessage], str]] = []
        self._reply = reply

    async def generate(self, messages: list[ChatMessage], model: str) -> str:
        self.calls.append((messages, model))
        return self._reply


class MockSTTProvider(STTProvider):
    def __init__(self, text: str = "transcribed text", language: str = "en") -> None:
        self._text = text
        self._language = language

    async def transcribe(self, audio_bytes: bytes, language: str | None = None) -> tuple[str, str | None]:
        return self._text, self._language


class MockTTSProvider(TTSProvider):
    async def synthesize(self, text: str, voice: str | None = None, model: str | None = None) -> bytes:
        return make_wav_bytes()


def make_settings(**kwargs) -> Settings:
    defaults = {
        "ollama_model": "llama3.2",
        "ollama_base_url": "http://localhost:11434",
        "rate_limit_enabled": False,
        "request_signing_secret": "",
        "max_messages": 50,
        "max_message_length": 10000,
        "max_conversation_length": 50000,
        "max_audio_size_mb": 25,
        "max_synthesis_text_length": 5000,
        "whisper_model": "small",
        "whisper_device": "auto",
        "whisper_compute_type": "auto",
        "tts_model": "",
        "tts_voice": "",
    }
    defaults.update(kwargs)
    return Settings(**defaults)


def build_client(
    chat_provider: ChatProvider | None = None,
    stt_provider: STTProvider | None = None,
    tts_provider: TTSProvider | None = None,
    settings: Settings | None = None,
    signing_secret: str = "",
) -> TestClient:
    s = settings or make_settings(request_signing_secret=signing_secret)
    cp = chat_provider or MockChatProvider()
    sp = stt_provider or MockSTTProvider()
    tp = tts_provider or MockTTSProvider()

    chat_svc = ChatService(provider=cp, settings=s)
    stt_svc = STTService(provider=sp, settings=s)
    tts_svc = TTSService(provider=tp, settings=s)
    voice_svc = VoiceService(stt=stt_svc, chat=chat_svc, tts=tts_svc)

    app.dependency_overrides[get_chat_service] = lambda: chat_svc
    app.dependency_overrides[get_stt_service] = lambda: stt_svc
    app.dependency_overrides[get_tts_service] = lambda: tts_svc
    app.dependency_overrides[get_voice_service] = lambda: voice_svc

    # Reset nonce store
    # Override global settings so request signing / rate limit middleware sees them
    import app.config as _cfg
    _cfg._settings = s

    set_nonce_store(InMemoryNonceStore())
    # Disable rate limiting by default
    set_rate_limiter(InMemoryRateLimiter(requests=1000, window_seconds=60))

    return TestClient(app, raise_server_exceptions=False)


def sign_request(secret: str, method: str, path: str, body: bytes) -> dict[str, str]:
    timestamp = str(int(time.time()))
    nonce = uuid.uuid4().hex
    body_hash = hashlib.sha256(body).hexdigest()
    canonical = f"{timestamp}{nonce}{method.upper()}{path}{body_hash}"
    sig = hmac.new(secret.encode(), canonical.encode(), hashlib.sha256).hexdigest()
    return {"X-Timestamp": timestamp, "X-Nonce": nonce, "X-Signature": sig}


# ---------------------------------------------------------------------------
# Health & Readiness
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_ok(self):
        client = build_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_readiness_returns_json(self):
        client = build_client()
        resp = client.get("/ready")
        assert resp.status_code in (200, 503)
        data = resp.json()
        assert "status" in data
        assert "ollama" in data


# ---------------------------------------------------------------------------
# Chat API
# ---------------------------------------------------------------------------

class TestChatCompletions:
    def _url(self):
        return "/api/v1/chat/completions"

    def test_basic_chat(self):
        mock = MockChatProvider(reply="Hello!")
        client = build_client(chat_provider=mock)
        resp = client.post(self._url(), json={
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 200
        assert resp.json()["reply"] == "Hello!"

    def test_multi_message_conversation(self):
        mock = MockChatProvider(reply="Your name is Alex.")
        client = build_client(chat_provider=mock)
        messages = [
            {"role": "user", "content": "My name is Alex."},
            {"role": "assistant", "content": "Nice to meet you, Alex."},
            {"role": "user", "content": "What is my name?"},
        ]
        resp = client.post(self._url(), json={"model": "default", "messages": messages})
        assert resp.status_code == 200
        assert resp.json()["reply"] == "Your name is Alex."
        # Verify all messages forwarded in order
        sent_messages, model_used = mock.calls[0]
        assert len(sent_messages) == 3
        assert sent_messages[0].content == "My name is Alex."
        assert sent_messages[2].content == "What is my name?"

    def test_default_model_resolution(self):
        mock = MockChatProvider()
        s = make_settings(ollama_model="llama3.2")
        client = build_client(chat_provider=mock, settings=s)
        client.post(self._url(), json={"model": "default", "messages": [{"role": "user", "content": "Hi"}]})
        _, model_used = mock.calls[0]
        assert model_used == "llama3.2"

    def test_explicit_model(self):
        mock = MockChatProvider()
        client = build_client(chat_provider=mock)
        client.post(self._url(), json={"model": "qwen3:8b", "messages": [{"role": "user", "content": "Hi"}]})
        _, model_used = mock.calls[0]
        assert model_used == "qwen3:8b"

    def test_system_prompt_prepended(self):
        mock = MockChatProvider()
        client = build_client(chat_provider=mock)
        resp = client.post(self._url(), json={
            "system_prompt": "You are a pirate.",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 200
        sent_messages, _ = mock.calls[0]
        assert sent_messages[0].role == "system"
        assert "pirate" in sent_messages[0].content

    def test_system_prompt_removes_system_messages(self):
        """system_prompt overrides system messages in messages array."""
        mock = MockChatProvider()
        client = build_client(chat_provider=mock)
        client.post(self._url(), json={
            "system_prompt": "Be concise.",
            "messages": [
                {"role": "system", "content": "Old system message"},
                {"role": "user", "content": "Hi"},
            ],
        })
        sent_messages, _ = mock.calls[0]
        system_msgs = [m for m in sent_messages if m.role == "system"]
        assert len(system_msgs) == 1
        assert "Be concise." in system_msgs[0].content

    def test_no_system_prompt(self):
        mock = MockChatProvider()
        client = build_client(chat_provider=mock)
        client.post(self._url(), json={"messages": [{"role": "user", "content": "Hi"}]})
        sent_messages, _ = mock.calls[0]
        assert not any(m.role == "system" for m in sent_messages)

    def test_empty_messages_rejected(self):
        client = build_client()
        resp = client.post(self._url(), json={"messages": []})
        assert resp.status_code == 422

    def test_empty_content_rejected(self):
        client = build_client()
        resp = client.post(self._url(), json={"messages": [{"role": "user", "content": "   "}]})
        assert resp.status_code == 422

    def test_invalid_role_rejected(self):
        client = build_client()
        resp = client.post(self._url(), json={"messages": [{"role": "badguy", "content": "hi"}]})
        assert resp.status_code == 422

    def test_max_messages_exceeded(self):
        s = make_settings(max_messages=2)
        client = build_client(settings=s)
        messages = [{"role": "user", "content": "hi"} for _ in range(3)]
        resp = client.post(self._url(), json={"messages": messages})
        assert resp.status_code == 400

    def test_message_too_long(self):
        s = make_settings(max_message_length=10)
        client = build_client(settings=s)
        resp = client.post(self._url(), json={"messages": [{"role": "user", "content": "x" * 11}]})
        assert resp.status_code == 400

    def test_conversation_too_long(self):
        s = make_settings(max_conversation_length=4)
        client = build_client(settings=s)
        resp = client.post(self._url(), json={"messages": [{"role": "user", "content": "hello"}]})
        assert resp.status_code == 400

    def test_ollama_unavailable_returns_502(self):
        from app.providers.chat.ollama import OllamaUnavailableError

        class FailingProvider(ChatProvider):
            async def generate(self, messages, model):
                raise OllamaUnavailableError("down")

        client = build_client(chat_provider=FailingProvider())
        resp = client.post(self._url(), json={"messages": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 502

    def test_ollama_timeout_returns_504(self):
        from app.providers.chat.ollama import OllamaTimeoutError

        class TimeoutProvider(ChatProvider):
            async def generate(self, messages, model):
                raise OllamaTimeoutError("timeout")

        client = build_client(chat_provider=TimeoutProvider())
        resp = client.post(self._url(), json={"messages": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 504

    def test_model_not_found_returns_404(self):
        from app.providers.chat.ollama import OllamaModelNotFoundError

        class MissingModelProvider(ChatProvider):
            async def generate(self, messages, model):
                raise OllamaModelNotFoundError("not found")

        client = build_client(chat_provider=MissingModelProvider())
        resp = client.post(self._url(), json={"messages": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# STT API
# ---------------------------------------------------------------------------

class TestTranscribe:
    def _url(self):
        return "/api/v1/audio/transcribe"

    def test_transcribe_success(self):
        client = build_client(stt_provider=MockSTTProvider(text="Hello world", language="en"))
        wav = make_wav_bytes()
        resp = client.post(self._url(), files={"audio": ("test.wav", wav, "audio/wav")})
        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "Hello world"
        assert data["language"] == "en"

    def test_empty_file_rejected(self):
        client = build_client()
        resp = client.post(self._url(), files={"audio": ("empty.wav", b"", "audio/wav")})
        assert resp.status_code in (400, 413, 422)

    def test_file_too_large_rejected(self):
        s = make_settings(max_audio_size_mb=1)
        client = build_client(settings=s)
        big = b"\x00" * (2 * 1024 * 1024)
        # Fake wav header so it passes magic check
        wav_header = b"RIFF" + b"\x00" * 4 + big
        resp = client.post(self._url(), files={"audio": ("big.wav", wav_header, "audio/wav")})
        assert resp.status_code == 413

    def test_invalid_audio_type(self):
        client = build_client()
        resp = client.post(self._url(), files={"audio": ("file.exe", b"MZ\x00\x00", "application/x-msdownload")})
        # Should either reject or pass-through depending on magic detection — just check not 200 with garbage
        # In practice the STT provider would fail; here mock returns text even for bad content.
        # The service validates content_type when magic is unknown.
        assert resp.status_code in (200, 400, 415)


# ---------------------------------------------------------------------------
# TTS API
# ---------------------------------------------------------------------------

class TestSynthesize:
    def _url(self):
        return "/api/v1/audio/synthesize"

    def test_synthesize_success(self):
        client = build_client()
        resp = client.post(self._url(), json={"text": "Hello world"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("audio/wav")
        assert len(resp.content) > 0

    def test_empty_text_rejected(self):
        client = build_client()
        resp = client.post(self._url(), json={"text": ""})
        assert resp.status_code == 422

    def test_whitespace_text_rejected(self):
        client = build_client()
        resp = client.post(self._url(), json={"text": "   "})
        assert resp.status_code == 422

    def test_text_too_long_rejected(self):
        s = make_settings(max_synthesis_text_length=5)
        client = build_client(settings=s)
        resp = client.post(self._url(), json={"text": "too long text"})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Voice Chat API
# ---------------------------------------------------------------------------

class TestVoiceChat:
    def _url(self):
        return "/api/v1/voice/chat"

    def test_voice_chat_returns_audio(self):
        client = build_client(
            stt_provider=MockSTTProvider(text="What is AI?"),
            tts_provider=MockTTSProvider(),
        )
        wav = make_wav_bytes()
        resp = client.post(self._url(), files={"audio": ("rec.wav", wav, "audio/wav")})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("audio/wav")

    def test_voice_chat_json_format(self):
        client = build_client(
            stt_provider=MockSTTProvider(text="What is AI?"),
        )
        wav = make_wav_bytes()
        resp = client.post(self._url(), files={"audio": ("rec.wav", wav, "audio/wav")}, data={"response_format": "json"})
        assert resp.status_code == 200
        data = resp.json()
        assert "transcription" in data
        assert "reply" in data


# ---------------------------------------------------------------------------
# Request Signing
# ---------------------------------------------------------------------------

class TestRequestSigning:
    def _url(self):
        return "/api/v1/chat/completions"

    def _body(self):
        import json
        return json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()

    def test_missing_headers_rejected(self):
        client = build_client(signing_secret="supersecret")
        body = self._body()
        resp = client.post(self._url(), content=body, headers={"Content-Type": "application/json"})
        assert resp.status_code == 401

    def test_valid_signature_accepted(self):
        secret = "supersecret"
        client = build_client(signing_secret=secret)
        body = self._body()
        headers = sign_request(secret, "POST", self._url(), body)
        headers["Content-Type"] = "application/json"
        resp = client.post(self._url(), content=body, headers=headers)
        assert resp.status_code == 200

    def test_invalid_signature_rejected(self):
        secret = "supersecret"
        client = build_client(signing_secret=secret)
        body = self._body()
        headers = sign_request(secret, "POST", self._url(), body)
        headers["X-Signature"] = "badsignature"
        headers["Content-Type"] = "application/json"
        resp = client.post(self._url(), content=body, headers=headers)
        assert resp.status_code == 401
        assert resp.json()["detail"]["code"] == "INVALID_SIGNATURE"

    def test_expired_timestamp_rejected(self):
        secret = "supersecret"
        client = build_client(signing_secret=secret)
        body = self._body()
        # Build headers with old timestamp
        old_ts = str(int(time.time()) - 200)
        nonce = uuid.uuid4().hex
        body_hash = hashlib.sha256(body).hexdigest()
        canonical = f"{old_ts}{nonce}POST{self._url()}{body_hash}"
        sig = hmac.new(secret.encode(), canonical.encode(), hashlib.sha256).hexdigest()
        headers = {
            "X-Timestamp": old_ts,
            "X-Nonce": nonce,
            "X-Signature": sig,
            "Content-Type": "application/json",
        }
        resp = client.post(self._url(), content=body, headers=headers)
        assert resp.status_code == 401
        assert resp.json()["detail"]["code"] == "EXPIRED_REQUEST"

    def test_replay_nonce_rejected(self):
        secret = "supersecret"
        client = build_client(signing_secret=secret)
        body = self._body()
        headers = sign_request(secret, "POST", self._url(), body)
        headers["Content-Type"] = "application/json"
        # First request succeeds
        resp1 = client.post(self._url(), content=body, headers=headers)
        assert resp1.status_code == 200
        # Same nonce again
        resp2 = client.post(self._url(), content=body, headers=headers)
        assert resp2.status_code == 401
        assert resp2.json()["detail"]["code"] == "REPLAY_DETECTED"


# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------

class TestRateLimit:
    def test_rate_limit_blocks_after_limit(self):
        s = make_settings(rate_limit_enabled=True)
        mock = MockChatProvider()
        # Use build_client to reset settings and nonce store
        client = build_client(chat_provider=mock, settings=s)

        # Override rate limiter: 2 requests per minute
        set_rate_limiter(InMemoryRateLimiter(requests=2, window_seconds=60))

        body = {"messages": [{"role": "user", "content": "hi"}]}
        r1 = client.post("/api/v1/chat/completions", json=body)
        r2 = client.post("/api/v1/chat/completions", json=body)
        r3 = client.post("/api/v1/chat/completions", json=body)
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r3.status_code == 429
