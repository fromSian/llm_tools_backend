# llm_tools_backend

A production-oriented local AI backend using Python, FastAPI, Ollama, Whisper, and Piper.

## Features

- **Chat API** — Multi-message stateless conversations via local Ollama
- **STT API** — Speech-to-text using local Whisper (faster-whisper)
- **TTS API** — Text-to-speech using local Piper
- **Voice Chat** — Audio-in → Whisper → Ollama → Piper → audio-out pipeline
- **Request signing** — HMAC-SHA256 replay/integrity protection
- **Rate limiting** — Sliding-window rate limiter (in-memory or Redis-backed)
- **Configurable** — All models, limits and secrets via environment variables

---

## Project Structure

```
app/
├── main.py              # FastAPI application
├── config.py            # Pydantic Settings
├── dependencies.py      # Dependency injection
├── api/v1/
│   ├── chat.py          # POST /api/v1/chat/completions
│   ├── audio.py         # POST /api/v1/audio/transcribe|synthesize
│   ├── voice.py         # POST /api/v1/voice/chat
│   └── health.py        # GET /health  GET /ready
├── schemas/
│   ├── chat.py
│   └── audio.py
├── services/
│   ├── chat_service.py
│   ├── stt_service.py
│   ├── tts_service.py
│   └── voice_service.py
├── providers/
│   ├── chat/{base,ollama}.py
│   ├── stt/{base,whisper}.py
│   └── tts/{base,piper}.py
├── security/
│   ├── request_signature.py
│   └── nonce_store.py
└── middleware/
    └── rate_limit.py
tests/
    test_api.py
```

---

## Quick Start (macOS / Ubuntu)

### 1. Clone and set up environment

```bash
git clone <repo>
cd llm_tools_backend

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env to set OLLAMA_MODEL, REQUEST_SIGNING_SECRET, etc.
```

### 3. Start dependencies

**Ollama**:
```bash
# macOS
brew install ollama && ollama serve
# Ubuntu
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```

Pull a model:
```bash
ollama pull llama3.2
```

**Piper** (TTS):
```bash
pip install piper-tts
# Download a voice model and set TTS_MODEL in .env
```

### 4. Run

```bash
uvicorn app.main:app --reload
```

API docs: http://localhost:8000/docs

---

## API Reference

### Health

```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl http://localhost:8000/ready
# {"status":"ok","ollama":"ok","stt":"whisper","tts":"piper"}
```

### Chat

```bash
curl -X POST http://localhost:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "system_prompt": "You are a helpful assistant.",
    "messages": [
      {"role": "user", "content": "My name is Alex."},
      {"role": "assistant", "content": "Nice to meet you, Alex."},
      {"role": "user", "content": "What is my name?"}
    ]
  }'
```

Response:
```json
{"reply": "Your name is Alex."}
```

### Speech-to-Text

```bash
curl -X POST http://localhost:8000/api/v1/audio/transcribe \
  -F "audio=@recording.wav" \
  -F "language=en"
```

Response:
```json
{"text": "Hello, how are you?", "language": "en"}
```

### Text-to-Speech

```bash
curl -X POST http://localhost:8000/api/v1/audio/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you?"}' \
  --output reply.wav
```

### Voice Chat

```bash
curl -X POST http://localhost:8000/api/v1/voice/chat \
  -F "audio=@question.wav" \
  -F "system_prompt=You are a helpful assistant." \
  --output answer.wav
```

JSON response format:
```bash
curl -X POST http://localhost:8000/api/v1/voice/chat \
  -F "audio=@question.wav" \
  -F "response_format=json"
```

---

## Request Signing

When `REQUEST_SIGNING_SECRET` is set, all API endpoints require signed requests.

Required headers:
- `X-Timestamp` — Unix timestamp (seconds)
- `X-Nonce` — Random UUID hex string
- `X-Signature` — HMAC-SHA256 signature

Signature construction:
```
canonical = timestamp + nonce + METHOD + path + SHA256(body)
signature = HMAC-SHA256(REQUEST_SIGNING_SECRET, canonical)
```

Python example:
```python
import hashlib, hmac, time, uuid, json

secret = "your-secret"
body = json.dumps({"messages": [{"role": "user", "content": "Hi"}]}).encode()
timestamp = str(int(time.time()))
nonce = uuid.uuid4().hex
body_hash = hashlib.sha256(body).hexdigest()
canonical = f"{timestamp}{nonce}POST/api/v1/chat/completions{body_hash}"
signature = hmac.new(secret.encode(), canonical.encode(), hashlib.sha256).hexdigest()

headers = {
    "X-Timestamp": timestamp,
    "X-Nonce": nonce,
    "X-Signature": signature,
    "Content-Type": "application/json",
}
```

---

## Error Format

All errors use:
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable description."
  }
}
```

Common codes: `INVALID_REQUEST`, `INVALID_SIGNATURE`, `EXPIRED_REQUEST`, `REPLAY_DETECTED`, `OLLAMA_UNAVAILABLE`, `MODEL_NOT_FOUND`, `RATE_LIMITED`.

---

## Docker

```bash
cp .env.example .env
docker compose up --build
```

To use an Ollama running on the host instead of Docker:
```env
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

For NVIDIA GPU support, uncomment the `deploy` section in `docker-compose.yml` and install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

---

## Testing

```bash
pytest
```

Tests mock all external providers (Ollama, Whisper, Piper) — no real services required.

---

## Configuration Reference

See `.env.example` for all options.

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Default model |
| `WHISPER_MODEL` | `small` | Whisper model size |
| `REQUEST_SIGNING_SECRET` | _(empty)_ | Enables request signing when set |
| `REDIS_URL` | _(empty)_ | Redis for nonce/rate-limit storage |
| `RATE_LIMIT_REQUESTS` | `30` | Max requests per window |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate limit window |
| `MAX_MESSAGES` | `50` | Max messages per request |
| `MAX_MESSAGE_LENGTH` | `10000` | Max characters per message |
| `MAX_CONVERSATION_LENGTH` | `50000` | Max total conversation characters |
| `MAX_AUDIO_SIZE_MB` | `25` | Max audio file size |
