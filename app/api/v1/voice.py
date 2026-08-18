from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from app.dependencies import get_voice_service
from app.middleware.rate_limit import get_rate_limiter
from app.schemas.audio import VoiceChatTextResponse
from app.security.request_signature import verify_request_signature
from app.services.voice_service import VoiceService, VoiceServiceError
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

_CODE_TO_STATUS = {
    "INVALID_REQUEST": 400,
    "PAYLOAD_TOO_LARGE": 413,
    "UNSUPPORTED_MEDIA_TYPE": 415,
    "STT_EMPTY": 422,
    "MODEL_NOT_FOUND": 404,
    "OLLAMA_UNAVAILABLE": 502,
    "GATEWAY_TIMEOUT": 504,
    "INTERNAL_ERROR": 500,
}


async def _check_rate_limit(request: Request) -> None:
    settings = get_settings()
    if not settings.rate_limit_enabled:
        return
    limiter = get_rate_limiter()
    client_ip = request.client.host if request.client else "unknown"
    if not await limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail={"code": "RATE_LIMITED", "message": "Too many requests."},
        )


@router.post(
    "/chat",
    summary="Voice chat pipeline",
    description=(
        "Submit audio and receive audio back. "
        "Internally: audio → Whisper STT → Ollama → Piper TTS → audio. "
        "Set response_format=json to receive JSON instead of audio."
    ),
    responses={
        200: {
            "content": {"audio/wav": {}, "application/json": {}},
            "description": "WAV audio or JSON",
        },
        400: {"description": "Invalid request"},
        429: {"description": "Rate limited"},
    },
)
async def voice_chat(
    request: Request,
    audio: UploadFile = File(..., description="Audio file"),
    model: str = Form(default="default"),
    system_prompt: str | None = Form(default=None),
    voice: str = Form(default="default"),
    response_format: str = Form(default="audio"),
    _sig: None = Depends(verify_request_signature),
    service: VoiceService = Depends(get_voice_service),
):
    await _check_rate_limit(request)
    content_type = audio.content_type
    audio_bytes = await audio.read()
    try:
        transcription, reply_text, wav_bytes = await service.voice_chat(
            audio_bytes=audio_bytes,
            content_type=content_type,
            model=model,
            system_prompt=system_prompt,
            voice=voice,
        )
    except VoiceServiceError as exc:
        status = _CODE_TO_STATUS.get(exc.code, 500)
        raise HTTPException(status_code=status, detail={"code": exc.code, "message": str(exc)})

    if response_format == "json":
        return VoiceChatTextResponse(transcription=transcription, reply=reply_text)

    return Response(content=wav_bytes, media_type="audio/wav")
