from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from app.dependencies import get_stt_service, get_tts_service
from app.middleware.rate_limit import get_rate_limiter
from app.schemas.audio import SynthesisRequest, TranscriptionResponse
from app.security.request_signature import verify_request_signature
from app.services.stt_service import STTService, STTServiceError
from app.services.tts_service import TTSService, TTSServiceError
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

_STT_CODE_TO_STATUS = {
    "INVALID_REQUEST": 400,
    "PAYLOAD_TOO_LARGE": 413,
    "UNSUPPORTED_MEDIA_TYPE": 415,
    "STT_EMPTY": 422,
    "INTERNAL_ERROR": 500,
}

_TTS_CODE_TO_STATUS = {
    "INVALID_REQUEST": 400,
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
    "/transcribe",
    response_model=TranscriptionResponse,
    summary="Transcribe audio to text",
    description="Upload an audio file and receive a text transcription using local Whisper.",
    responses={
        400: {"description": "Invalid audio"},
        413: {"description": "File too large"},
        415: {"description": "Unsupported media type"},
        429: {"description": "Rate limited"},
    },
)
async def transcribe(
    request: Request,
    audio: UploadFile = File(..., description="Audio file to transcribe"),
    language: str | None = Form(default=None, description="Language hint (e.g. 'en')"),
    model: str = Form(default="default", description="Model to use ('default' = server default)"),
    _sig: None = Depends(verify_request_signature),
    service: STTService = Depends(get_stt_service),
) -> TranscriptionResponse:
    await _check_rate_limit(request)
    content_type = audio.content_type
    audio_bytes = await audio.read()
    try:
        text, detected_lang = await service.transcribe(audio_bytes, content_type=content_type, language=language)
        return TranscriptionResponse(text=text, language=detected_lang or language)
    except STTServiceError as exc:
        status = _STT_CODE_TO_STATUS.get(exc.code, 500)
        raise HTTPException(status_code=status, detail={"code": exc.code, "message": str(exc)})


@router.post(
    "/synthesize",
    summary="Synthesize text to speech",
    description="Convert text to speech audio using local Piper TTS. Returns WAV audio.",
    responses={
        200: {"content": {"audio/wav": {}}, "description": "WAV audio"},
        400: {"description": "Invalid request"},
        429: {"description": "Rate limited"},
        500: {"description": "TTS error"},
    },
)
async def synthesize(
    body: SynthesisRequest,
    request: Request,
    _sig: None = Depends(verify_request_signature),
    service: TTSService = Depends(get_tts_service),
) -> Response:
    await _check_rate_limit(request)
    try:
        audio_bytes = await service.synthesize(body.text, voice=body.voice, model=body.model)
        return Response(content=audio_bytes, media_type="audio/wav")
    except TTSServiceError as exc:
        status = _TTS_CODE_TO_STATUS.get(exc.code, 500)
        raise HTTPException(status_code=status, detail={"code": exc.code, "message": str(exc)})
