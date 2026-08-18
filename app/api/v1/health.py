from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter

from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", summary="Health check", description="Returns ok if the application is running.")
async def health():
    return {"status": "ok"}


@router.get("/ready", summary="Readiness check", description="Checks availability of required dependencies.")
async def ready():
    settings = get_settings()
    result: dict[str, str] = {}
    overall_ok = True

    # Check Ollama
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
        result["ollama"] = "ok" if resp.status_code == 200 else "error"
    except Exception:
        result["ollama"] = "error"
        overall_ok = False

    # STT and TTS are local — just report configured
    result["stt"] = settings.stt_provider
    result["tts"] = settings.tts_provider

    status_code = 200 if overall_ok else 503
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status_code,
        content={"status": "ok" if overall_ok else "degraded", **result},
    )
