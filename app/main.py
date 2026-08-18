from __future__ import annotations

import logging
import logging.config

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import chat, audio, voice, health
from app.config import get_settings

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(
    title="Local AI Backend",
    description="Production-oriented local AI backend using Ollama, Whisper, and Piper.",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
)

# CORS
origins = settings.cors_origins_list or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logging.getLogger(__name__).exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "INTERNAL_ERROR", "message": "An internal error occurred."}},
    )


# Health routes at top level
app.include_router(health.router, tags=["Health"])

# API v1 routes
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(audio.router, prefix="/api/v1/audio", tags=["Audio"])
app.include_router(voice.router, prefix="/api/v1/voice", tags=["Voice"])
