from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from app.dependencies import get_chat_service
from app.middleware.rate_limit import get_rate_limiter
from app.schemas.chat import ChatRequest, ChatResponse
from app.security.request_signature import verify_request_signature
from app.services.chat_service import ChatService, ChatServiceError
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

_CODE_TO_STATUS = {
    "INVALID_REQUEST": 400,
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
    "/completions",
    response_model=ChatResponse,
    summary="Generate a chat completion",
    description=(
        "Send a conversation history to the local Ollama model and receive a reply. "
        "The client is responsible for maintaining conversation history."
    ),
    responses={
        400: {"description": "Invalid request"},
        401: {"description": "Invalid or missing request signature"},
        429: {"description": "Rate limited"},
        502: {"description": "Ollama unavailable"},
        504: {"description": "Ollama timeout"},
    },
)
async def chat_completions(
    body: ChatRequest,
    request: Request,
    _sig: None = Depends(verify_request_signature),
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    await _check_rate_limit(request)
    try:
        reply = await service.chat(body)
        return ChatResponse(reply=reply)
    except ChatServiceError as exc:
        status = _CODE_TO_STATUS.get(exc.code, 500)
        raise HTTPException(status_code=status, detail={"code": exc.code, "message": str(exc)})
