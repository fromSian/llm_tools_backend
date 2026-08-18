from __future__ import annotations

import hashlib
import hmac
import logging
import time
import uuid
from typing import Optional

from fastapi import Header, HTTPException, Request

from app.config import get_settings
from app.security.nonce_store import get_nonce_store

logger = logging.getLogger(__name__)


def _canonical_request(timestamp: str, nonce: str, method: str, path: str, body_hash: str) -> str:
    return f"{timestamp}{nonce}{method.upper()}{path}{body_hash}"


def _compute_signature(secret: str, canonical: str) -> str:
    return hmac.new(secret.encode(), canonical.encode(), hashlib.sha256).hexdigest()


async def verify_request_signature(
    request: Request,
    x_timestamp: Optional[str] = Header(default=None),
    x_nonce: Optional[str] = Header(default=None),
    x_signature: Optional[str] = Header(default=None),
) -> None:
    settings = get_settings()
    if not settings.signing_enabled:
        return

    if not x_timestamp or not x_nonce or not x_signature:
        raise HTTPException(
            status_code=401,
            detail={"code": "MISSING_SIGNATURE", "message": "Request signing headers are required."},
        )

    # Validate timestamp
    try:
        ts = int(x_timestamp)
    except ValueError:
        raise HTTPException(
            status_code=401,
            detail={"code": "INVALID_REQUEST", "message": "X-Timestamp must be a Unix timestamp integer."},
        )

    now = int(time.time())
    tolerance = settings.request_timestamp_tolerance
    if abs(now - ts) > tolerance:
        raise HTTPException(
            status_code=401,
            detail={"code": "EXPIRED_REQUEST", "message": "Request timestamp is outside the allowed time window."},
        )

    # Validate nonce
    nonce_store = get_nonce_store()
    if await nonce_store.exists(x_nonce):
        raise HTTPException(
            status_code=401,
            detail={"code": "REPLAY_DETECTED", "message": "Request nonce has already been used."},
        )

    # Validate signature
    body = await request.body()
    body_hash = hashlib.sha256(body).hexdigest()
    canonical = _canonical_request(x_timestamp, x_nonce, request.method, request.url.path, body_hash)
    expected = _compute_signature(settings.request_signing_secret, canonical)

    if not hmac.compare_digest(expected, x_signature):
        raise HTTPException(
            status_code=401,
            detail={"code": "INVALID_SIGNATURE", "message": "Request signature is invalid."},
        )

    # Store nonce
    await nonce_store.add(x_nonce, ttl=tolerance * 2)
