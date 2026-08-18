from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class RateLimiter(ABC):
    @abstractmethod
    async def is_allowed(self, key: str) -> bool:
        ...


class InMemoryRateLimiter(RateLimiter):
    """Sliding window rate limiter backed by in-memory storage."""

    def __init__(self, requests: int, window_seconds: int) -> None:
        self._requests = requests
        self._window = window_seconds
        self._windows: dict[str, deque[float]] = defaultdict(deque)

    async def is_allowed(self, key: str) -> bool:
        now = time.time()
        window_start = now - self._window
        dq = self._windows[key]
        while dq and dq[0] < window_start:
            dq.popleft()
        if len(dq) >= self._requests:
            return False
        dq.append(now)
        return True


class RedisRateLimiter(RateLimiter):
    def __init__(self, redis_url: str, requests: int, window_seconds: int) -> None:
        self._redis_url = redis_url
        self._requests = requests
        self._window = window_seconds
        self._client = None

    async def _get_client(self):
        if self._client is None:
            import redis.asyncio as aioredis  # type: ignore[import-untyped]
            self._client = aioredis.from_url(self._redis_url, decode_responses=True)
        return self._client

    async def is_allowed(self, key: str) -> bool:
        client = await self._get_client()
        pipe_key = f"rl:{key}"
        now = time.time()
        window_start = now - self._window
        pipe = client.pipeline()
        pipe.zremrangebyscore(pipe_key, "-inf", window_start)
        pipe.zcard(pipe_key)
        pipe.zadd(pipe_key, {str(now): now})
        pipe.expire(pipe_key, self._window + 1)
        results = await pipe.execute()
        count_before_add = results[1]
        return count_before_add < self._requests


_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        from app.config import get_settings
        settings = get_settings()
        if settings.redis_url:
            _rate_limiter = RedisRateLimiter(settings.redis_url, settings.rate_limit_requests, settings.rate_limit_window_seconds)
        else:
            _rate_limiter = InMemoryRateLimiter(settings.rate_limit_requests, settings.rate_limit_window_seconds)
    return _rate_limiter


def set_rate_limiter(limiter: RateLimiter) -> None:
    global _rate_limiter
    _rate_limiter = limiter
