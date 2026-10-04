"""Small, dependency-free guards: per-client rate limiting and context-window trimming."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Sequence

from .providers import ChatMessage


class RateLimiter:
    """Token bucket per client key, kept in a bounded LRU so memory cannot grow without limit.
    In-process only: run one instance or put a shared limiter (e.g. Redis) in front for more."""

    def __init__(
        self,
        per_minute: int,
        burst: int | None = None,
        max_clients: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.rate = per_minute / 60.0
        self.capacity = float(burst or per_minute)
        self.max_clients = max_clients
        self.clock = clock
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()

    def allow(self, key: str) -> tuple[bool, float]:
        """Returns (allowed, seconds until the next token)."""
        now = self.clock()
        tokens, last = self._buckets.pop(key, (self.capacity, now))
        tokens = min(self.capacity, tokens + (now - last) * self.rate)
        allowed = tokens >= 1
        if allowed:
            tokens -= 1
        self._buckets[key] = (tokens, now)
        if len(self._buckets) > self.max_clients:
            self._buckets.popitem(last=False)
        return allowed, 0.0 if allowed else (1 - tokens) / self.rate


def trim_history(messages: Sequence[ChatMessage], max_chars: int) -> list[ChatMessage]:
    """Keep the most recent turns that fit in ``max_chars`` (always the latest message), starting on a
    user turn so the model never sees an orphaned assistant reply first."""
    kept: list[ChatMessage] = []
    used = 0
    for message in reversed(messages):
        if kept and used + len(message.content) > max_chars:
            break
        kept.append(message)
        used += len(message.content)
    kept.reverse()
    while len(kept) > 1 and kept[0].role != "user":
        kept.pop(0)
    return kept
