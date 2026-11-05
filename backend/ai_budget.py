"""YUBI v3.0 spend governor — token ledger, cost estimates, caps, circuit breaker.

Every model call in a session goes through one ``AIBudget``. It decides whether
a call may run (session USD cap, per-minute call cap, error cooldowns), records
token usage with an estimated cost, and turns provider failures into
vendor-neutral pause reasons so a depleted key never causes a retry storm.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import date
from typing import Any

from backend.config import Settings

# USD per 1M tokens (input, output). Output pricing also applies to thinking tokens.
_PRICES: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-3-flash": (0.50, 3.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.1-pro": (2.00, 12.00),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-4-argon": (2.00, 10.00),
}
_PRICE_PREFIXES = sorted(_PRICES, key=len, reverse=True)
_INTRO_PRICED = ("gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash")
_INTRO_PRICE = (0.75, 3.75)
_INTRO_PRICE_ENDS = date(2027, 1, 1)
_POST_INTRO_PRICE = (1.50, 7.50)
# Unknown models are costed at the top of the range so estimates never under-report.
_FALLBACK_PRICE = (2.00, 12.00)

_MAX_BACKOFF_SEC = 900.0


def _bare_model(model: str) -> str:
    return model.lower().strip().removeprefix("models/")


def price_per_million(model: str, today: date | None = None) -> tuple[float, float]:
    name = _bare_model(model)
    if name.startswith(_INTRO_PRICED):
        current = today or date.today()
        return _INTRO_PRICE if current < _INTRO_PRICE_ENDS else _POST_INTRO_PRICE
    for prefix in _PRICE_PREFIXES:
        if name.startswith(prefix):
            return _PRICES[prefix]
    return _FALLBACK_PRICE


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    today: date | None = None,
) -> float:
    input_price, output_price = price_per_million(model, today)
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


@dataclass(frozen=True)
class AIFailure:
    kind: str
    message: str
    cooldown_sec: float | None
    scope: str


def classify_error(exc: BaseException) -> AIFailure:
    """Map a provider exception to a pause policy and a vendor-neutral message."""
    code = getattr(exc, "code", None)
    status = str(getattr(exc, "status", "") or "").upper()
    text = str(exc).lower()
    if code == 402 or "credits are depleted" in text or "billing" in text:
        return AIFailure(
            "billing",
            "YUBI v3.0 paused — API credits are depleted. Top up billing for the "
            "configured API key; local analysis continues.",
            900.0,
            "global",
        )
    if code == 429 or status == "RESOURCE_EXHAUSTED":
        return AIFailure(
            "quota", "YUBI v3.0 rate-limited by the provider — backing off.", 60.0, "global"
        )
    if code in (401, 403) or status in {"PERMISSION_DENIED", "UNAUTHENTICATED"}:
        return AIFailure(
            "auth",
            "YUBI v3.0 API key was rejected — check the key in .env.",
            None,
            "global",
        )
    if code == 404 or status == "NOT_FOUND":
        return AIFailure(
            "model",
            "A configured YUBI v3.0 model tier is unavailable for this API key.",
            None,
            "model",
        )
    if code == 400 or status == "INVALID_ARGUMENT":
        return AIFailure("invalid", "YUBI v3.0 rejected the request.", 30.0, "feature")
    return AIFailure(
        "transient", "YUBI v3.0 temporarily unavailable — retrying shortly.", 15.0, "feature"
    )


@dataclass
class _Ledger:
    calls: int = 0
    failures: int = 0
    skipped: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    last_latency_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "failures": self.failures,
            "skipped": self.skipped,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "cached_tokens": self.cached_tokens,
            "cost_usd": round(self.cost_usd, 5),
            "last_latency_ms": round(self.last_latency_ms, 1),
        }


@dataclass(frozen=True)
class UsageRecord:
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    cost_usd: float


def _usage_int(usage: Any, name: str) -> int:
    value = getattr(usage, name, None) if usage is not None else None
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


class AIBudget:
    """Thread-safe per-session ledger and admission control for model calls."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._ledgers: dict[str, _Ledger] = {}
        self._recent_calls: deque[float] = deque()
        self._global_pause_until = 0.0
        self._global_pause_reason: str | None = None
        self._global_failures = 0
        self._feature_pause: dict[str, tuple[float, str]] = {}
        self._feature_failures: dict[str, int] = {}
        self._disabled_models: dict[str, str] = {}
        self._session_disabled_reason: str | None = None

    def reset(self) -> None:
        with self._lock:
            self._ledgers.clear()
            self._recent_calls.clear()
            self._global_pause_until = 0.0
            self._global_pause_reason = None
            self._global_failures = 0
            self._feature_pause.clear()
            self._feature_failures.clear()
            self._disabled_models.clear()
            self._session_disabled_reason = None

    def _ledger(self, feature: str) -> _Ledger:
        ledger = self._ledgers.get(feature)
        if ledger is None:
            ledger = _Ledger()
            self._ledgers[feature] = ledger
        return ledger

    def _total_cost(self) -> float:
        return sum(ledger.cost_usd for ledger in self._ledgers.values())

    def _blocked_reason(
        self, feature: str, model: str, now: float, essential: bool
    ) -> str | None:
        if self._session_disabled_reason:
            return self._session_disabled_reason
        model_reason = self._disabled_models.get(_bare_model(model))
        if model_reason:
            return model_reason
        if now < self._global_pause_until:
            return self._global_pause_reason
        feature_pause = self._feature_pause.get(feature)
        if feature_pause and now < feature_pause[0]:
            return feature_pause[1]
        if essential:
            return None
        cap = self._settings.ai_session_budget_usd
        if cap > 0 and self._total_cost() >= cap:
            return (
                f"Session AI budget reached (${cap:.2f}) — YUBI v3.0 paused; "
                "local analysis continues."
            )
        while self._recent_calls and now - self._recent_calls[0] > 60.0:
            self._recent_calls.popleft()
        if len(self._recent_calls) >= self._settings.ai_max_calls_per_min:
            return "YUBI v3.0 call rate cap reached — deferring."
        return None

    def blocked_reason(
        self, feature: str, model: str, *, essential: bool = False
    ) -> str | None:
        """Why a call would be refused right now (None when it may run)."""
        with self._lock:
            return self._blocked_reason(feature, model, time.time(), essential)

    def acquire(self, feature: str, model: str, *, essential: bool = False) -> str | None:
        """Admit a call (returns None) or return the reason it is blocked."""
        now = time.time()
        with self._lock:
            reason = self._blocked_reason(feature, model, now, essential)
            if reason is not None:
                return reason
            self._recent_calls.append(now)
            self._ledger(feature).calls += 1
            return None

    def record_usage(
        self, feature: str, model: str, usage: Any, latency_ms: float
    ) -> UsageRecord:
        input_tokens = _usage_int(usage, "prompt_token_count")
        candidates = _usage_int(usage, "candidates_token_count")
        thinking = _usage_int(usage, "thoughts_token_count")
        cached = _usage_int(usage, "cached_content_token_count")
        cost = estimate_cost_usd(model, input_tokens, candidates + thinking)
        with self._lock:
            ledger = self._ledger(feature)
            ledger.input_tokens += input_tokens
            ledger.output_tokens += candidates
            ledger.thinking_tokens += thinking
            ledger.cached_tokens += cached
            ledger.cost_usd += cost
            ledger.last_latency_ms = latency_ms
            self._feature_failures.pop(feature, None)
            self._feature_pause.pop(feature, None)
            self._global_failures = 0
        return UsageRecord(input_tokens, candidates, thinking, cost)

    def record_skip(self, feature: str, count: int = 1) -> None:
        with self._lock:
            self._ledger(feature).skipped += count

    def record_failure(self, feature: str, model: str, exc: BaseException) -> AIFailure:
        failure = classify_error(exc)
        now = time.time()
        with self._lock:
            self._ledger(feature).failures += 1
            if failure.cooldown_sec is None:
                if failure.scope == "model":
                    self._disabled_models[_bare_model(model)] = failure.message
                else:
                    self._session_disabled_reason = failure.message
                return failure
            if failure.scope == "global":
                self._global_failures += 1
                backoff = failure.cooldown_sec * (2 ** (self._global_failures - 1))
                self._global_pause_until = now + min(backoff, _MAX_BACKOFF_SEC)
                self._global_pause_reason = failure.message
            else:
                streak = self._feature_failures.get(feature, 0) + 1
                self._feature_failures[feature] = streak
                backoff = failure.cooldown_sec * (2 ** (streak - 1))
                self._feature_pause[feature] = (
                    now + min(backoff, _MAX_BACKOFF_SEC),
                    failure.message,
                )
        return failure

    def total_cost_usd(self) -> float:
        with self._lock:
            return self._total_cost()

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            totals = _Ledger()
            for ledger in self._ledgers.values():
                totals.calls += ledger.calls
                totals.failures += ledger.failures
                totals.skipped += ledger.skipped
                totals.input_tokens += ledger.input_tokens
                totals.output_tokens += ledger.output_tokens
                totals.thinking_tokens += ledger.thinking_tokens
                totals.cached_tokens += ledger.cached_tokens
                totals.cost_usd += ledger.cost_usd
            paused_reason = self._session_disabled_reason
            resume_in: float | None = None
            if paused_reason is None and now < self._global_pause_until:
                paused_reason = self._global_pause_reason
                resume_in = self._global_pause_until - now
            cap = self._settings.ai_session_budget_usd
            if paused_reason is None and cap > 0 and totals.cost_usd >= cap:
                paused_reason = f"Session AI budget reached (${cap:.2f})."
            payload = totals.as_dict()
            payload.update(
                {
                    "budget_usd": cap,
                    "budget_used_pct": (
                        round(min(100.0, totals.cost_usd / cap * 100.0), 1) if cap > 0 else 0.0
                    ),
                    "paused": paused_reason is not None,
                    "paused_reason": paused_reason,
                    "resume_in_sec": int(resume_in) if resume_in is not None else None,
                    "features": {
                        name: ledger.as_dict() for name, ledger in sorted(self._ledgers.items())
                    },
                }
            )
            return payload
