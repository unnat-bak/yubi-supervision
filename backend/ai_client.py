"""Shared YUBI v3.0 model access with per-family reasoning and image-detail control.

All enrichers call ``AIClient.generate`` instead of building their own SDK client,
so every request is admitted and metered by ``AIBudget`` and gets the cheapest
correct settings for its model family:

- 2.x models take ``thinking_budget`` (0 disables thinking on Flash tiers).
- 3.x / 4.x models take ``thinkingLevel``; the installed SDK predates that field,
  so it is sent through ``HttpOptions.extra_body`` (merged into the raw request).
- Perception images use low media resolution on 3.x+ (280 tokens per image).
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types

from backend.ai_budget import AIBudget
from backend.config import Settings

THINKING_LEVELS = ("minimal", "low", "medium", "high")
_LEGACY_THINKING_BUDGETS = {"minimal": 0, "low": 1024, "medium": 4096, "high": -1}
_LEGACY_PRO_MIN_BUDGET = 128
_MEDIA_RESOLUTIONS = {
    "low": types.MediaResolution.MEDIA_RESOLUTION_LOW,
    "medium": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
    "high": types.MediaResolution.MEDIA_RESOLUTION_HIGH,
}
_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


class AIUnavailable(Exception):
    """Raised when a call is refused (budget, cooldown) or fails at the provider."""


@dataclass(frozen=True)
class AIResult:
    text: str
    data: Any
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    cost_usd: float
    latency_ms: float


def _bare_model(model: str) -> str:
    return model.lower().strip().removeprefix("models/")


def uses_thinking_budget(model: str) -> bool:
    return _bare_model(model).startswith(("gemini-1.", "gemini-2.", "gemini-2-"))


def supports_minimal_thinking(model: str) -> bool:
    name = _bare_model(model)
    return "flash-lite" in name or name.startswith(("gemini-3-flash", "gemini-3.6-flash"))


def thinking_settings(
    model: str, level: str
) -> tuple[types.ThinkingConfig | None, dict[str, Any] | None]:
    """Return (thinking_config, extra_body) for the requested reasoning level."""
    level = level if level in THINKING_LEVELS else "low"
    if uses_thinking_budget(model):
        budget = _LEGACY_THINKING_BUDGETS[level]
        if "pro" in _bare_model(model) and budget == 0:
            budget = _LEGACY_PRO_MIN_BUDGET
        return types.ThinkingConfig(thinking_budget=budget), None
    if level == "minimal" and not supports_minimal_thinking(model):
        level = "low"
    return None, {"generationConfig": {"thinkingConfig": {"thinkingLevel": level}}}


def media_resolution_for(model: str, setting: str) -> types.MediaResolution | None:
    """Low-detail images for 3.x+ perception; 2.x keeps its default tiling for boxes."""
    if uses_thinking_budget(model):
        return None
    return _MEDIA_RESOLUTIONS.get(setting.lower().strip())


def parse_json_text(text: str) -> Any:
    cleaned = _FENCE.sub("", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def _is_thinking_rejection(exc: BaseException) -> bool:
    return getattr(exc, "code", None) == 400 and "thinking" in str(exc).lower()


class AIClient:
    def __init__(self, settings: Settings, budget: AIBudget | None = None) -> None:
        self._settings = settings
        self.budget = budget or AIBudget(settings)
        self._lock = threading.Lock()
        self._client: genai.Client | None = None
        self._thinking_unsupported: set[str] = set()

    @property
    def available(self) -> bool:
        return self._settings.gemini_active

    @property
    def perception_model(self) -> str:
        return self._settings.gemini_model

    @property
    def insight_model(self) -> str:
        return self._settings.gemini_insight_model or self._settings.gemini_model

    def _sdk(self) -> genai.Client:
        with self._lock:
            if self._client is None:
                self._client = genai.Client(api_key=self._settings.gemini_api_key)
            return self._client

    def _config(
        self,
        model: str,
        *,
        json_mode: bool,
        temperature: float,
        thinking: str,
        media_resolution: str | None,
        system_instruction: str | None,
        timeout_sec: float,
        max_output_tokens: int | None,
    ) -> types.GenerateContentConfig:
        thinking_config: types.ThinkingConfig | None = None
        extra_body: dict[str, Any] | None = None
        if model not in self._thinking_unsupported:
            thinking_config, extra_body = thinking_settings(model, thinking)
        return types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json" if json_mode else None,
            system_instruction=system_instruction,
            thinking_config=thinking_config,
            media_resolution=(
                media_resolution_for(model, media_resolution) if media_resolution else None
            ),
            max_output_tokens=max_output_tokens,
            http_options=types.HttpOptions(
                timeout=int(timeout_sec * 1000),
                extra_body=extra_body,
            ),
        )

    def generate(
        self,
        *,
        feature: str,
        model: str,
        contents: list[Any] | str,
        json_mode: bool = False,
        temperature: float = 0.1,
        thinking: str = "minimal",
        media_resolution: str | None = None,
        system_instruction: str | None = None,
        timeout_sec: float | None = None,
        max_output_tokens: int | None = None,
        essential: bool = False,
    ) -> AIResult:
        if not self.available:
            raise AIUnavailable("YUBI v3.0 is offline.")
        reason = self.budget.acquire(feature, model, essential=essential)
        if reason is not None:
            raise AIUnavailable(reason)

        timeout = timeout_sec or self._settings.ai_request_timeout_sec
        start = time.perf_counter()
        response = None
        for attempt in range(2):
            config = self._config(
                model,
                json_mode=json_mode,
                temperature=temperature,
                thinking=thinking,
                media_resolution=media_resolution,
                system_instruction=system_instruction,
                timeout_sec=timeout,
                max_output_tokens=max_output_tokens,
            )
            try:
                response = self._sdk().models.generate_content(
                    model=model, contents=contents, config=config
                )
                break
            except Exception as exc:
                if attempt == 0 and _is_thinking_rejection(exc):
                    self._thinking_unsupported.add(model)
                    continue
                failure = self.budget.record_failure(feature, model, exc)
                print(f"YUBI v3.0 {feature} call failed ({failure.kind}): {exc}")
                raise AIUnavailable(failure.message) from exc

        latency_ms = (time.perf_counter() - start) * 1000.0
        usage = self.budget.record_usage(
            feature, model, getattr(response, "usage_metadata", None), latency_ms
        )
        text = (getattr(response, "text", None) or "").strip()
        data: Any = None
        if json_mode:
            try:
                data = parse_json_text(text or "{}")
            except json.JSONDecodeError as exc:
                raise AIUnavailable("YUBI v3.0 returned malformed JSON.") from exc
        return AIResult(
            text=text,
            data=data,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            thinking_tokens=usage.thinking_tokens,
            cost_usd=usage.cost_usd,
            latency_ms=latency_ms,
        )
