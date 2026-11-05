from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest
from backend.ai_budget import AIBudget, classify_error, estimate_cost_usd, price_per_million
from backend.ai_client import (
    media_resolution_for,
    parse_json_text,
    supports_minimal_thinking,
    thinking_settings,
)
from backend.config import Settings
from google.genai import types


class FakeProviderError(Exception):
    def __init__(self, code: int, status: str, message: str = "") -> None:
        super().__init__(f"{code} {status}. {message}")
        self.code = code
        self.status = status


def _settings(**overrides) -> Settings:
    base = {"gemini_enabled": False, "gemini_api_key": ""}
    base.update(overrides)
    return Settings(**base)


def _usage(prompt: int, output: int, thoughts: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=output,
        thoughts_token_count=thoughts,
        cached_content_token_count=0,
    )


def test_pricing_prefers_longest_prefix_and_intro_window() -> None:
    assert price_per_million("gemini-2.5-flash-lite") == (0.10, 0.40)
    assert price_per_million("gemini-2.5-flash") == (0.30, 2.50)
    assert price_per_million("models/gemini-3.5-flash-lite-preview") == (0.30, 2.50)
    assert price_per_million("gemini-3.8-flash", date(2026, 11, 5)) == (0.75, 3.75)
    assert price_per_million("gemini-3.8-flash", date(2027, 2, 1)) == (1.50, 7.50)
    assert price_per_million("some-future-model") == (2.00, 12.00)


def test_cost_counts_thinking_as_output() -> None:
    budget = AIBudget(_settings())
    record = budget.record_usage("scene", "gemini-2.5-flash", _usage(1_000_000, 0, 1_000_000), 10)
    assert record.cost_usd == pytest.approx(0.30 + 2.50)
    assert estimate_cost_usd("gemini-2.5-flash", 1_000_000, 0) == pytest.approx(0.30)


def test_session_budget_blocks_non_essential_calls() -> None:
    budget = AIBudget(_settings(ai_session_budget_usd=0.01))
    assert budget.acquire("scene", "gemini-2.5-flash") is None
    budget.record_usage("scene", "gemini-2.5-flash", _usage(100_000, 10_000), 5)
    assert "budget" in (budget.acquire("scene", "gemini-2.5-flash") or "").lower()
    assert budget.acquire("report", "gemini-2.5-flash", essential=True) is None
    snap = budget.snapshot()
    assert snap["paused"] is True
    assert snap["budget_used_pct"] == 100.0


def test_rate_cap_defers_calls() -> None:
    budget = AIBudget(_settings(ai_max_calls_per_min=2))
    assert budget.acquire("scene", "m") is None
    assert budget.acquire("scene", "m") is None
    assert "rate cap" in (budget.acquire("identity", "m") or "")


def test_billing_failure_pauses_everything_with_neutral_message() -> None:
    budget = AIBudget(_settings())
    error = FakeProviderError(402, "RESOURCE_EXHAUSTED", "Your prepayment credits are depleted.")
    failure = budget.record_failure("scene", "gemini-3.5-flash-lite", error)
    assert failure.kind == "billing"
    for text in (failure.message, budget.acquire("insights", "gemini-3.8-flash") or ""):
        assert "credits" in text
        assert "gemini" not in text.lower() and "google" not in text.lower()
    snap = budget.snapshot()
    assert snap["paused"] is True and snap["resume_in_sec"] > 800
    budget.reset()
    assert budget.acquire("scene", "gemini-3.5-flash-lite") is None


def test_error_classification_scopes() -> None:
    assert classify_error(FakeProviderError(429, "RESOURCE_EXHAUSTED")).kind == "quota"
    assert classify_error(FakeProviderError(403, "PERMISSION_DENIED")).cooldown_sec is None
    assert classify_error(FakeProviderError(404, "NOT_FOUND")).scope == "model"
    assert classify_error(TimeoutError("read timed out")).scope == "feature"


def test_missing_model_only_disables_that_model() -> None:
    budget = AIBudget(_settings())
    budget.record_failure("insights", "gemini-9-ultra", FakeProviderError(404, "NOT_FOUND"))
    assert budget.acquire("insights", "gemini-9-ultra") is not None
    assert budget.acquire("scene", "gemini-3.5-flash-lite") is None


def test_snapshot_never_exposes_model_names() -> None:
    budget = AIBudget(_settings())
    budget.acquire("scene", "gemini-3.5-flash-lite")
    budget.record_usage("scene", "gemini-3.5-flash-lite", _usage(1200, 80), 420)
    assert "gemini" not in repr(budget.snapshot()).lower()


def test_thinking_settings_per_model_family() -> None:
    config, extra = thinking_settings("gemini-2.5-flash", "minimal")
    assert config is not None and config.thinking_budget == 0 and extra is None
    config, _ = thinking_settings("gemini-2.5-pro", "minimal")
    assert config.thinking_budget == 128
    config, extra = thinking_settings("gemini-3.8-flash", "minimal")
    assert config is None
    assert extra == {"generationConfig": {"thinkingConfig": {"thinkingLevel": "low"}}}
    _, extra = thinking_settings("gemini-3.5-flash-lite", "minimal")
    assert extra["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "minimal"
    assert supports_minimal_thinking("gemini-3-flash-preview")


def test_media_resolution_only_lowered_for_modern_models() -> None:
    assert media_resolution_for("gemini-2.5-flash", "low") is None
    assert media_resolution_for("gemini-3.5-flash-lite", "low") == (
        types.MediaResolution.MEDIA_RESOLUTION_LOW
    )
    assert media_resolution_for("gemini-3.5-flash-lite", "default") is None


def test_parse_json_text_handles_fences_and_prose() -> None:
    assert parse_json_text('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_text('Here you go: {"b": [2]} thanks') == {"b": [2]}
