from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
from backend.ai_budget import AIBudget
from backend.ai_client import AIResult, AIUnavailable
from backend.config import Settings
from backend.gemini_vision import SceneChangeGate, scene_signature
from backend.insight_engine import InsightEngine
from backend.micro_expressions import Episode
from backend.session_report import (
    ANALYSIS_HEADING,
    apply_fixes,
    enrich_session_report_events,
    evidence_payload,
    render_analysis_markdown,
)

ONLINE = Settings(gemini_enabled=True, gemini_api_key="test-key", session_report_passes=3)


class FakeAI:
    """Stands in for AIClient: records calls and returns scripted payloads per feature."""

    available = True
    perception_model = "fake-perception"
    insight_model = "fake-insight"

    def __init__(self, responder, settings: Settings = ONLINE) -> None:
        self.budget = AIBudget(settings)
        self.calls: list[dict[str, Any]] = []
        self._responder = responder

    def generate(self, **kwargs: Any) -> AIResult:
        self.calls.append(kwargs)
        reply = self._responder(kwargs)
        if isinstance(reply, Exception):
            raise reply
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return AIResult(
            text=text,
            data=None if isinstance(reply, str) else reply,
            input_tokens=100,
            output_tokens=20,
            thinking_tokens=0,
            cost_usd=0.0001,
            latency_ms=5.0,
        )


def _episode(eid: str, at: float, kind: str = "micro", cues: list[str] | None = None) -> Episode:
    return Episode(
        id=eid, subject="A", kind=kind, label="Disgust flash", at=at, t=at,
        duration_ms=260, intensity=0.7, aus=["AU9", "AU10"], emotion="disgust",
        emotion_score=0.6, cues=cues or ["nose_wrinkle"],
    )


def _analysis() -> dict[str, Any]:
    episodes = [_episode("E0001", 12.4).as_dict(), _episode("E0002", 40.0, "macro").as_dict()]
    return {
        "session": {"session_id": "S1", "profile": {"label": "Interview"},
                    "source_label": "interview.mp4", "duration_sec": 95.0},
        "expressions": {
            "subjects": [{
                "id": "A", "observed_sec": 90.0, "calibrated": True,
                "episodes": {"micro": 1, "macro": 1, "behavior": 0},
                "blink": {"total": 30, "rate_per_min": 20.0, "long_closures": 0},
                "gaze_away_pct": 12.5, "head_nods": 3, "head_shakes": 1, "self_touch": 2,
                "pulse": {"mean": 74.0, "min": 70.0, "max": 81.0, "samples": 40},
                "affect": {"time_by_emotion_sec": {"neutral": 60.0, "happiness": 30.0}},
            }],
            "episodes": episodes,
            "timeline": {"A": [{"t": float(i), "v": 0.0, "a": 0.1} for i in range(200)]},
            "counts": {"micro": 1, "macro": 1, "behavior": 0},
            "cue_counts": {"nose_wrinkle": 2},
        },
        "insights": [{
            "id": "I001", "title": "Brief aversion at question 3", "observation": "AU9+AU10 flash",
            "interpretation": "May reflect distaste.", "confidence": 0.55, "subject": "A",
            "at": 12.4, "at_clock": "00:12.4", "evidence": ["E0001"],
        }],
        "running_summary": "Mostly relaxed; one aversive flash.",
        "ai_usage": {"calls": 4, "input_tokens": 9000, "output_tokens": 800, "cost_usd": 0.012},
    }


def test_scene_gate_skips_static_scene_and_backs_off() -> None:
    gate = SceneChangeGate(4.0, 60.0, 6.0)
    thumb = np.full((18, 32), 100, dtype=np.uint8)
    assert gate.decide(0.0, thumb, ()) == "first"
    gate.commit(0.0, thumb, (), "first")
    assert gate.decide(2.0, thumb, ()) is None
    assert gate.decide(5.0, thumb, ()) is None
    assert gate.decide(8.0, thumb, ()) == "refresh"
    gate.commit(8.0, thumb, (), "refresh")
    assert gate.interval == 16.0
    assert gate.decide(20.0, thumb, ()) is None
    brighter = np.full((18, 32), 120, dtype=np.uint8)
    assert gate.decide(10.0, brighter, ()) is None
    assert gate.decide(12.5, brighter, ()) == "changed"
    assert gate.decide(12.5, thumb, ("laptop",)) == "changed"
    assert gate.decide(12.5, thumb, (), force=True) == "changed"
    for step in range(6):
        gate.commit(30.0 + step * 100, thumb, (), "refresh")
    assert gate.interval == 60.0


def test_scene_signature_ignores_low_confidence() -> None:
    tracks = [{"label": "Person", "confidence": 0.9}, {"label": "cup", "confidence": 0.3}]
    assert scene_signature(tracks) == ("person",)


def test_insight_engine_interprets_notable_episode() -> None:
    reply = {
        "insights": [{"title": "Aversive flash", "observation": "AU9+AU10 for 260 ms",
                      "interpretation": "May reflect distaste.", "category": "affect",
                      "confidence": 70, "subject": "A", "at": 12.4, "evidence": ["E0001"]}],
        "running_summary": "One aversive flash so far.",
    }
    ai = FakeAI(lambda _: reply)
    engine = InsightEngine(ONLINE, ai, lambda: {"subjects": []})
    engine.start("interview")
    engine.set_active(True)
    engine._last_call = 0.0
    engine.push_episodes([_episode("E0001", 12.4)])
    deadline = time.time() + 3.0
    while time.time() < deadline and not engine.status()["insights"]:
        time.sleep(0.05)
    engine.stop()

    insights = engine.all_insights()
    assert insights["running_summary"] == "One aversive flash so far."
    assert insights["insights"][0]["confidence"] == 0.7
    assert insights["insights"][0]["at_clock"] == "00:12.4"
    call = ai.calls[0]
    assert call["feature"] == "insights" and call["json_mode"] is True
    prompt = call["contents"][-1]
    assert "E0001" in prompt and "nose_wrinkle" in prompt


def test_insight_engine_waits_for_interval_without_notable_events() -> None:
    ai = FakeAI(lambda _: {"insights": []})
    engine = InsightEngine(ONLINE, ai, lambda: {"subjects": []})
    engine.start("general")
    engine.set_active(True)
    engine.push_episodes([_episode("E0001", 3.0, kind="behavior", cues=["head_nod"])])
    time.sleep(1.3)
    engine.stop()
    assert ai.calls == []


def test_render_analysis_markdown_is_vendor_neutral() -> None:
    markdown = render_analysis_markdown(_analysis())
    assert markdown.startswith(ANALYSIS_HEADING)
    assert "| A |" in markdown and "AU9+AU10" in markdown
    assert "Brief aversion at question 3" in markdown
    assert "not evidence of deception" in markdown
    assert "gemini" not in markdown.lower() and "google" not in markdown.lower()
    assert render_analysis_markdown({"expressions": {"subjects": []}, "insights": []}) == ""


def test_evidence_payload_is_compact() -> None:
    payload = evidence_payload(_analysis(), {"events": [1, 2, 3]})
    assert len(payload["affect_timeline"]["A"]) == 60
    assert payload["log"]["event_count"] == 3
    assert "au_label" not in payload["episodes"][0]


def test_apply_fixes_only_replaces_exact_matches() -> None:
    text = "The subject was clearly lying about the date."
    fixed, applied = apply_fixes(
        text,
        [{"find": "was clearly lying about", "replace": "showed incongruent signals around"},
         {"find": "not present in report", "replace": "x"}],
    )
    assert applied == 1 and "incongruent" in fixed


def test_report_pipeline_analyze_compose_audit() -> None:
    def responder(call: dict[str, Any]):
        prompt = call["contents"]
        json_mode = call.get("json_mode", False)
        if json_mode and "Audit this report" in prompt:
            return {"fixes": [{"find": "certainly disgusted",
                               "replace": "showed a disgust-like flash"}]}
        if json_mode:
            return {"executive_summary": "One aversive flash.", "findings": [
                {"title": "Aversive flash", "detail": "E0001", "evidence": ["E0001"]}]}
        return "# Session brief\n\nThe subject was certainly disgusted at 00:12."

    ai = FakeAI(responder)
    events = list(enrich_session_report_events("# Draft", {"id": "S1"}, ONLINE, ai, _analysis()))
    assert events[0]["steps"] == [
        "Analyzing behavioral evidence", "Composing intelligence brief",
        "Auditing claims against evidence",
    ]
    done = events[-1]
    assert done["phase"] == "done" and done["enriched"] is True
    assert done["passes_completed"] == 3
    assert "disgust-like flash" in done["markdown"]
    compose = ai.calls[1]
    assert compose["model"] == "fake-perception"
    assert ANALYSIS_HEADING in compose["contents"] and "Analyst findings" in compose["contents"]
    assert all(call["essential"] for call in ai.calls)


def test_report_pipeline_falls_back_to_measured_draft_on_failure() -> None:
    ai = FakeAI(lambda _: AIUnavailable("YUBI v3.0 paused — API credits are depleted."))
    events = list(enrich_session_report_events("# Draft", {}, ONLINE, ai, _analysis()))
    assert any(e["phase"] == "error" for e in events)
    done = events[-1]
    assert done["enriched"] is False
    assert done["markdown"].startswith("# Draft") and ANALYSIS_HEADING in done["markdown"]
