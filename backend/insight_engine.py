"""Event-driven YUBI v3.0 interpretation of locally detected behavioral signals.

The micro-expression engine measures everything on-device. This engine only
calls the reasoning tier when there is something worth interpreting — a
micro-expression, masking / leakage cue, gaze or blink anomaly — and sends a
compact text digest plus at most a couple of low-resolution apex face crops. A
running summary is carried between calls, so context accumulates without
resending history and cost stays flat over long sessions.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import cv2
from google.genai import types

from backend.ai_client import AIClient, AIUnavailable
from backend.config import Settings
from backend.micro_expressions import Episode

INSIGHT_CATEGORIES = (
    "affect",
    "incongruence",
    "stress",
    "engagement",
    "attention",
    "rapport",
    "other",
)

SYSTEM_INSTRUCTION = (
    "You are YUBI v3.0, the behavioral-intelligence layer of YUBI Supervision. You receive "
    "measurements from an on-device facial-action and behavior tracker — FACS action units, "
    "micro-expression episodes with millisecond timing, blinks, gaze, head gestures, "
    "hand-to-face contact, an experimental camera pulse estimate — and occasionally small face "
    "crops captured at an expression's apex.\n"
    "Rules:\n"
    "- Interpret observable behavior only. Never claim to know thoughts, intent, truthfulness, "
    "guilt, or any diagnosis, and never say someone is lying. Use calibrated language: "
    '"consistent with", "may reflect", "worth probing".\n'
    "- Ground every insight in evidence: episode ids, timestamps, AU codes, and changes "
    "relative to the subject's own baseline (baseline deviations matter more than absolutes).\n"
    "- Micro-expressions (<500 ms) are often involuntary; treat incongruence (a smile with "
    "negative leakage, suppression) as a question to explore, not a conclusion.\n"
    "- Speech articulation and head motion cause false positives: discount mouth-region "
    "episodes marked speaking=true.\n"
    "- If a face crop contradicts the tracker, say so and lower confidence.\n"
    "- Prefer no insight over a weak one. Be concise.\n"
    "Refer to yourself only as YUBI v3.0 and never mention model vendors."
)

_RESPONSE_SHAPE = (
    'Return JSON: {"insights": [{"title": str (<= 60 chars), "observation": str (what was '
    'measured), "interpretation": str (what it may reflect, hedged), "category": one of '
    + json.dumps(list(INSIGHT_CATEGORIES))
    + ', "confidence": 0.0-1.0, "subject": str, "at": seconds (media time), '
    '"evidence": [episode ids]}], "running_summary": str (<= 600 chars, cumulative read of '
    "the session so far)}. At most 3 insights; return an empty list when nothing new is "
    "meaningful."
)

_MAX_PENDING = 60
_MAX_INSIGHTS = 120
_POLL_SEC = 1.0


def format_clock(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours >= 1:
        return f"{int(hours)}:{int(minutes):02d}:{secs:04.1f}"
    return f"{int(minutes):02d}:{secs:04.1f}"


def _compact_episode(episode: Episode) -> dict[str, Any]:
    data = episode.as_dict()
    data["at_clock"] = format_clock(episode.at)
    for key in ("au_label", "cue_labels"):
        data.pop(key, None)
    if not data["speaking"]:
        data.pop("speaking")
    return data


def _compact_subject(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": summary.get("id"),
        "observed_sec": summary.get("observed_sec"),
        "episodes": summary.get("episodes"),
        "top_cues": dict(list((summary.get("cues") or {}).items())[:6]),
        "blink_rate_per_min": (summary.get("blink") or {}).get("rate_per_min"),
        "gaze_away_pct": summary.get("gaze_away_pct"),
        "head_nods": summary.get("head_nods"),
        "head_shakes": summary.get("head_shakes"),
        "self_touch": summary.get("self_touch"),
        "pulse_bpm": summary.get("pulse"),
        "valence_mean": (summary.get("affect") or {}).get("valence_mean"),
    }


def _clean_insight(raw: Any, seq: int, fallback_at: float) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    title = str(raw.get("title", "")).strip()[:80]
    observation = str(raw.get("observation", "")).strip()
    if not title or not observation:
        return None
    category = str(raw.get("category", "other")).strip().lower()
    try:
        confidence = float(raw.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    if confidence > 1.0:
        confidence /= 100.0
    try:
        at = float(raw.get("at", fallback_at))
    except (TypeError, ValueError):
        at = fallback_at
    evidence = raw.get("evidence") or []
    return {
        "id": f"I{seq:03d}",
        "title": title,
        "observation": observation,
        "interpretation": str(raw.get("interpretation", "")).strip(),
        "category": category if category in INSIGHT_CATEGORIES else "other",
        "confidence": round(max(0.0, min(1.0, confidence)), 2),
        "subject": str(raw.get("subject", "")).strip()[:8],
        "at": round(max(0.0, at), 2),
        "at_clock": format_clock(at),
        "evidence": [str(e) for e in evidence][:8] if isinstance(evidence, list) else [],
        "created_at": time.time(),
    }


class InsightEngine:
    def __init__(
        self,
        settings: Settings,
        ai: AIClient,
        summary_provider: Callable[[], dict[str, Any]],
    ) -> None:
        self._settings = settings
        self._ai = ai
        self._summary_provider = summary_provider
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._active = False
        self._profile = "general"
        self._scene = ""
        self._pending: deque[Episode] = deque(maxlen=_MAX_PENDING)
        self._insights: list[dict[str, Any]] = []
        self._running_summary = ""
        self._seq = 0
        self._last_call = 0.0
        self._in_flight = False
        self._error: str | None = None
        self._last_media_sec = 0.0

    @property
    def enabled(self) -> bool:
        return self._ai.available and self._settings.insights_enabled

    def start(self, profile: str) -> None:
        with self._lock:
            self._profile = profile
            self._scene = ""
            self._pending.clear()
            self._insights = []
            self._running_summary = ""
            self._seq = 0
            self._last_call = time.time()
            self._error = None
            self._running = True
        if not self.enabled:
            return
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._worker, name="yubi-insights", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._active = False
            self._pending.clear()
        self._wake.set()

    def set_active(self, active: bool) -> None:
        with self._lock:
            self._active = active
            if not active:
                self._pending.clear()

    @property
    def wants_crops(self) -> bool:
        with self._lock:
            return (
                self._active
                and self._running
                and self.enabled
                and self._settings.insight_max_images > 0
            )

    def note_scene(self, summary: str) -> None:
        if summary:
            with self._lock:
                self._scene = summary

    def push_episodes(self, episodes: list[Episode]) -> None:
        if not episodes:
            return
        with self._lock:
            if not (self._running and self._active):
                return
            for episode in episodes:
                self._pending.append(episode)
                self._last_media_sec = max(self._last_media_sec, episode.at)
        if any(e.notable for e in episodes):
            self._wake.set()

    def _due(self, now: float) -> bool:
        if self._in_flight or not self._pending:
            return False
        elapsed = now - self._last_call
        notable = any(e.notable for e in self._pending)
        if notable and elapsed >= self._settings.insight_min_interval_sec:
            return True
        return elapsed >= self._settings.insight_digest_interval_sec

    def _worker(self) -> None:
        while True:
            self._wake.wait(timeout=_POLL_SEC)
            self._wake.clear()
            with self._lock:
                if not self._running:
                    return
                now = time.time()
                if not self._active or not self._due(now):
                    continue
                blocked = self._ai.budget.blocked_reason("insights", self._ai.insight_model)
                if blocked:
                    self._error = blocked
                    continue
                batch = list(self._pending)
                self._pending.clear()
                self._in_flight = True
                self._last_call = now
                profile = self._profile
                scene = self._scene
                previous = self._running_summary
            try:
                self._run(batch, profile, scene, previous)
            finally:
                with self._lock:
                    self._in_flight = False

    def _run(self, batch: list[Episode], profile: str, scene: str, previous: str) -> None:
        images = sorted(
            (e for e in batch if e.apex_crop is not None and e.kind != "behavior"),
            key=lambda e: (e.notable, e.intensity),
            reverse=True,
        )[: self._settings.insight_max_images]
        parts: list[Any] = []
        attached: list[str] = []
        for episode in images:
            ok, jpeg = cv2.imencode(".jpg", episode.apex_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                parts.append(types.Part.from_bytes(data=jpeg.tobytes(), mime_type="image/jpeg"))
                attached.append(episode.id)

        summary = self._summary_provider()
        digest = {
            "analysis_profile": profile,
            "scene_context": scene or None,
            "previous_summary": previous or None,
            "subjects": [_compact_subject(s) for s in summary.get("subjects", [])],
            "new_events": [_compact_episode(e) for e in batch[-40:]],
            "attached_crops": [
                {"image_index": index, "episode": episode_id}
                for index, episode_id in enumerate(attached)
            ],
        }
        prompt = (
            "New behavioral measurements since the last read. Interpret what is meaningful.\n"
            + _RESPONSE_SHAPE
            + "\n\n"
            + json.dumps(digest, separators=(",", ":"))
        )
        parts.append(prompt)
        try:
            result = self._ai.generate(
                feature="insights",
                model=self._ai.insight_model,
                contents=parts,
                json_mode=True,
                temperature=0.2,
                thinking=self._settings.insight_thinking,
                media_resolution="low",
                system_instruction=SYSTEM_INSTRUCTION,
                max_output_tokens=4096,
            )
        except AIUnavailable as exc:
            with self._lock:
                self._error = str(exc)
                for episode in batch[-10:]:
                    if episode.notable:
                        self._pending.appendleft(episode)
            return

        payload = result.data if isinstance(result.data, dict) else {}
        fallback_at = batch[-1].at if batch else 0.0
        with self._lock:
            for raw in (payload.get("insights") or [])[:3]:
                self._seq += 1
                insight = _clean_insight(raw, self._seq, fallback_at)
                if insight is not None:
                    self._insights.append(insight)
            if len(self._insights) > _MAX_INSIGHTS:
                del self._insights[: len(self._insights) - _MAX_INSIGHTS]
            summary_text = str(payload.get("running_summary", "")).strip()
            if summary_text:
                self._running_summary = summary_text[:900]
            self._error = None

    def status(self) -> dict[str, Any]:
        with self._lock:
            if not self.enabled:
                state = "off"
            elif not (self._running and self._active):
                state = "idle"
            elif self._in_flight:
                state = "thinking"
            elif self._error:
                state = "paused"
            else:
                state = "watching"
            next_in = None
            if self._running and self._active and not self._in_flight:
                wait = self._settings.insight_min_interval_sec - (time.time() - self._last_call)
                next_in = max(0, int(wait))
            return {
                "enabled": self.enabled,
                "state": state,
                "insights": list(reversed(self._insights[-8:])),
                "running_summary": self._running_summary,
                "pending": len(self._pending),
                "error": self._error,
                "next_eligible_in": next_in,
            }

    def all_insights(self) -> dict[str, Any]:
        with self._lock:
            return {
                "insights": list(self._insights),
                "running_summary": self._running_summary,
            }
