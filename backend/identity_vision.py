"""Per-track identity enrichment (YUBI v3.0 vision).

Periodically sends crops of tracked people to the YUBI v3.0 model and asks for
a concise identity — a jersey number and name for sports footage when legible,
otherwise a short visual descriptor. Results are cached on the ByteTrack
``tracker_id`` so the same person keeps a stable label without re-querying.

Structured like ``GeminiEnricher``: a single in-flight background call, gated by
an interval, never blocking the capture/inference loop.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any

from backend.config import Settings

PERSON_LABELS = ("person", "man", "woman", "player", "human", "people")


@dataclass
class IdentityEntry:
    identity: str
    confidence: float
    updated_at: float


def is_person_label(label: str) -> bool:
    lowered = label.lower()
    return any(token in lowered for token in PERSON_LABELS)


def collect_person_hints(
    tracks: list[dict[str, Any]],
    detections: Any,
    width: int,
    height: int,
    max_persons: int,
) -> list[dict[str, Any]]:
    """Build VLM hints ([{tracker_id, box_2d}]) for tracked people."""
    if detections.is_empty() or detections.xyxy is None or detections.tracker_id is None:
        return []

    person_ids = {
        int(t["tracker_id"])
        for t in tracks
        if t.get("tracker_id") is not None and is_person_label(str(t.get("label", "")))
    }
    if not person_ids:
        return []

    hints: list[dict[str, Any]] = []
    tracker_ids = detections.tracker_id
    for index in range(len(detections)):
        raw_tid = tracker_ids[index]
        if raw_tid is None:
            continue
        tid = int(raw_tid)
        if tid not in person_ids:
            continue
        x1, y1, x2, y2 = (int(v) for v in detections.xyxy[index])
        if x2 <= x1 or y2 <= y1:
            continue
        hints.append(
            {
                "tracker_id": tid,
                "box_2d": [
                    int(y1 * 1000 / height),
                    int(x1 * 1000 / width),
                    int(y2 * 1000 / height),
                    int(x2 * 1000 / width),
                ],
            }
        )
    hints.sort(key=lambda h: h["tracker_id"])
    return hints[:max_persons]


class IdentityCache:
    """Sticky identity per ByteTrack id with TTL eviction."""

    def __init__(self, max_age_sec: float) -> None:
        self._max_age_sec = max_age_sec
        self._lock = threading.Lock()
        self._entries: dict[int, IdentityEntry] = {}

    def reset(self) -> None:
        with self._lock:
            self._entries.clear()

    def get(self, tracker_id: int) -> IdentityEntry | None:
        with self._lock:
            entry = self._entries.get(tracker_id)
            if entry is None:
                return None
            if time.time() - entry.updated_at > self._max_age_sec:
                del self._entries[tracker_id]
                return None
            return entry

    def set(self, tracker_id: int, identity: str, confidence: float) -> None:
        with self._lock:
            self._entries[tracker_id] = IdentityEntry(
                identity=identity,
                confidence=round(confidence, 2),
                updated_at=time.time(),
            )

    def fresh_ids(self) -> set[int]:
        now = time.time()
        with self._lock:
            return {
                tid
                for tid, entry in self._entries.items()
                if now - entry.updated_at <= self._max_age_sec
            }

    def prune(self, active_ids: set[int]) -> None:
        with self._lock:
            for tid in list(self._entries):
                if tid not in active_ids:
                    del self._entries[tid]


def apply_identity_labels(
    detections: Any,
    labels: list[str],
    cache: IdentityCache,
) -> list[str]:
    """Prefix overlay labels with cached identities.

    ``detections`` and ``labels`` must be index-aligned (the set actually being
    drawn). Labels for tracker ids with a fresh cached identity get the identity
    string prepended.
    """
    if not labels or detections.is_empty() or detections.tracker_id is None:
        return labels
    out = list(labels)
    tracker_ids = detections.tracker_id
    for position in range(min(len(out), len(tracker_ids))):
        raw_tid = tracker_ids[position]
        if raw_tid is None:
            continue
        entry = cache.get(int(raw_tid))
        if entry is None or not entry.identity:
            continue
        out[position] = f"{entry.identity} · {out[position]}"
    return out


class IdentityEnricher:
    """Periodic identity resolution for tracked people."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._running = False
        self._in_flight = False
        self._latest_jpeg: bytes | None = None
        self._pending_hints: list[dict[str, Any]] = []
        self._last_run = 0.0
        self.cache = IdentityCache(settings.identity_label_cache_sec)
        self._error: str | None = None

    @property
    def enabled(self) -> bool:
        return (
            bool(self._settings.gemini_api_key)
            and self._settings.gemini_enabled
            and self._settings.identity_enabled
        )

    def start(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._running = True
            self._last_run = 0.0
        self.cache.reset()

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._latest_jpeg = None
            self._pending_hints = []
        self.cache.reset()

    def wants_new_analysis(self, hints: list[dict[str, Any]]) -> bool:
        """True when at least one tracked person lacks a fresh identity and the
        interval has elapsed (so the loop can skip JPEG encode otherwise)."""
        if not self.enabled or not hints:
            return False
        with self._lock:
            if not self._running or self._in_flight:
                return False
            last_run = self._last_run
        if time.time() - last_run < self._settings.identity_interval_sec:
            return False
        fresh = self.cache.fresh_ids()
        return any(h["tracker_id"] not in fresh for h in hints)

    def push_frame(self, jpeg_bytes: bytes, hints: list[dict[str, Any]]) -> None:
        if not self.enabled or not hints:
            return
        with self._lock:
            if not self._running or self._in_flight:
                return
            self._latest_jpeg = jpeg_bytes
            self._pending_hints = hints
            self._in_flight = True
            self._last_run = time.time()
            frame = jpeg_bytes
            pending = list(hints)
        threading.Thread(
            target=self._run_analysis, args=(frame, pending), daemon=True
        ).start()

    def get_error(self) -> str | None:
        with self._lock:
            return self._error

    def _run_analysis(self, jpeg_bytes: bytes, hints: list[dict[str, Any]]) -> None:
        try:
            results = self._analyze(jpeg_bytes, hints)
            for item in results:
                tid = item.get("tracker_id")
                identity = str(item.get("identity", "")).strip()
                if tid is None or not identity:
                    continue
                conf = item.get("confidence", 0.7)
                try:
                    conf_f = float(conf)
                except (TypeError, ValueError):
                    conf_f = 0.7
                if conf_f > 1.0:
                    conf_f /= 100.0
                self.cache.set(int(tid), identity, max(0.0, min(1.0, conf_f)))
            with self._lock:
                self._error = None
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
        finally:
            with self._lock:
                self._in_flight = False

    def _analyze(
        self, jpeg_bytes: bytes, hints: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._settings.gemini_api_key)
        prompt = (
            "This frame contains tracked people, each given as a numbered region "
            "(box_2d is [ymin, xmin, ymax, xmax], normalized 0-1000). For every "
            "region return a concise identity:\n"
            "- If a sports player with a legible jersey, use '#<number> <surname>' "
            "(e.g. '#23 Robinson'); use just '#<number>' if the name is unknown.\n"
            "- Otherwise a short visual descriptor (e.g. 'man in red jacket').\n"
            "Keep each identity under 24 characters. Echo back the tracker_id.\n"
            'Return JSON: {"identities": [{"tracker_id": int, "identity": str, '
            '"confidence": 0.0-1.0}]}\n\nRegions:\n'
            + json.dumps(hints, indent=2)
        )

        response = client.models.generate_content(
            model=self._settings.gemini_model,
            contents=[
                types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )
        payload = json.loads(response.text or "{}")
        identities = payload.get("identities", [])
        return identities if isinstance(identities, list) else []
