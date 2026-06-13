from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from backend.config import Settings


@dataclass
class GeminiObject:
    label: str
    confidence: float
    box_2d: list[int]


@dataclass
class GeminiInsight:
    state: str = "disabled"
    scene_summary: str = ""
    objects: list[GeminiObject] = field(default_factory=list)
    error: str | None = None
    updated_at: float | None = None


class GeminiEnricher:
    """Periodic Gemini Vision analysis — runs off the realtime inference loop."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._latest_jpeg: bytes | None = None
        self._insight = GeminiInsight(
            state="disabled" if not self.enabled else "idle"
        )

    @property
    def enabled(self) -> bool:
        return bool(self._settings.gemini_api_key) and self._settings.gemini_enabled

    def push_frame(self, jpeg_bytes: bytes) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._latest_jpeg = jpeg_bytes

    def start(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._running:
                return
            self._running = True
            self._insight = GeminiInsight(state="idle")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        with self._lock:
            self._latest_jpeg = None
            self._insight = GeminiInsight(
                state="disabled" if not self.enabled else "idle"
            )

    def get_insight(self) -> GeminiInsight:
        with self._lock:
            return GeminiInsight(
                state=self._insight.state,
                scene_summary=self._insight.scene_summary,
                objects=list(self._insight.objects),
                error=self._insight.error,
                updated_at=self._insight.updated_at,
            )

    def _set_insight(self, **kwargs: Any) -> None:
        with self._lock:
            self._insight = GeminiInsight(
                state=kwargs.get("state", self._insight.state),
                scene_summary=kwargs.get("scene_summary", self._insight.scene_summary),
                objects=kwargs.get("objects", self._insight.objects),
                error=kwargs.get("error", self._insight.error),
                updated_at=kwargs.get("updated_at", self._insight.updated_at),
            )

    def _loop(self) -> None:
        while True:
            with self._lock:
                if not self._running:
                    break
            time.sleep(self._settings.gemini_interval_sec)

            with self._lock:
                if not self._running:
                    break
                frame = self._latest_jpeg
            if not frame:
                continue

            self._set_insight(state="thinking", error=None)
            try:
                insight = self._analyze(frame)
                self._set_insight(
                    state="ready",
                    scene_summary=insight.scene_summary,
                    objects=insight.objects,
                    error=None,
                    updated_at=time.time(),
                )
            except Exception as exc:
                self._set_insight(state="error", error=str(exc))

    def _analyze(self, jpeg_bytes: bytes) -> GeminiInsight:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._settings.gemini_api_key)
        prompt = (
            "Analyze this live webcam frame. Return JSON with:\n"
            '- "scene_summary": one vivid sentence about the scene, person, and activity\n'
            '- "objects": array of prominent items. Each item needs:\n'
            '  - "label": specific name (e.g. "person waving", "wooden chair", "straw hat")\n'
            '  - "confidence": 0.0-1.0\n'
            '  - "box_2d": [ymin, xmin, ymax, xmax] normalized to 0-1000\n'
            f"Limit to {self._settings.gemini_max_objects} objects. "
            "Include the person's face/head region as an object when visible."
        )

        response = client.models.generate_content(
            model=self._settings.gemini_model,
            contents=[
                types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )

        raw = response.text or "{}"
        payload = json.loads(raw)
        objects: list[GeminiObject] = []
        for item in payload.get("objects", []):
            box = item.get("box_2d") or item.get("bounding_box")
            if not box or len(box) != 4:
                continue
            objects.append(
                GeminiObject(
                    label=str(item.get("label", "object")),
                    confidence=float(item.get("confidence", 0.75)),
                    box_2d=[int(v) for v in box],
                )
            )

        return GeminiInsight(
            state="ready",
            scene_summary=str(payload.get("scene_summary", "")).strip(),
            objects=objects,
            updated_at=time.time(),
        )
