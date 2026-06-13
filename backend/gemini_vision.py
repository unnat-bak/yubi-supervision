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


def normalize_confidence(raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.75
    if value > 1.0:
        value /= 100.0
    return max(0.0, min(1.0, value))


def normalize_box_2d(raw: list[Any]) -> list[int] | None:
    """Parse Gemini box_2d as [ymin, xmin, ymax, xmax] in 0–1000 space."""
    if not raw or len(raw) != 4:
        return None
    try:
        vals = [float(v) for v in raw]
    except (TypeError, ValueError):
        return None

    if max(vals) <= 1.0:
        vals = [v * 1000.0 for v in vals]

    ymin, xmin, ymax, xmax = vals[0], vals[1], vals[2], vals[3]
    ymin, ymax = min(ymin, ymax), max(ymin, ymax)
    xmin, xmax = min(xmin, xmax), max(xmin, xmax)

    ymin_i = int(max(0, min(1000, ymin)))
    xmin_i = int(max(0, min(1000, xmin)))
    ymax_i = int(max(0, min(1000, ymax)))
    xmax_i = int(max(0, min(1000, xmax)))

    if ymax_i - ymin_i < 3 or xmax_i - xmin_i < 3:
        return None
    return [ymin_i, xmin_i, ymax_i, xmax_i]


def box_2d_to_xyxy(box_2d: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    ymin, xmin, ymax, xmax = box_2d
    x1 = int(xmin * width / 1000)
    y1 = int(ymin * height / 1000)
    x2 = int(xmax * width / 1000)
    y2 = int(ymax * height / 1000)
    return (
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )


def box_iou(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def labels_match(gemini_label: str, local_label: str) -> bool:
    g = gemini_label.lower().strip()
    y = local_label.lower().strip()
    if not g or not y:
        return False
    if g == y or g in y or y in g:
        return True
    # Common semantic aliases between v3.0 labels and COCO / UI names
    aliases = {
        "person": ("man", "woman", "human", "face", "head"),
        "cell phone": ("phone", "mobile", "smartphone"),
        "laptop": ("computer", "notebook"),
        "tv": ("monitor", "screen", "television"),
        "chair": ("seat", "stool"),
    }
    for canonical, words in aliases.items():
        if canonical in g or canonical in y:
            if any(w in g or w in y for w in words):
                return True
    return False


def insight_is_fresh(insight: GeminiInsight, max_age_sec: float) -> bool:
    if insight.updated_at is None or not insight.objects:
        return False
    return time.time() - insight.updated_at <= max_age_sec


def effective_object_confidence(
    base_confidence: float, insight: GeminiInsight, max_age_sec: float
) -> float:
    """Lower local threshold slightly when v3.0 recently confirmed scene content."""
    if not insight_is_fresh(insight, max_age_sec):
        return base_confidence
    if any(obj.confidence >= 0.5 for obj in insight.objects):
        return max(0.15, base_confidence - 0.08)
    return base_confidence


def merge_insight_into_tracks(
    tracks: list[dict[str, Any]],
    detections: Any,
    insight: GeminiInsight,
    width: int,
    height: int,
    max_age_sec: float,
) -> list[dict[str, Any]]:
    """Add v3.0-only objects to the track list when local YOLO missed them."""
    if not insight_is_fresh(insight, max_age_sec):
        return tracks

    yolo_boxes: list[tuple[int, int, int, int]] = []
    if not detections.is_empty() and detections.xyxy is not None:
        for xyxy in detections.xyxy:
            yolo_boxes.append(tuple(int(v) for v in xyxy))

    merged = list(tracks)
    seen_labels: set[str] = {str(t.get("label", "")).lower() for t in tracks}

    for obj in insight.objects:
        if obj.confidence < 0.45:
            continue
        box = box_2d_to_xyxy(obj.box_2d, width, height)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        if any(box_iou(box, yb) > 0.2 for yb in yolo_boxes):
            continue
        label_key = obj.label.lower()
        if label_key in seen_labels:
            continue
        merged.append(
            {
                "label": obj.label,
                "confidence": round(obj.confidence, 2),
                "tracker_id": None,
            }
        )
        seen_labels.add(label_key)

    merged.sort(key=lambda item: float(item["confidence"]), reverse=True)
    return merged


def analysis_width_for_frame(frame_width: int, settings: Settings) -> int:
    width = max(320, int(frame_width * settings.gemini_analysis_scale))
    if settings.gemini_analysis_max_width > 0:
        width = min(width, settings.gemini_analysis_max_width)
    return width


class GeminiEnricher:
    """Periodic v3.0 scene analysis — first frame ASAP, then on an interval."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._running = False
        self._analysis_in_flight = False
        self._latest_jpeg: bytes | None = None
        self._insight = GeminiInsight(
            state="disabled" if not self.enabled else "idle"
        )

    @property
    def enabled(self) -> bool:
        return bool(self._settings.gemini_api_key) and self._settings.gemini_enabled

    def push_frame(self, jpeg_bytes: bytes) -> None:
        """Store latest clean frame; trigger analysis on first frame or interval."""
        if not self.enabled:
            return
        with self._lock:
            if not self._running:
                return
            self._latest_jpeg = jpeg_bytes
            insight = self._insight
            in_flight = self._analysis_in_flight
        if in_flight:
            return

        now = time.time()
        is_first = insight.updated_at is None
        interval_ok = (
            insight.updated_at is not None
            and now - insight.updated_at >= self._settings.gemini_interval_sec
        )
        if is_first or interval_ok:
            self._schedule_analysis()

    def start(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._running:
                return
            self._running = True
            self._insight = GeminiInsight(state="idle")

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._latest_jpeg = None
            if not self.enabled:
                self._insight = GeminiInsight(state="disabled")
            else:
                self._insight = GeminiInsight(
                    state="idle",
                    scene_summary=self._insight.scene_summary,
                    objects=list(self._insight.objects),
                    updated_at=self._insight.updated_at,
                )

    def get_insight(self) -> GeminiInsight:
        with self._lock:
            state = self._insight.state
            if self._analysis_in_flight and state != "error":
                state = "thinking"
            return GeminiInsight(
                state=state,
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

    def _schedule_analysis(self) -> None:
        with self._lock:
            if not self._running or not self._latest_jpeg or self._analysis_in_flight:
                return
            frame = self._latest_jpeg
            self._analysis_in_flight = True

        threading.Thread(
            target=self._run_analysis,
            args=(frame,),
            daemon=True,
        ).start()

    def _run_analysis(self, jpeg_bytes: bytes) -> None:
        try:
            insight = self._analyze(jpeg_bytes)
            self._set_insight(
                state="ready",
                scene_summary=insight.scene_summary,
                objects=insight.objects,
                error=None,
                updated_at=time.time(),
            )
        except Exception as exc:
            self._set_insight(state="error", error=str(exc))
        finally:
            self._analysis_in_flight = False

    def _analyze(self, jpeg_bytes: bytes) -> GeminiInsight:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._settings.gemini_api_key)
        prompt = (
            "Analyze this live webcam frame for a real-time vision system. Return JSON:\n"
            '- "scene_summary": one sentence about the scene and activity\n'
            '- "objects": array of visible items. Each item:\n'
            '  - "label": short name matching common detection classes when possible '
            '(person, chair, laptop, cell phone, cup, book, etc.)\n'
            '  - "confidence": 0.0-1.0\n'
            '  - "box_2d": [ymin, xmin, ymax, xmax] normalized 0-1000\n'
            "Tight boxes on visible object edges. ymin/xmin = top-left, ymax/xmax = bottom-right. "
            f"Limit to {self._settings.gemini_max_objects} objects."
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

        raw = response.text or "{}"
        payload = json.loads(raw)
        objects: list[GeminiObject] = []
        for item in payload.get("objects", []):
            box = item.get("box_2d") or item.get("bounding_box")
            if not box:
                continue
            normalized = normalize_box_2d(box)
            if normalized is None:
                continue
            objects.append(
                GeminiObject(
                    label=str(item.get("label", "object")),
                    confidence=normalize_confidence(item.get("confidence", 0.75)),
                    box_2d=normalized,
                )
            )

        return GeminiInsight(
            state="ready",
            scene_summary=str(payload.get("scene_summary", "")).strip(),
            objects=objects,
            updated_at=time.time(),
        )
