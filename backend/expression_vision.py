from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarksConnections as FLC,
)

from backend.config import Settings
from backend.gemini_vision import normalize_box_2d

# MediaPipe face mesh edge groups for expression mode (high-precision grid).
EXPR_MESH_CONNECTIONS: tuple[tuple[int, int], ...] = tuple(
    (conn.start, conn.end)
    for group in (
        FLC.FACE_LANDMARKS_TESSELATION,
        FLC.FACE_LANDMARKS_FACE_OVAL,
        FLC.FACE_LANDMARKS_LEFT_EYEBROW,
        FLC.FACE_LANDMARKS_RIGHT_EYEBROW,
        FLC.FACE_LANDMARKS_LEFT_EYE,
        FLC.FACE_LANDMARKS_RIGHT_EYE,
        FLC.FACE_LANDMARKS_LEFT_IRIS,
        FLC.FACE_LANDMARKS_RIGHT_IRIS,
        FLC.FACE_LANDMARKS_NOSE,
    )
    for conn in group
)

MICRO_BLENDSHAPE_NAMES: frozenset[str] = frozenset(
    {
        "browInnerUp",
        "browOuterUpLeft",
        "browOuterUpRight",
        "browDownLeft",
        "browDownRight",
        "eyeSquintLeft",
        "eyeSquintRight",
        "cheekSquintLeft",
        "cheekSquintRight",
        "eyeWideLeft",
        "eyeWideRight",
    }
)

MICRO_LABELS: dict[str, str] = {
    "browInnerUp": "Brow raise (inner)",
    "browOuterUpLeft": "Brow arch (left)",
    "browOuterUpRight": "Brow arch (right)",
    "browDownLeft": "Brow lower (left)",
    "browDownRight": "Brow lower (right)",
    "eyeSquintLeft": "Under-eye tension (left)",
    "eyeSquintRight": "Under-eye tension (right)",
    "cheekSquintLeft": "Cheek twitch (left)",
    "cheekSquintRight": "Cheek twitch (right)",
    "eyeWideLeft": "Eye widen (left)",
    "eyeWideRight": "Eye widen (right)",
}


@dataclass
class MicroEvent:
    label: str
    region: str
    intensity: float
    ts: float


@dataclass
class ExpressionGuidance:
    state: str = "idle"
    face_box_2d: list[int] | None = None
    left_eyebrow_box: list[int] | None = None
    right_eyebrow_box: list[int] | None = None
    left_under_eye_box: list[int] | None = None
    right_under_eye_box: list[int] | None = None
    micro_cues: list[str] = field(default_factory=list)
    structure_notes: str = ""
    error: str | None = None
    updated_at: float | None = None


@dataclass
class _FaceHistory:
    scores: dict[str, float] = field(default_factory=dict)
    history: list[dict[str, float]] = field(default_factory=list)


class MicroExpressionTracker:
    """Detect subtle blendshape deltas between frames."""

    def __init__(
        self,
        history_len: int = 10,
        micro_threshold: float = 0.035,
        spike_threshold: float = 0.055,
    ) -> None:
        self._history_len = history_len
        self._micro_threshold = micro_threshold
        self._spike_threshold = spike_threshold
        self._faces: dict[int, _FaceHistory] = {}
        self._active: list[MicroEvent] = []

    def reset(self) -> None:
        self._faces.clear()
        self._active.clear()

    def get_active_events(self) -> list[MicroEvent]:
        return list(self._active)

    def update(self, face_index: int, blendshapes: list[Any]) -> list[MicroEvent]:
        scores: dict[str, float] = {}
        for item in blendshapes:
            name = (
                getattr(item, "category_name", None)
                or getattr(item, "display_name", None)
                or ""
            )
            if name not in MICRO_BLENDSHAPE_NAMES:
                continue
            scores[name] = float(getattr(item, "score", 0.0))

        if not scores:
            return []

        bucket = self._faces.setdefault(face_index, _FaceHistory())
        prev = dict(bucket.scores)
        bucket.scores = scores
        bucket.history.append(scores)
        if len(bucket.history) > self._history_len:
            bucket.history.pop(0)

        baseline: dict[str, float] = {}
        for key in scores:
            values = [h.get(key, 0.0) for h in bucket.history]
            baseline[key] = sum(values) / len(values)

        events: list[MicroEvent] = []
        now = time.time()
        for key, value in scores.items():
            base = baseline.get(key, value)
            delta = value - base
            spike = value - prev.get(key, value) if prev else 0.0
            magnitude = max(abs(delta), abs(spike))
            if magnitude < self._micro_threshold and abs(spike) < self._spike_threshold:
                continue
            region = "eyebrow" if "brow" in key else "under-eye"
            events.append(
                MicroEvent(
                    label=MICRO_LABELS.get(key, key),
                    region=region,
                    intensity=round(min(1.0, magnitude * 4.0), 2),
                    ts=now,
                )
            )

        self._active = events
        return events


def landmarks_to_points(
    landmarks: list[Any], width: int, height: int
) -> np.ndarray:
    return np.array(
        [[lm.x * width, lm.y * height] for lm in landmarks],
        dtype=np.float32,
    )


def refine_landmarks_with_guidance(
    points: np.ndarray,
    guidance: ExpressionGuidance,
    width: int,
    height: int,
    max_shift_px: float = 14.0,
) -> np.ndarray:
    if guidance.face_box_2d is None or len(points) == 0:
        return points
    ymin, xmin, ymax, xmax = guidance.face_box_2d
    ai_cx = (xmin + xmax) * width / 2000.0
    ai_cy = (ymin + ymax) * height / 2000.0
    mp_cx = float(points[:, 0].mean())
    mp_cy = float(points[:, 1].mean())
    dx = ai_cx - mp_cx
    dy = ai_cy - mp_cy
    dist = (dx * dx + dy * dy) ** 0.5
    if dist < 3.0:
        return points
    if dist > max_shift_px:
        dx *= max_shift_px / dist
        dy *= max_shift_px / dist
    return points + np.array([dx, dy], dtype=np.float32)


def crop_face_jpeg(
    frame_bgr: np.ndarray,
    landmarks: list[Any],
    scale: float,
    quality: int,
    padding: float = 0.18,
) -> bytes | None:
    height, width = frame_bgr.shape[:2]
    pts = landmarks_to_points(landmarks, width, height)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    ix1 = max(0, int(x1 - pad_x))
    iy1 = max(0, int(y1 - pad_y))
    ix2 = min(width, int(x2 + pad_x))
    iy2 = min(height, int(y2 + pad_y))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    crop = frame_bgr[iy1:iy2, ix1:ix2]
    target_w = max(160, int(crop.shape[1] * scale))
    if crop.shape[1] > target_w:
        scale_factor = target_w / crop.shape[1]
        crop = cv2.resize(
            crop,
            (target_w, int(crop.shape[0] * scale_factor)),
            interpolation=cv2.INTER_AREA,
        )
    ok, jpeg = cv2.imencode(
        ".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    return jpeg.tobytes() if ok else None


def _box_to_xyxy(box: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    ymin, xmin, ymax, xmax = box
    return (
        int(xmin * width / 1000),
        int(ymin * height / 1000),
        int(xmax * width / 1000),
        int(ymax * height / 1000),
    )


def _connection_indices(group: Any) -> set[int]:
    return {conn.start for conn in group} | {conn.end for conn in group}


EXPR_BROW_INDICES: frozenset[int] = frozenset(
    _connection_indices(FLC.FACE_LANDMARKS_LEFT_EYEBROW)
    | _connection_indices(FLC.FACE_LANDMARKS_RIGHT_EYEBROW)
)
EXPR_EYE_INDICES: frozenset[int] = frozenset(
    _connection_indices(FLC.FACE_LANDMARKS_LEFT_EYE)
    | _connection_indices(FLC.FACE_LANDMARKS_RIGHT_EYE)
)
EXPR_IRIS_INDICES: frozenset[int] = frozenset(
    _connection_indices(FLC.FACE_LANDMARKS_LEFT_IRIS)
    | _connection_indices(FLC.FACE_LANDMARKS_RIGHT_IRIS)
)

# BGR — aligned with overlay pearl / signal palette in vision.py
_TRACK_DOT_BASE = (210, 204, 197)
_TRACK_DOT_ACTIVE = (235, 230, 224)
_TRACK_MESH = (132, 144, 158)
_TRACK_BOX = (118, 128, 142)


def _draw_track_dots(
    canvas: np.ndarray,
    pts_i: np.ndarray,
    indices: frozenset[int],
    radius: int,
    color: tuple[int, int, int],
) -> None:
    for idx in indices:
        if idx >= len(pts_i):
            continue
        cv2.circle(canvas, tuple(pts_i[idx]), radius, color, -1, cv2.LINE_AA)


def draw_expression_overlay(
    scene: np.ndarray,
    face_landmarks: list[list[Any]],
    guidance: ExpressionGuidance,
    micro_events: list[MicroEvent],
    mesh_alpha: float = 0.24,
) -> np.ndarray:
    if not face_landmarks:
        return scene

    height, width = scene.shape[:2]
    mesh_layer = scene.copy()
    highlight_regions: set[str] = {e.region for e in micro_events}
    brow_active = "eyebrow" in highlight_regions
    eye_active = "under-eye" in highlight_regions

    for landmarks in face_landmarks:
        points = landmarks_to_points(landmarks, width, height)
        points = refine_landmarks_with_guidance(points, guidance, width, height)
        pts_i = points.astype(np.int32)

        for start, end in EXPR_MESH_CONNECTIONS:
            if start >= len(pts_i) or end >= len(pts_i):
                continue
            cv2.line(
                mesh_layer,
                tuple(pts_i[start]),
                tuple(pts_i[end]),
                _TRACK_MESH,
                1,
                cv2.LINE_AA,
            )

        for box in (
            guidance.left_eyebrow_box,
            guidance.right_eyebrow_box,
            guidance.left_under_eye_box,
            guidance.right_under_eye_box,
        ):
            if not box:
                continue
            x1, y1, x2, y2 = _box_to_xyxy(box, width, height)
            cv2.rectangle(mesh_layer, (x1, y1), (x2, y2), _TRACK_BOX, 1, cv2.LINE_AA)

    result = cv2.addWeighted(mesh_layer, mesh_alpha, scene, 1.0 - mesh_alpha, 0)

    for landmarks in face_landmarks:
        points = landmarks_to_points(landmarks, width, height)
        points = refine_landmarks_with_guidance(points, guidance, width, height)
        pts_i = points.astype(np.int32)

        _draw_track_dots(
            result,
            pts_i,
            EXPR_BROW_INDICES,
            2 if brow_active else 1,
            _TRACK_DOT_ACTIVE if brow_active else _TRACK_DOT_BASE,
        )
        _draw_track_dots(
            result,
            pts_i,
            EXPR_EYE_INDICES,
            2 if eye_active else 1,
            _TRACK_DOT_ACTIVE if eye_active else _TRACK_DOT_BASE,
        )
        _draw_track_dots(
            result,
            pts_i,
            EXPR_IRIS_INDICES,
            1,
            _TRACK_DOT_ACTIVE if eye_active else _TRACK_DOT_BASE,
        )

    return result


class ExpressionEnricher:
    """Periodic v3.0 face-structure analysis to stabilize local landmark drawing."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._active = False
        self._analysis_in_flight = False
        self._latest_jpeg: bytes | None = None
        self._guidance = ExpressionGuidance(
            state="disabled" if not self._api_enabled else "idle"
        )

    @property
    def _api_enabled(self) -> bool:
        return bool(self._settings.gemini_api_key) and self._settings.gemini_enabled

    def set_active(self, active: bool) -> None:
        with self._lock:
            self._active = active and self._api_enabled
            if not self._active:
                self._latest_jpeg = None

    def push_face_frame(self, jpeg_bytes: bytes) -> None:
        if not self._active or not jpeg_bytes:
            return
        with self._lock:
            self._latest_jpeg = jpeg_bytes
            guidance = self._guidance
            in_flight = self._analysis_in_flight
        if in_flight:
            return
        now = time.time()
        is_first = guidance.updated_at is None
        interval_ok = (
            guidance.updated_at is not None
            and now - guidance.updated_at >= self._settings.gemini_interval_sec
        )
        if is_first or interval_ok:
            self._schedule_analysis()

    def get_guidance(self) -> ExpressionGuidance:
        with self._lock:
            state = self._guidance.state
            if self._analysis_in_flight and state != "error":
                state = "thinking"
            return ExpressionGuidance(
                state=state,
                face_box_2d=self._guidance.face_box_2d,
                left_eyebrow_box=self._guidance.left_eyebrow_box,
                right_eyebrow_box=self._guidance.right_eyebrow_box,
                left_under_eye_box=self._guidance.left_under_eye_box,
                right_under_eye_box=self._guidance.right_under_eye_box,
                micro_cues=list(self._guidance.micro_cues),
                structure_notes=self._guidance.structure_notes,
                error=self._guidance.error,
                updated_at=self._guidance.updated_at,
            )

    def _schedule_analysis(self) -> None:
        with self._lock:
            if not self._active or not self._latest_jpeg or self._analysis_in_flight:
                return
            frame = self._latest_jpeg
            self._analysis_in_flight = True
        threading.Thread(target=self._run_analysis, args=(frame,), daemon=True).start()

    def _run_analysis(self, jpeg_bytes: bytes) -> None:
        try:
            guidance = self._analyze(jpeg_bytes)
            with self._lock:
                self._guidance = guidance
        except Exception as exc:
            with self._lock:
                self._guidance = ExpressionGuidance(
                    state="error",
                    error=str(exc),
                    updated_at=self._guidance.updated_at,
                    face_box_2d=self._guidance.face_box_2d,
                    left_eyebrow_box=self._guidance.left_eyebrow_box,
                    right_eyebrow_box=self._guidance.right_eyebrow_box,
                    left_under_eye_box=self._guidance.left_under_eye_box,
                    right_under_eye_box=self._guidance.right_under_eye_box,
                    micro_cues=list(self._guidance.micro_cues),
                    structure_notes=self._guidance.structure_notes,
                )
        finally:
            self._analysis_in_flight = False

    def _analyze(self, jpeg_bytes: bytes) -> ExpressionGuidance:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._settings.gemini_api_key)
        prompt = (
            "This is a cropped face for micro-expression tracking. Return JSON:\n"
            '- "face_box_2d": [ymin, xmin, ymax, xmax] normalized 0-1000 for full face\n'
            '- "left_eyebrow_box": tight box for left eyebrow region\n'
            '- "right_eyebrow_box": tight box for right eyebrow region\n'
            '- "left_under_eye_box": tight box under left eye (orbital muscle area)\n'
            '- "right_under_eye_box": tight box under right eye\n'
            '- "micro_cues": array of short strings for subtle cues seen NOW '
            '(e.g. "slight brow arch", "under-eye twitch left")\n'
            '- "structure_notes": one sentence on face structure/orientation '
            "to help landmark alignment\n"
            "Boxes use [ymin, xmin, ymax, xmax] normalized 0-1000. "
            "Focus on eyebrows and under-eye micro-movements."
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

        def parse_box(key: str) -> list[int] | None:
            raw = payload.get(key)
            if not raw:
                return None
            return normalize_box_2d(raw)

        cues = payload.get("micro_cues") or []
        if not isinstance(cues, list):
            cues = []

        return ExpressionGuidance(
            state="ready",
            face_box_2d=parse_box("face_box_2d"),
            left_eyebrow_box=parse_box("left_eyebrow_box"),
            right_eyebrow_box=parse_box("right_eyebrow_box"),
            left_under_eye_box=parse_box("left_under_eye_box"),
            right_under_eye_box=parse_box("right_under_eye_box"),
            micro_cues=[str(c) for c in cues[:8]],
            structure_notes=str(payload.get("structure_notes", "")).strip(),
            updated_at=time.time(),
        )
