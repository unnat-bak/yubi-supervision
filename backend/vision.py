from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import traceback
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import supervision as sv
from ultralytics import YOLO

from backend.ai_budget import AIBudget
from backend.ai_client import AIClient
from backend.config import Settings, get_settings
from backend.expression_vision import draw_expression_overlay
from backend.gemini_vision import (
    GeminiEnricher,
    GeminiInsight,
    TrackLabelCache,
    analysis_width_for_frame,
    build_reconciled_detection_labels,
    collect_uncertain_hints,
    effective_object_confidence,
    merge_insight_into_tracks,
    normalize_box_2d,
    reconcile_tracked_objects,
    scene_signature,
    scene_thumbnail,
)
from backend.identity_vision import (
    IdentityEnricher,
    apply_identity_labels,
    collect_person_hints,
)
from backend.insight_engine import InsightEngine
from backend.live_capture import (
    FfmpegPipeReader,
    LatestFrameSlot,
    OpenCvReader,
    ffmpeg_available,
    open_ffmpeg_pipe,
)
from backend.micro_expressions import ExpressionAnalyzer, ExpressionFrame
from backend.profiles import AnalysisProfile, resolve_profile
from backend.sources import (
    ResolvedSource,
    classify_source,
    refresh_youtube_live_stream,
    resolve_source,
)

MODEL_URLS = {
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
}

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)

# Curated BlazePose (MediaPipe, 33-landmark) joints worth labeling — index → tag.
POSE_LABELS = {
    0: "head",
    11: "L shoulder",
    12: "R shoulder",
    13: "L elbow",
    14: "R elbow",
    15: "L wrist",
    16: "R wrist",
    23: "L hip",
    24: "R hip",
    25: "L knee",
    26: "R knee",
    27: "L ankle",
    28: "R ankle",
}

CAMERA_ERROR = (
    "Could not access the webcam. Grant camera permission in System Settings "
    "(Terminal or Python), close other apps using the camera, then click Retry."
)

# Remote live feeds: lighter MJPEG publish + processing dimensions.
LIVE_REMOTE_MAX_WIDTH = 960


def _enable_ffmpeg_low_latency() -> None:
    """Reduce HLS/RTSP demuxer buffering inside OpenCV's ffmpeg backend."""
    key = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
    if "nobuffer" in os.environ.get(key, ""):
        return
    os.environ[key] = "fflags;nobuffer|flags;low_delay|max_delay;500000"


@dataclass(frozen=True)
class FrameMeta:
    """Decode position of a captured frame: index within the clip and loop count."""

    index: int
    loop: int


class _CaptureWorker:
    """Dedicated capture thread — publishes into a ``LatestFrameSlot``."""

    def __init__(
        self,
        reader: OpenCvReader | FfmpegPipeReader,
        slot: LatestFrameSlot,
        *,
        drop_frames: bool = True,
        loop_on_eof: bool = False,
    ) -> None:
        self._reader = reader
        self._slot = slot
        self._drop = drop_frames
        self._loop_on_eof = loop_on_eof
        self._queue: queue.Queue[tuple[np.ndarray, FrameMeta]] = queue.Queue(maxsize=1)
        self._running = False
        self._ended = False
        self._thread: threading.Thread | None = None
        self._frame_index = 0
        self._loop_count = 0

    @property
    def ended(self) -> bool:
        return self._ended or getattr(self._reader, "ended", False)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="yubi-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        interrupt = getattr(self._reader, "interrupt", None)
        if callable(interrupt):
            interrupt()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=8.0)
        self._thread = None
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _run(self) -> None:
        try:
            while self._running:
                ok, frame = self._reader.read()
                if not ok or frame is None:
                    if self._loop_on_eof and isinstance(self._reader, OpenCvReader):
                        if self._reader.seek_to_start():
                            self._frame_index = 0
                            self._loop_count += 1
                            time.sleep(0.005)
                            continue
                    self._ended = True
                    time.sleep(0.01)
                    continue
                self._ended = False
                self._slot.publish(frame)
                item = (frame, FrameMeta(index=self._frame_index, loop=self._loop_count))
                self._frame_index += 1
                if self._drop:
                    if self._queue.full():
                        try:
                            self._queue.get_nowait()
                        except queue.Empty:
                            pass
                    self._queue.put(item)
                else:
                    while self._running:
                        try:
                            self._queue.put(item, timeout=0.1)
                            break
                        except queue.Full:
                            continue
        finally:
            # OpenCV/ffmpeg must be released on the capture thread to avoid macOS SIGSEGV.
            try:
                self._reader.release()
            except Exception:
                pass

    def snapshot_frame(self) -> tuple[int, np.ndarray | None]:
        return self._slot.snapshot()

    def read_frame(
        self, timeout: float = 0.05
    ) -> tuple[bool, np.ndarray | None, FrameMeta | None]:
        try:
            frame, meta = self._queue.get(timeout=timeout)
        except queue.Empty:
            return False, None, None
        return True, frame, meta

    def read_latest(self, timeout: float = 0.05) -> tuple[bool, np.ndarray | None]:
        ok, frame, _ = self.read_frame(timeout)
        return ok, frame


def ensure_models(models_dir: Path, timeout_sec: float = 60.0) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in MODEL_URLS.items():
        dest = models_dir / filename
        if dest.exists():
            continue
        print(f"Downloading {filename}...")
        tmp = dest.with_suffix(".download")
        try:
            with urllib.request.urlopen(url, timeout=timeout_sec) as response:
                tmp.write_bytes(response.read())
            tmp.rename(dest)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        print(f"Saved {dest}")


def keypoints_from_hands(result, resolution_wh: tuple[int, int]) -> sv.KeyPoints:
    if not result.hand_landmarks:
        return sv.KeyPoints.empty()
    width, height = resolution_wh
    xy, confidence = [], []
    for hand in result.hand_landmarks:
        xy.append([[lm.x * width, lm.y * height] for lm in hand])
        confidence.append([getattr(lm, "presence", 1.0) for lm in hand])
    return sv.KeyPoints(
        xy=np.array(xy, dtype=np.float32),
        confidence=np.array(confidence, dtype=np.float32),
    )


def normalized_hand_points(result) -> list[np.ndarray]:
    if result is None or not result.hand_landmarks:
        return []
    return [
        np.array([[lm.x, lm.y] for lm in hand], dtype=np.float32)
        for hand in result.hand_landmarks
    ]


def summarize_objects(
    detections: sv.Detections, class_names: dict[int, str]
) -> list[dict[str, float | int | str]]:
    if detections.is_empty():
        return []

    grouped: dict[str, dict[str, float | int]] = {}
    for class_id, confidence in zip(detections.class_id, detections.confidence):
        if class_id is None or confidence is None:
            continue
        label = class_names.get(int(class_id), str(class_id))
        bucket = grouped.setdefault(label, {"count": 0, "confidence": 0.0})
        bucket["count"] = int(bucket["count"]) + 1
        bucket["confidence"] = max(float(bucket["confidence"]), float(confidence))

    return [
        {
            "label": label,
            "count": int(meta["count"]),
            "confidence": round(float(meta["confidence"]), 2),
        }
        for label, meta in sorted(
            grouped.items(), key=lambda item: item[1]["confidence"], reverse=True
        )
    ]


def list_tracked_objects(
    detections: sv.Detections, class_names: dict[int, str]
) -> list[dict[str, float | int | str | None]]:
    if detections.is_empty():
        return []

    items: list[dict[str, float | int | str | None]] = []
    tracker_ids = detections.tracker_id
    for index, (class_id, confidence) in enumerate(
        zip(detections.class_id, detections.confidence)
    ):
        if class_id is None or confidence is None:
            continue
        tracker_id = None
        if tracker_ids is not None and index < len(tracker_ids):
            tracker_id = int(tracker_ids[index]) if tracker_ids[index] is not None else None
        items.append(
            {
                "label": class_names.get(int(class_id), str(class_id)),
                "confidence": round(float(confidence), 2),
                "tracker_id": tracker_id,
            }
        )

    items.sort(key=lambda item: float(item["confidence"]), reverse=True)
    return items


def build_detection_labels(
    detections: sv.Detections, class_names: dict[int, str]
) -> list[str]:
    labels: list[str] = []
    tracker_ids = detections.tracker_id
    for index, (class_id, confidence) in enumerate(
        zip(detections.class_id, detections.confidence)
    ):
        if class_id is None or confidence is None:
            labels.append("object")
            continue
        label = class_names.get(int(class_id), str(class_id))
        suffix = f" #{tracker_ids[index]}" if tracker_ids is not None else ""
        labels.append(f"{label}{suffix} {confidence:.0%}")
    return labels


def draw_face_outlines(
    scene: np.ndarray, face_kp: sv.KeyPoints, color: tuple[int, int, int]
) -> np.ndarray:
    if face_kp.is_empty():
        return scene
    overlay = scene.copy()
    for face in face_kp.xy:
        points = face.astype(np.int32)
        if len(points) < 3:
            continue
        hull = cv2.convexHull(points)
        cv2.polylines(overlay, [hull], True, color, 3, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.85, scene, 0.15, 0)


def draw_pose_labels(
    scene: np.ndarray,
    pose_kp: sv.KeyPoints,
    color: tuple[int, int, int],
) -> np.ndarray:
    """Tag curated BlazePose joints (head, shoulders, elbows, …) with text."""
    if pose_kp.is_empty():
        return scene
    height, width = scene.shape[:2]
    confidence = pose_kp.confidence
    for person_idx, person in enumerate(pose_kp.xy):
        for landmark_idx, tag in POSE_LABELS.items():
            if landmark_idx >= len(person):
                continue
            if confidence is not None and person_idx < len(confidence):
                conf = confidence[person_idx]
                if landmark_idx < len(conf) and float(conf[landmark_idx]) < 0.3:
                    continue
            x, y = person[landmark_idx]
            px, py = int(x), int(y)
            if px <= 0 and py <= 0:
                continue
            px = max(0, min(width - 1, px))
            py = max(0, min(height - 1, py))
            (text_w, text_h), baseline = cv2.getTextSize(
                tag, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
            )
            bx1 = max(0, px + 4)
            by2 = max(text_h + baseline, py - 4)
            cv2.rectangle(
                scene,
                (bx1, by2 - text_h - baseline),
                (bx1 + text_w + 6, by2),
                color,
                -1,
            )
            cv2.putText(
                scene,
                tag,
                (bx1 + 3, by2 - baseline + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (20, 24, 28),
                1,
                cv2.LINE_AA,
            )
    return scene


def encode_frame_jpeg(
    frame: np.ndarray,
    max_width: int | None = None,
    quality: int = 85,
) -> bytes | None:
    target = frame
    width = frame.shape[1]
    if max_width and width > max_width:
        height = frame.shape[0]
        scale = max_width / width
        target = cv2.resize(
            frame,
            (max_width, int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    ok, jpeg = cv2.imencode(
        ".jpg", target, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    return jpeg.tobytes() if ok else None


def downscale_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    width = frame.shape[1]
    if width <= max_width:
        return frame
    height = frame.shape[0]
    scale = max_width / width
    return cv2.resize(
        frame,
        (max_width, int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def draw_gemini_boxes(
    scene: np.ndarray,
    insight: GeminiInsight,
    color: tuple[int, int, int],
    max_age_sec: float,
) -> np.ndarray:
    if not insight.objects or insight.updated_at is None:
        return scene
    if time.time() - insight.updated_at > max_age_sec:
        return scene

    height, width = scene.shape[:2]
    annotated = scene.copy()
    for item in insight.objects:
        box = normalize_box_2d(item.box_2d)
        if box is None:
            continue
        ymin, xmin, ymax, xmax = box
        x1 = int(xmin * width / 1000)
        y1 = int(ymin * height / 1000)
        x2 = int(xmax * width / 1000)
        y2 = int(ymax * height / 1000)
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"{item.label} {item.confidence:.0%}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            annotated,
            (x1, max(0, y1 - text_h - baseline - 8)),
            (x1 + text_w + 10, y1),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 5, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


@dataclass
class VisionConfig:
    show_objects: bool = True
    show_pose: bool = True
    show_face: bool = True
    show_hands: bool = True
    show_gemini: bool = True
    show_expressions: bool = False
    show_masks: bool = False
    show_pose_labels: bool = False
    show_identity: bool = False
    confidence: float = 0.35


@dataclass
class VisionStats:
    state: str = "idle"
    face_count: int = 0
    pose_count: int = 0
    hand_count: int = 0
    object_count: int = 0
    fps: float = 0.0
    latency_ms: float = 0.0
    objects: list[dict[str, float | int | str]] = field(default_factory=list)
    tracks: list[dict[str, float | int | str | None]] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    recording: bool = False
    alerts: list[dict[str, float | str]] = field(default_factory=list)
    startup_message: str | None = None
    error: str | None = None
    session_id: str = ""
    frame_index: int = 0
    uptime_sec: float = 0.0
    source_label: str = ""
    source_kind: str = ""
    source_is_live: bool = False
    stream_frame_seq: int = 0
    stream_fps: float = 0.0
    profile: str = "general"
    media_position_sec: float = 0.0
    media_duration_sec: float = 0.0
    processing_speed: float = 0.0


class _PlaybackClock:
    """Media-time bookkeeping and 1x pacing for file sources."""

    def __init__(self, fps: float, duration_sec: float, pace: bool) -> None:
        self.fps = fps if fps > 0 else 30.0
        self.duration_sec = duration_sec
        self.pace = pace
        self._loop = 0
        self._loop_offset = 0.0
        self._last_media = 0.0
        self._anchor_wall: float | None = None
        self._anchor_t = 0.0
        self._speed_wall: float | None = None
        self._speed_t = 0.0
        self.speed = 0.0

    def stamp(self, meta: FrameMeta) -> tuple[float, float]:
        """(monotonic analysis time, position within the clip) for a decoded frame."""
        media_sec = meta.index / self.fps
        if meta.loop != self._loop:
            clip_length = self.duration_sec or (self._last_media + 1.0 / self.fps)
            self._loop_offset += clip_length * (meta.loop - self._loop)
            self._loop = meta.loop
            self._anchor_wall = None
        self._last_media = media_sec
        return self._loop_offset + media_sec, media_sec

    def pace_and_measure(self, t: float) -> None:
        now = time.perf_counter()
        if self._speed_wall is not None and now > self._speed_wall:
            instant = (t - self._speed_t) / (now - self._speed_wall)
            self.speed = round(0.9 * self.speed + 0.1 * instant, 2) if self.speed else instant
        self._speed_wall, self._speed_t = now, t
        if not self.pace:
            return
        if self._anchor_wall is None:
            self._anchor_wall, self._anchor_t = now, t
            return
        delay = self._anchor_wall + (t - self._anchor_t) - now
        if delay > 0:
            time.sleep(min(delay, 0.25))
        elif delay < -0.5:
            self._anchor_wall, self._anchor_t = now, t


class VisionEngine:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._lock = threading.Lock()
        self._running = False
        self._starting = False
        self._thread: threading.Thread | None = None
        self._bootstrap_thread: threading.Thread | None = None
        self._latest_jpeg: bytes | None = None
        self._stream_jpeg: bytes | None = None
        self._stream_frame_seq = 0
        self._stream_fps = 0.0
        self._annotated_jpeg: bytes | None = None
        self._stats = VisionStats()
        self._config = VisionConfig(
            confidence=self._settings.default_confidence,
            show_masks=self._settings.show_masks_default,
            show_pose_labels=self._settings.show_pose_labels_default,
            show_identity=self._settings.show_identity_default,
        )
        self._ai_budget = AIBudget(self._settings)
        self.ai = AIClient(self._settings, self._ai_budget)
        self._gemini = GeminiEnricher(self._settings, self.ai)
        self._identity = IdentityEnricher(self._settings, self.ai)
        self._expressions = ExpressionAnalyzer(self._settings)
        self._insights = InsightEngine(self._settings, self.ai, self._expressions.summary)
        self._profile: AnalysisProfile = resolve_profile(None, self._settings.default_profile)
        self._loop_files = self._settings.source_loop
        self._media_fps = 0.0
        self._media_duration = 0.0
        self._media_position = 0.0
        self._processing_speed = 0.0
        self._session_started_at: float | None = None
        self._session_ended_at: float | None = None
        self._last_source_label = ""
        self._last_source_kind = ""
        self._verify_requested: set[int] = set()
        self._track_labels = TrackLabelCache(self._settings.gemini_label_cache_sec)
        self._source_request: str | None = None
        self._source_label = ""
        self._source_kind = ""
        self._source_is_live = False
        self._source_cleanup_path: str | None = None
        self._webcam_index = self._settings.camera_index
        self._stream_width: int | None = None
        self._stream_height: int | None = None
        self._prefer_ffmpeg_pipe = False
        self._frame_slot = LatestFrameSlot()
        self._frame_reader: OpenCvReader | FfmpegPipeReader | None = None
        self._capture_worker: _CaptureWorker | None = None
        self._pose: mp.tasks.vision.PoseLandmarker | None = None
        self._face: mp.tasks.vision.FaceLandmarker | None = None
        self._hand: mp.tasks.vision.HandLandmarker | None = None
        self._yolo: YOLO | None = None
        self._tracker = self._make_tracker()
        self._record_request: str | None = None
        self._alerts: list[dict[str, float | str]] = []
        self._alert_last_fired: dict[str, float] = {}
        self._watch_labels = {
            label.strip().lower()
            for label in self._settings.alert_classes.split(",")
            if label.strip()
        }
        self._session_id = uuid.uuid4().hex[:8].upper()
        self._live_since: float | None = None
        self._frame_index = 0
        self._loop_last_tick = 0.0
        self._loop_error_streak = 0
        self._last_youtube_reconnect_at = 0.0
        self._preview_running = False
        self._preview_thread: threading.Thread | None = None
        self._last_infer_at = 0.0
        self._last_infer_seq = 0
        self._last_stream_encoded_seq = 0
        self._stream_fps_clock = time.perf_counter()
        self._palette = sv.ColorPalette.from_hex([
            "#B8A48C",
            "#8FA89A",
            "#9AABBC",
            "#A89BB8",
            "#B0A8B8",
            "#9BA8B4",
            "#A8B4C4",
            "#B8B0A0",
            "#98A8B8",
            "#B0B8C4",
        ])
        overlay_slate = sv.Color.from_hex("#9BA8B4")
        overlay_pearl = sv.Color.from_hex("#C5CCD4")
        overlay_mist = sv.Color.from_hex("#A8B4C4")
        self._pose_edge = sv.EdgeAnnotator(color=overlay_slate, thickness=1)
        self._face_edge = sv.EdgeAnnotator(color=overlay_pearl, thickness=1)
        self._face_vertex = sv.VertexAnnotator(color=overlay_pearl, radius=1)
        self._hand_edge = sv.EdgeAnnotator(
            color=overlay_mist, thickness=1, edges=HAND_CONNECTIONS
        )
        self._hand_vertex = sv.VertexAnnotator(color=overlay_mist, radius=2)
        self._mask_annotator = sv.MaskAnnotator(
            color=self._palette,
            opacity=0.45,
            color_lookup=sv.ColorLookup.CLASS,
        )
        self._box_annotator = sv.BoxAnnotator(
            color=self._palette,
            thickness=1,
            color_lookup=sv.ColorLookup.CLASS,
        )
        self._box_corner_annotator = sv.BoxCornerAnnotator(
            color=self._palette,
            thickness=1,
            corner_length=10,
            color_lookup=sv.ColorLookup.CLASS,
        )
        self._label_annotator = sv.LabelAnnotator(
            color=self._palette,
            text_color=sv.Color.WHITE,
            text_scale=0.45,
            text_thickness=1,
            text_padding=6,
            color_lookup=sv.ColorLookup.CLASS,
        )

    @staticmethod
    def _make_tracker() -> sv.ByteTrack:
        # Long lost_track_buffer keeps IDs stable through brief occlusions;
        # minimum_consecutive_frames filters one-frame false positives.
        return sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=60,
            minimum_matching_threshold=0.8,
            minimum_consecutive_frames=2,
        )

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def is_starting(self) -> bool:
        with self._lock:
            return self._starting

    def _is_remote_live_stream(self) -> bool:
        return self._source_is_live and self._source_kind in {"youtube", "stream"}

    def get_latest_jpeg(self) -> bytes | None:
        with self._lock:
            if self._is_remote_live_stream():
                return self._stream_jpeg
            return self._latest_jpeg or self._annotated_jpeg

    def get_stream_frame_seq(self) -> int:
        with self._lock:
            return self._stream_frame_seq

    @staticmethod
    def _clone_config(config: VisionConfig) -> VisionConfig:
        return VisionConfig(
            show_objects=config.show_objects,
            show_pose=config.show_pose,
            show_face=config.show_face,
            show_hands=config.show_hands,
            show_gemini=config.show_gemini,
            show_expressions=config.show_expressions,
            show_masks=config.show_masks,
            show_pose_labels=config.show_pose_labels,
            show_identity=config.show_identity,
            confidence=config.confidence,
        )

    def get_config(self) -> VisionConfig:
        with self._lock:
            return self._clone_config(self._config)

    def get_gemini_insight(self) -> GeminiInsight:
        return self._gemini.get_insight()

    def get_expression_status(self) -> dict[str, object]:
        return self._expressions.status()

    def get_intelligence_status(self) -> dict[str, object]:
        status = self._insights.status()
        status["scene_refresh_sec"] = round(self._gemini.refresh_interval, 1)
        return status

    def get_ai_usage(self) -> dict[str, object]:
        return self._ai_budget.snapshot()

    def get_analysis(self) -> dict[str, object]:
        """Full session analysis — retained after Terminate until the next start."""
        insight = self._gemini.get_insight()
        with self._lock:
            started = self._session_started_at
            ended = self._session_ended_at
            meta = {
                "session_id": self._session_id,
                "profile": self._profile.as_dict(),
                "source_label": self._source_label or self._last_source_label,
                "source_kind": self._source_kind or self._last_source_kind,
                "media_duration_sec": round(self._media_duration, 1),
                "started_at": started,
                "ended_at": ended,
                "duration_sec": round(((ended or time.time()) - started), 1) if started else 0.0,
            }
        insights = self._insights.all_insights()
        return {
            "session": meta,
            "expressions": self._expressions.analysis(),
            "insights": insights["insights"],
            "running_summary": insights["running_summary"],
            "scene_summary": insight.scene_summary,
            "ai_usage": self._ai_budget.snapshot(),
        }

    def gemini_enabled(self) -> bool:
        return self._gemini.enabled

    def _sync_expression_layers(self) -> None:
        enabled = self._config.show_expressions
        self._expressions.set_enabled(enabled)
        self._insights.set_active(enabled and self._profile.insights)

    def update_config(self, **kwargs: object) -> VisionConfig:
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._config, key) and value is not None:
                    setattr(self._config, key, value)
            self._sync_expression_layers()
            if not self._config.show_identity:
                self._identity.cache.reset()
            return self._clone_config(self._config)

    def get_stats(self) -> VisionStats:
        with self._lock:
            uptime = 0.0
            if self._live_since is not None:
                uptime = round(time.time() - self._live_since, 1)
            return VisionStats(
                state=self._stats.state,
                face_count=self._stats.face_count,
                pose_count=self._stats.pose_count,
                hand_count=self._stats.hand_count,
                object_count=self._stats.object_count,
                fps=self._stats.fps,
                latency_ms=self._stats.latency_ms,
                objects=list(self._stats.objects),
                tracks=list(self._stats.tracks),
                degraded=list(self._stats.degraded),
                recording=self._stats.recording,
                alerts=list(self._stats.alerts),
                startup_message=self._stats.startup_message,
                error=self._stats.error,
                session_id=self._session_id,
                frame_index=self._frame_index,
                uptime_sec=uptime,
                source_label=self._source_label,
                source_kind=self._source_kind,
                source_is_live=self._source_is_live,
                stream_frame_seq=self._stream_frame_seq,
                stream_fps=self._stream_fps,
                profile=self._profile.key,
                media_position_sec=round(self._media_position, 2),
                media_duration_sec=round(self._media_duration, 2),
                processing_speed=self._processing_speed,
            )

    def get_snapshot(self) -> tuple[bytes, dict] | None:
        with self._lock:
            if self._is_remote_live_stream():
                jpeg = self._annotated_jpeg or self._stream_jpeg
            else:
                jpeg = self._latest_jpeg or self._annotated_jpeg
        if not jpeg:
            return None
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        ok, png = cv2.imencode(".png", img)
        if not ok:
            return None
        stats = self.get_stats()
        insight = self._gemini.get_insight()
        payload = {
            "captured_at": datetime.now().isoformat(timespec="seconds"),
            "object_count": stats.object_count,
            "objects": stats.objects,
            "tracks": stats.tracks,
            "fps": stats.fps,
            "latency_ms": stats.latency_ms,
            "pose_count": stats.pose_count,
            "face_count": stats.face_count,
            "hand_count": stats.hand_count,
            "gemini": {
                "scene_summary": insight.scene_summary,
                "objects": [
                    {
                        "label": obj.label,
                        "confidence": obj.confidence,
                        "box_2d": obj.box_2d,
                    }
                    for obj in insight.objects
                ],
            },
            "expressions": self.get_expression_status(),
        }
        return png.tobytes(), payload

    def request_recording(self, action: str) -> bool:
        with self._lock:
            if not self._running:
                return False
            self._record_request = action
            return True

    def _fire_alerts(self, object_summary: list[dict]) -> list[dict[str, float | str]]:
        if not self._watch_labels:
            return list(self._alerts)
        present = {str(item["label"]).lower() for item in object_summary}
        now_ts = time.time()
        for label in sorted(present & self._watch_labels):
            if now_ts - self._alert_last_fired.get(label, 0.0) < (
                self._settings.alert_cooldown_sec
            ):
                continue
            self._alert_last_fired[label] = now_ts
            alert: dict[str, float | str] = {
                "label": label,
                "ts": round(now_ts, 2),
                "time": datetime.now().strftime("%H:%M:%S"),
            }
            self._alerts.append(alert)
            self._alerts = self._alerts[-20:]
            if self._settings.alert_webhook_url:
                threading.Thread(
                    target=self._post_webhook, args=(alert,), daemon=True
                ).start()
        return list(self._alerts)

    def _post_webhook(self, alert: dict[str, float | str]) -> None:
        try:
            request = urllib.request.Request(
                self._settings.alert_webhook_url,
                data=json.dumps(alert).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(request, timeout=3)
        except Exception as exc:
            print(f"Alert webhook failed: {exc}")

    def _set_startup_message(self, message: str) -> None:
        with self._lock:
            self._stats = VisionStats(state="starting", startup_message=message)

    def _apply_live_remote_lite(self) -> None:
        if not self._settings.live_remote_lite or not self._profile.remote_live_lite:
            return
        self._config.show_pose = False
        self._config.show_face = False
        self._config.show_hands = False
        self._config.show_expressions = False
        self._config.show_masks = False
        self._config.show_pose_labels = False
        self._config.show_identity = False
        self._config.show_gemini = False

    def _end_session_from_loop(self, *, error: str | None = None) -> None:
        """Release capture and models when the inference loop exits unexpectedly."""
        self._stop_live_preview()
        self._gemini.stop()
        self._identity.stop()
        self._insights.stop()
        self._expressions.set_enabled(False)
        self._stop_capture_worker()
        with self._lock:
            self._session_ended_at = time.time()
            degraded = list(self._stats.degraded)
            self._release_models(None, self._pose, self._face, self._hand)
            self._frame_reader = None
            self._pose = None
            self._face = None
            self._hand = None
            self._yolo = None
            self._running = False
            if error:
                self._stats = VisionStats(state="error", error=error, degraded=degraded)
            else:
                self._stats = VisionStats(state="idle", degraded=degraded)

    def _reap_zombie_session(self) -> None:
        """Clean up a prior session that ended without ``stop()`` (e.g. clip EOF)."""
        self._stop_live_preview()
        self._gemini.stop()
        self._identity.stop()
        self._insights.stop()
        self._stop_capture_worker()
        with self._lock:
            if self._thread and not self._thread.is_alive():
                self._thread = None
            self._release_models(None, self._pose, self._face, self._hand)
            self._frame_reader = None
            self._pose = None
            self._face = None
            self._hand = None
            self._yolo = None
            self._running = False

    def _open_video_capture(self, resolved: ResolvedSource) -> cv2.VideoCapture:
        target = str(resolved.target)
        if resolved.is_live:
            _enable_ffmpeg_low_latency()
            cap = cv2.VideoCapture(target, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(target)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open source: {resolved.label}")
        ok, _ = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(
                f"Could not read frames from source: {resolved.label}"
            )
        if not resolved.is_live:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._tune_capture(cap)
        return cap

    def _open_live_reader(
        self, resolved: ResolvedSource
    ) -> OpenCvReader | FfmpegPipeReader:
        if (
            resolved.prefer_ffmpeg_pipe
            and resolved.stream_width
            and resolved.stream_height
            and ffmpeg_available()
        ):
            try:
                return open_ffmpeg_pipe(
                    str(resolved.target),
                    resolved.stream_width,
                    resolved.stream_height,
                )
            except Exception as exc:
                print(f"ffmpeg pipe unavailable, falling back to OpenCV: {exc}")

        cap = self._open_video_capture(resolved)
        return OpenCvReader(cap)

    def _open_source(
        self,
    ) -> tuple[ResolvedSource, OpenCvReader | FfmpegPipeReader]:
        """Resolve the requested source and open a frame reader.

        Runs in the bootstrap thread (YouTube resolution may block on network).
        """
        raw = (
            self._source_request
            if self._source_request is not None
            else self._settings.camera_source
        )
        if classify_source(raw) == "youtube":
            self._set_startup_message("Resolving YouTube source…")
        resolved = resolve_source(
            raw,
            camera_index=self._settings.camera_index,
            youtube_format=self._settings.youtube_format,
            youtube_cache_dir=self._settings.youtube_cache_dir,
            youtube_cache_max_age_hours=float(
                self._settings.youtube_cache_max_age_hours
            ),
        )

        if resolved.kind == "webcam":
            cap = self._open_webcam(int(resolved.target))
            return resolved, OpenCvReader(cap)

        if resolved.is_live and resolved.prefer_ffmpeg_pipe:
            self._set_startup_message("Opening live stream (ffmpeg)…")
            return resolved, self._open_live_reader(resolved)

        return resolved, OpenCvReader(self._open_video_capture(resolved))

    def _open_webcam(self, index: int) -> cv2.VideoCapture:
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        result: dict[str, cv2.VideoCapture | None] = {"cap": None}
        settings = self._settings

        def opener() -> None:
            cap = cv2.VideoCapture(index, backend)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.camera_height)
            if not cap.isOpened():
                cap.release()
                return
            ok, _ = cap.read()
            if not ok:
                cap.release()
                return
            result["cap"] = cap

        opener_thread = threading.Thread(target=opener, daemon=True)
        opener_thread.start()
        opener_thread.join(timeout=settings.camera_timeout_sec)
        if opener_thread.is_alive():
            raise RuntimeError(
                "Camera timed out. Allow camera access in System Settings, "
                "then close other apps using the webcam and try again."
            )

        cap = result["cap"]
        if cap is None:
            raise RuntimeError(CAMERA_ERROR)
        self._tune_capture(cap)
        return cap

    @staticmethod
    def _tune_capture(cap: cv2.VideoCapture) -> None:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _start_capture_worker(
        self,
        reader: OpenCvReader | FfmpegPipeReader,
        *,
        drop_frames: bool = True,
        loop_on_eof: bool = False,
    ) -> None:
        if self._capture_worker is not None:
            self._capture_worker.stop()
        self._capture_worker = _CaptureWorker(
            reader,
            self._frame_slot,
            drop_frames=drop_frames,
            loop_on_eof=loop_on_eof,
        )
        self._capture_worker.start()

    def _replace_capture(
        self,
        reader: OpenCvReader | FfmpegPipeReader,
        *,
        drop_frames: bool = True,
        loop_on_eof: bool = False,
    ) -> None:
        """Swap capture devices without releasing while the reader thread is active."""
        self._stop_capture_worker()
        self._frame_reader = reader
        self._start_capture_worker(
            reader, drop_frames=drop_frames, loop_on_eof=loop_on_eof
        )

    def _stop_capture_worker(self) -> None:
        if self._capture_worker is not None:
            self._capture_worker.stop()
            self._capture_worker = None

    def _stop_live_preview(self) -> None:
        self._preview_running = False
        thread = self._preview_thread
        if thread and thread.is_alive():
            thread.join(timeout=1.0)
        self._preview_thread = None

    def _stream_publisher_loop(self) -> None:
        interval = 1.0 / max(1, self._settings.live_stream_fps)
        quality = self._settings.stream_jpeg_quality
        last_seq = 0
        while self._preview_running:
            loop_start = time.perf_counter()
            with self._lock:
                if not self._running:
                    break
                capture = self._capture_worker
            if capture is None:
                time.sleep(0.05)
                continue
            seq, frame = capture.snapshot_frame()
            if seq > last_seq and frame is not None:
                preview = encode_frame_jpeg(
                    frame,
                    max_width=LIVE_REMOTE_MAX_WIDTH,
                    quality=quality,
                )
                if preview:
                    with self._lock:
                        self._stream_jpeg = preview
                        self._stream_frame_seq = seq
                        now = time.perf_counter()
                        elapsed = now - self._stream_fps_clock
                        self._stream_fps_clock = now
                        if elapsed > 0:
                            self._stream_fps = round(
                                0.85 * self._stream_fps + 0.15 * (1.0 / elapsed), 1
                            )
                    last_seq = seq
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.0, interval - elapsed))

    def _start_live_preview(self) -> None:
        self._stop_live_preview()
        self._last_stream_encoded_seq = 0
        self._stream_fps_clock = time.perf_counter()
        self._preview_running = True
        self._preview_thread = threading.Thread(
            target=self._stream_publisher_loop,
            name="yubi-stream-publisher",
            daemon=True,
        )
        self._preview_thread.start()

    def _processing_max_width(self) -> int:
        if self._source_is_live and self._source_kind in {"youtube", "stream"}:
            return min(self._settings.processing_max_width, LIVE_REMOTE_MAX_WIDTH)
        return self._settings.processing_max_width

    def _quick_reopen_youtube_live(self) -> OpenCvReader | FfmpegPipeReader | None:
        if self._source_kind != "youtube" or not self._source_is_live:
            return None
        raw = (
            self._source_request
            if self._source_request is not None
            else self._settings.camera_source
        )
        if classify_source(raw) != "youtube":
            return None
        try:
            stream_url = refresh_youtube_live_stream(raw)
        except Exception as exc:
            print(f"YouTube live refresh failed: {exc}")
            return None

        if (
            self._prefer_ffmpeg_pipe
            and self._stream_width
            and self._stream_height
            and ffmpeg_available()
        ):
            try:
                return open_ffmpeg_pipe(
                    stream_url, self._stream_width, self._stream_height
                )
            except Exception as exc:
                print(f"YouTube ffmpeg refresh failed: {exc}")

        _enable_ffmpeg_low_latency()
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            return None
        ok, _ = cap.read()
        if not ok:
            cap.release()
            return None
        self._tune_capture(cap)
        return OpenCvReader(cap)

    def _quick_reopen_camera(self) -> OpenCvReader | None:
        if self._source_kind != "webcam":
            return None
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self._webcam_index, backend)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._settings.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._settings.camera_height)
        self._tune_capture(cap)
        return OpenCvReader(cap)

    def start_async(self, source: str | None = None, profile: str | None = None) -> None:
        with self._lock:
            if self._running or self._starting:
                return
            self._starting = True
            self._source_request = source
            self._profile = resolve_profile(profile, self._settings.default_profile)
            for key, value in self._profile.layers.items():
                setattr(self._config, key, value)
            self._loop_files = (
                self._settings.source_loop
                if self._profile.loop_files is None
                else self._profile.loop_files
            )
            self._stats = VisionStats(
                state="starting",
                startup_message="Preparing vision pipeline…",
                profile=self._profile.key,
            )

        self._bootstrap_thread = threading.Thread(
            target=self._bootstrap, daemon=True
        )
        self._bootstrap_thread.start()

    @property
    def source_label(self) -> str:
        with self._lock:
            return self._source_label

    def _still_starting(self) -> bool:
        with self._lock:
            return self._starting

    def _bootstrap(self) -> None:
        self._reap_zombie_session()
        reader: OpenCvReader | FfmpegPipeReader | None = None
        pose = None
        face = None
        hand = None
        yolo = None
        resolved: ResolvedSource | None = None
        degraded: list[str] = []
        try:
            models_dir = self._settings.models_dir
            self._set_startup_message("Checking MediaPipe models…")
            try:
                ensure_models(models_dir)
            except Exception as exc:
                print(f"Model download failed (continuing degraded): {exc}")

            if not self._still_starting():
                return

            self._set_startup_message("Opening input source…")
            resolved, reader = self._open_source()
            with self._lock:
                self._source_label = resolved.label
                self._source_kind = resolved.kind
                self._source_is_live = resolved.is_live
                self._source_cleanup_path = resolved.cleanup_path
                if resolved.kind == "webcam":
                    self._webcam_index = int(resolved.target)
                self._stream_width = resolved.stream_width
                self._stream_height = resolved.stream_height
                self._prefer_ffmpeg_pipe = resolved.prefer_ffmpeg_pipe
                self._last_source_label = resolved.label
                self._last_source_kind = resolved.kind
                self._media_fps, self._media_duration = 0.0, 0.0
                if not resolved.is_live and isinstance(reader, OpenCvReader):
                    fps, frames = reader.media_info()
                    self._media_fps = fps
                    self._media_duration = frames / fps if fps > 0 and frames > 0 else 0.0
                if resolved.is_live and resolved.kind in {"youtube", "stream"}:
                    self._apply_live_remote_lite()

            if not self._still_starting():
                reader.release()
                return

            self._set_startup_message("Loading pose, face, and hand models…")
            base = mp.tasks.BaseOptions
            vision = mp.tasks.vision
            running_mode = mp.tasks.vision.RunningMode.VIDEO

            try:
                pose = vision.PoseLandmarker.create_from_options(
                    vision.PoseLandmarkerOptions(
                        base_options=base(
                            model_asset_path=str(models_dir / "pose_landmarker_lite.task")
                        ),
                        running_mode=running_mode,
                        num_poses=2,
                    )
                )
            except Exception as exc:
                print(f"Pose landmarker unavailable: {exc}")
                degraded.append("pose")
            try:
                face = vision.FaceLandmarker.create_from_options(
                    vision.FaceLandmarkerOptions(
                        base_options=base(
                            model_asset_path=str(models_dir / "face_landmarker.task")
                        ),
                        running_mode=running_mode,
                        num_faces=2,
                        output_face_blendshapes=True,
                        output_facial_transformation_matrixes=True,
                    )
                )
            except Exception as exc:
                print(f"Face landmarker unavailable: {exc}")
                degraded.append("face")
            try:
                hand = vision.HandLandmarker.create_from_options(
                    vision.HandLandmarkerOptions(
                        base_options=base(
                            model_asset_path=str(models_dir / "hand_landmarker.task")
                        ),
                        running_mode=running_mode,
                        num_hands=2,
                    )
                )
            except Exception as exc:
                print(f"Hand landmarker unavailable: {exc}")
                degraded.append("hands")

            if not self._still_starting():
                self._release_models(reader, pose, face, hand)
                return

            self._set_startup_message("Loading object detection model…")
            try:
                yolo = YOLO(self._settings.yolo_model)
            except Exception as exc:
                print(f"YOLO unavailable: {exc}")
                degraded.append("objects")

            if pose is None and face is None and hand is None and yolo is None:
                raise RuntimeError(
                    "No vision models could be loaded. "
                    "Check network access and model files, then try again."
                )

            with self._lock:
                if not self._starting:
                    self._release_models(reader, pose, face, hand)
                    return
                self._frame_reader = reader
                self._start_capture_worker(
                    reader,
                    drop_frames=resolved.is_live,
                    loop_on_eof=(not resolved.is_live) and self._loop_files,
                )
                if resolved.is_live and resolved.kind in {"youtube", "stream"}:
                    self._start_live_preview()
                self._pose = pose
                self._face = face
                self._hand = hand
                self._yolo = yolo
                self._tracker = self._make_tracker()
                self._running = True
                self._starting = False
                self._latest_jpeg = None
                self._stream_jpeg = None
                self._stream_frame_seq = 0
                self._stream_fps = 0.0
                self._annotated_jpeg = None
                self._last_infer_seq = 0
                self._last_infer_at = 0.0
                self._live_since = time.time()
                self._session_started_at = self._live_since
                self._session_ended_at = None
                self._media_position = 0.0
                self._processing_speed = 0.0
                self._frame_index = 0
                self._loop_last_tick = time.time()
                self._loop_error_streak = 0
                self._stats = VisionStats(
                    state="live", degraded=degraded, profile=self._profile.key
                )
                self._ai_budget.reset()
                self._verify_requested.clear()
                self._expressions.reset()
                self._insights.start(self._profile.key)
                self._sync_expression_layers()

            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            self._gemini.start()
            self._identity.start()
        except Exception as exc:
            self._release_models(reader, pose, face, hand)
            with self._lock:
                self._starting = False
                self._running = False
                self._stats = VisionStats(state="error", error=str(exc))

    @staticmethod
    def _release_models(
        reader: OpenCvReader | FfmpegPipeReader | None,
        pose,
        face,
        hand,
    ) -> None:
        if reader is not None:
            reader.release()
        if pose is not None:
            pose.close()
        if face is not None:
            face.close()
        if hand is not None:
            hand.close()

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._starting = False

        self._gemini.stop()
        self._identity.stop()
        self._insights.stop()
        self._expressions.set_enabled(False)
        self._track_labels.reset()
        self._stop_live_preview()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None

        self._stop_capture_worker()

        if self._bootstrap_thread and self._bootstrap_thread.is_alive():
            self._bootstrap_thread.join(timeout=3.0)
        self._bootstrap_thread = None

        cleanup_path = None
        with self._lock:
            cleanup_path = self._source_cleanup_path
            self._source_cleanup_path = None
            # Reader is released on the capture thread; do not release again here.
            self._release_models(None, self._pose, self._face, self._hand)
            self._frame_reader = None
            self._pose = None
            self._face = None
            self._hand = None
            self._yolo = None
            self._latest_jpeg = None
            self._stream_jpeg = None
            self._stream_frame_seq = 0
            self._stream_fps = 0.0
            self._annotated_jpeg = None
            if self._session_started_at is not None and self._session_ended_at is None:
                self._session_ended_at = time.time()
            self._live_since = None
            self._frame_index = 0
            self._source_label = ""
            self._source_kind = ""
            self._source_is_live = False
            self._webcam_index = self._settings.camera_index
            self._stats = VisionStats(state="idle")

        if cleanup_path:
            try:
                Path(cleanup_path).unlink(missing_ok=True)
            except OSError:
                pass

    def _detect_objects(
        self, frame: np.ndarray, yolo: YOLO, confidence: float
    ) -> sv.Detections:
        results = yolo(
            frame,
            imgsz=self._settings.yolo_imgsz,
            conf=confidence,
            iou=self._settings.yolo_iou,
            max_det=self._settings.yolo_max_det,
            verbose=False,
        )[0]
        detections = sv.Detections.from_ultralytics(results)
        return self._tracker.update_with_detections(detections)

    def _maybe_request_scene(
        self,
        frame: np.ndarray,
        config: VisionConfig,
        track_items: list[dict[str, float | int | str | None | bool]],
        detections: sv.Detections,
    ) -> bytes | None:
        """Send a frame to YUBI v3.0 only when the scene changed or a new track needs a label."""
        if not (self._gemini.enabled and config.show_gemini and self._gemini.ready()):
            return None
        height, width = frame.shape[:2]
        uncertain: list[dict[str, object]] = []
        if track_items:
            uncertain = [
                hint
                for hint in collect_uncertain_hints(
                    track_items, detections, self._settings.gemini_verify_below, width, height
                )
                if self._track_labels.get(int(hint["tracker_id"])) is None
            ]
        # Each track forces at most one early verification pass; later passes are change-gated.
        fresh = [h for h in uncertain if int(h["tracker_id"]) not in self._verify_requested]
        reason = self._gemini.consider(
            scene_thumbnail(frame), scene_signature(track_items), force=bool(fresh)
        )
        if reason is None:
            return None
        jpeg = encode_frame_jpeg(
            frame,
            max_width=analysis_width_for_frame(width, self._settings),
            quality=self._settings.gemini_jpeg_quality,
        )
        if jpeg:
            self._verify_requested.update(int(h["tracker_id"]) for h in uncertain)
            self._gemini.push_frame(jpeg, uncertain)
        return jpeg

    def _loop(self) -> None:
        timestamp_ms = -1
        read_failures = 0
        fps_clock = time.perf_counter()
        fps_value = 0.0
        latency_value = 0.0
        frame_index = 0
        stride = max(1, self._settings.yolo_stride)
        last_detections = sv.Detections.empty()
        writer: cv2.VideoWriter | None = None
        wall_start = time.perf_counter()
        last_scene_at = 0.0

        with self._lock:
            degraded = list(self._stats.degraded)
            playback = (
                None
                if self._source_is_live
                else _PlaybackClock(
                    self._media_fps,
                    self._media_duration,
                    self._settings.source_realtime_pacing,
                )
            )

        while True:
            with self._lock:
                if not self._running:
                    break
                capture = self._capture_worker
                pose = self._pose
                face = self._face
                hand = self._hand
                yolo = self._yolo
                config = self._clone_config(self._config)

            if capture is None or self._frame_reader is None:
                break

            meta: FrameMeta | None = None
            if self._is_remote_live_stream():
                seq, frame = capture.snapshot_frame()
                if frame is None:
                    ok = False
                else:
                    ok = True
                    min_gap = 1.0 / max(0.5, self._settings.live_inference_fps)
                    now = time.time()
                    if seq == self._last_infer_seq or now - self._last_infer_at < min_gap:
                        time.sleep(0.01)
                        continue
                    self._last_infer_at = now
                    self._last_infer_seq = seq
            else:
                ok, frame, meta = capture.read_frame()
            if not ok or frame is None:
                if (
                    capture.ended
                    and self._source_kind == "youtube"
                    and self._source_is_live
                ):
                    now = time.time()
                    if now - self._last_youtube_reconnect_at >= 15.0:
                        self._last_youtube_reconnect_at = now
                        new_reader = self._quick_reopen_youtube_live()
                        if new_reader is not None:
                            self._replace_capture(
                                new_reader, drop_frames=True, loop_on_eof=False
                            )
                            capture = self._capture_worker
                            read_failures = 0
                            if not self._preview_running:
                                self._start_live_preview()
                            continue

                # A non-looping clip that has played out: end the session cleanly.
                if capture.ended and not self._source_is_live:
                    self._end_session_from_loop()
                    break

                # Live HLS/RTSP often blocks in ffmpeg — an empty queue is normal;
                # never release capture from here (that races the reader thread).
                if self._source_is_live:
                    time.sleep(0.02)
                    continue

                read_failures += 1
                if read_failures == 12 and self._source_kind == "webcam":
                    new_reader = self._quick_reopen_camera()
                    if new_reader is not None:
                        self._replace_capture(new_reader, drop_frames=True)
                        capture = self._capture_worker
                        read_failures = 0
                        continue
                if read_failures >= 60:
                    message = (
                        CAMERA_ERROR
                        if self._source_kind == "webcam"
                        else f"Lost input source: {self._source_label}"
                    )
                    self._end_session_from_loop(error=message)
                    break
                time.sleep(0.02)
                continue

            try:
                read_failures = 0
                frame = downscale_frame(frame, self._processing_max_width())
                height, width = frame.shape[:2]

                if playback is not None and meta is not None:
                    analysis_t, media_sec = playback.stamp(meta)
                else:
                    analysis_t = time.perf_counter() - wall_start
                    media_sec = analysis_t

                yolo_confidence = effective_object_confidence(
                    config.confidence,
                    self._gemini.get_insight(),
                    self._settings.gemini_box_max_age_sec,
                )

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                timestamp_ms = max(timestamp_ms + 1, int(analysis_t * 1000.0))
                infer_start = time.perf_counter()
                need_face = config.show_face or config.show_expressions
                pose_result = pose.detect_for_video(mp_image, timestamp_ms) if pose else None
                face_result = (
                    face.detect_for_video(mp_image, timestamp_ms)
                    if face and need_face
                    else None
                )
                hand_result = hand.detect_for_video(mp_image, timestamp_ms) if hand else None

                expression_frame: ExpressionFrame | None = None
                if config.show_expressions and face_result is not None:
                    self._expressions.capture_crops = self._insights.wants_crops
                    expression_frame = self._expressions.process(
                        face_result,
                        frame,
                        analysis_t,
                        media_sec,
                        hand_points=normalized_hand_points(hand_result),
                    )
                    if expression_frame.episodes:
                        self._insights.push_episodes(expression_frame.episodes)

                if config.show_objects and yolo is not None:
                    if frame_index % stride == 0:
                        last_detections = self._detect_objects(
                            frame, yolo, yolo_confidence
                        )
                    detections = last_detections
                else:
                    detections = sv.Detections.empty()
                    last_detections = detections
                frame_index += 1
                with self._lock:
                    self._frame_index = frame_index
                infer_ms = (time.perf_counter() - infer_start) * 1000.0
                latency_value = (
                    0.85 * latency_value + 0.15 * infer_ms if latency_value else infer_ms
                )

                resolution = (width, height)
                pose_kp = (
                    sv.KeyPoints.from_mediapipe(pose_result, resolution)
                    if pose_result
                    else sv.KeyPoints.empty()
                )
                face_kp = (
                    sv.KeyPoints.from_mediapipe(face_result, resolution)
                    if face_result and config.show_face
                    else sv.KeyPoints.empty()
                )
                hand_kp = (
                    keypoints_from_hands(hand_result, resolution)
                    if hand_result
                    else sv.KeyPoints.empty()
                )

                annotated = frame.copy()
                track_items: list[dict[str, float | int | str | None | bool]] = []
                display_detections = detections
                box_labels: list[str] = []
                has_objects = (
                    config.show_objects and not detections.is_empty() and yolo is not None
                )
                if has_objects:
                    track_items = list_tracked_objects(detections, yolo.names)
                scene_jpeg = self._maybe_request_scene(frame, config, track_items, detections)
                gemini_insight = self._gemini.get_insight()
                if gemini_insight.updated_at and gemini_insight.updated_at != last_scene_at:
                    last_scene_at = gemini_insight.updated_at
                    self._insights.note_scene(gemini_insight.scene_summary)

                if has_objects:
                    class_names = yolo.names
                    if self._gemini.enabled and config.show_gemini:
                        track_items = reconcile_tracked_objects(
                            track_items,
                            detections,
                            gemini_insight,
                            width,
                            height,
                            self._track_labels,
                            self._settings.gemini_verify_below,
                            self._settings.gemini_box_max_age_sec,
                            config.confidence,
                        )
                        track_items = merge_insight_into_tracks(
                            track_items,
                            detections,
                            gemini_insight,
                            width,
                            height,
                            self._settings.gemini_box_max_age_sec,
                        )
                        indices, box_labels = build_reconciled_detection_labels(
                            track_items,
                            detections,
                            class_names,
                            config.confidence,
                        )
                        display_detections = (
                            detections[indices] if indices else sv.Detections.empty()
                        )
                    else:
                        box_labels = build_detection_labels(detections, class_names)

                    if (
                        config.show_identity
                        and self._identity.enabled
                        and not detections.is_empty()
                    ):
                        person_hints = collect_person_hints(
                            track_items,
                            detections,
                            width,
                            height,
                            self._settings.identity_max_persons,
                        )
                        if person_hints and self._identity.wants_new_analysis(
                            person_hints
                        ):
                            identity_jpeg = scene_jpeg or encode_frame_jpeg(
                                frame,
                                max_width=analysis_width_for_frame(
                                    width, self._settings
                                ),
                                quality=self._settings.gemini_jpeg_quality,
                            )
                            if identity_jpeg:
                                self._identity.push_frame(identity_jpeg, person_hints)
                        box_labels = apply_identity_labels(
                            display_detections, box_labels, self._identity.cache
                        )

                if config.show_objects and not display_detections.is_empty():
                    if config.show_masks and display_detections.mask is not None:
                        annotated = self._mask_annotator.annotate(
                            scene=annotated,
                            detections=display_detections,
                        )
                    annotated = self._box_annotator.annotate(
                        scene=annotated,
                        detections=display_detections,
                    )
                    annotated = self._box_corner_annotator.annotate(
                        scene=annotated,
                        detections=display_detections,
                    )
                    annotated = self._label_annotator.annotate(
                        scene=annotated,
                        detections=display_detections,
                        labels=box_labels,
                    )
                if config.show_pose:
                    annotated = self._pose_edge.annotate(scene=annotated, key_points=pose_kp)
                    if config.show_pose_labels and not pose_kp.is_empty():
                        annotated = draw_pose_labels(
                            annotated, pose_kp, color=(197, 204, 212)
                        )
                if config.show_expressions:
                    if expression_frame is not None and expression_frame.renders:
                        annotated = draw_expression_overlay(annotated, expression_frame.renders)
                elif config.show_face:
                    annotated = self._face_edge.annotate(scene=annotated, key_points=face_kp)
                    annotated = self._face_vertex.annotate(scene=annotated, key_points=face_kp)
                if config.show_hands:
                    annotated = self._hand_edge.annotate(scene=annotated, key_points=hand_kp)
                    annotated = self._hand_vertex.annotate(scene=annotated, key_points=hand_kp)

                if config.show_gemini and config.show_objects and self._gemini.enabled:
                    annotated = draw_gemini_boxes(
                        annotated,
                        gemini_insight,
                        color=(220, 220, 220),
                        max_age_sec=self._settings.gemini_box_max_age_sec,
                    )

                with self._lock:
                    record_request = self._record_request
                    self._record_request = None
                if record_request == "start" and writer is None:
                    recordings_dir = self._settings.recordings_dir
                    recordings_dir.mkdir(parents=True, exist_ok=True)
                    clip_name = datetime.now().strftime("clip-%Y%m%d-%H%M%S.mp4")
                    writer = cv2.VideoWriter(
                        str(recordings_dir / clip_name),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        max(fps_value, 10.0),
                        (width, height),
                    )
                elif record_request == "stop" and writer is not None:
                    writer.release()
                    writer = None
                if writer is not None:
                    writer.write(annotated)

                ok, jpeg = cv2.imencode(
                    ".jpg",
                    annotated,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self._settings.stream_jpeg_quality],
                )
                if not ok:
                    continue

                jpeg_bytes = jpeg.tobytes()

                now = time.perf_counter()
                elapsed = now - fps_clock
                fps_clock = now
                if elapsed > 0:
                    fps_value = 0.85 * fps_value + 0.15 * (1.0 / elapsed)

                class_names = yolo.names if yolo is not None else {}
                object_summary = summarize_objects(detections, class_names)
                if not track_items and not detections.is_empty() and yolo is not None:
                    track_items = list_tracked_objects(detections, class_names)
                alert_items = self._fire_alerts(object_summary)
                with self._lock:
                    self._annotated_jpeg = jpeg_bytes
                    if not self._is_remote_live_stream():
                        self._latest_jpeg = jpeg_bytes
                    self._stats = VisionStats(
                        state="live",
                        face_count=(
                            len(face_result.face_landmarks)
                            if face_result and face_result.face_landmarks
                            else 0
                        ),
                        pose_count=len(pose_kp.xy) if config.show_pose else 0,
                        hand_count=len(hand_kp.xy) if config.show_hands else 0,
                        object_count=len(track_items),
                        fps=round(fps_value, 1),
                        latency_ms=round(latency_value, 1),
                        objects=object_summary,
                        tracks=track_items,
                        degraded=degraded,
                        recording=writer is not None,
                        alerts=alert_items,
                    )
                    self._media_position = media_sec

                if playback is not None:
                    playback.pace_and_measure(analysis_t)
                    self._processing_speed = playback.speed

                self._loop_last_tick = time.time()
                self._loop_error_streak = 0
            except Exception:
                self._loop_error_streak += 1
                traceback.print_exc()
                if self._loop_error_streak >= 25:
                    self._end_session_from_loop(
                        error="Vision processing fault — check server logs"
                    )
                    break
                time.sleep(0.05)

            time.sleep(0.001)

        if writer is not None:
            writer.release()
