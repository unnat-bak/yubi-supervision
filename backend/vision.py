from __future__ import annotations

import json
import sys
import threading
import time
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

from backend.config import Settings, get_settings
from backend.expression_vision import (
    ExpressionEnricher,
    MicroExpressionTracker,
    crop_face_jpeg,
    draw_expression_overlay,
)
from backend.gemini_vision import (
    GeminiEnricher,
    GeminiInsight,
    analysis_width_for_frame,
    effective_object_confidence,
    merge_insight_into_tracks,
    normalize_box_2d,
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

CAMERA_ERROR = (
    "Could not access the webcam. Grant camera permission in System Settings "
    "(Terminal or Python), close other apps using the camera, then click Retry."
)
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


class VisionEngine:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._lock = threading.Lock()
        self._running = False
        self._starting = False
        self._thread: threading.Thread | None = None
        self._bootstrap_thread: threading.Thread | None = None
        self._latest_jpeg: bytes | None = None
        self._stats = VisionStats()
        self._config = VisionConfig(confidence=self._settings.default_confidence)
        self._gemini = GeminiEnricher(self._settings)
        self._expression = ExpressionEnricher(self._settings)
        self._micro_tracker = MicroExpressionTracker()
        self._cap: cv2.VideoCapture | None = None
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

    def get_latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def get_config(self) -> VisionConfig:
        with self._lock:
            return VisionConfig(
                show_objects=self._config.show_objects,
                show_pose=self._config.show_pose,
                show_face=self._config.show_face,
                show_hands=self._config.show_hands,
                show_gemini=self._config.show_gemini,
                show_expressions=self._config.show_expressions,
                confidence=self._config.confidence,
            )

    def get_gemini_insight(self) -> GeminiInsight:
        return self._gemini.get_insight()

    def get_expression_status(self) -> dict[str, object]:
        with self._lock:
            enabled = self._config.show_expressions
        guidance = self._expression.get_guidance()
        events = self._micro_tracker.get_active_events()
        return {
            "enabled": enabled,
            "state": guidance.state,
            "events": [
                {
                    "label": e.label,
                    "region": e.region,
                    "intensity": e.intensity,
                    "ts": e.ts,
                }
                for e in events
            ],
            "micro_cues": guidance.micro_cues,
            "structure_notes": guidance.structure_notes,
            "error": guidance.error,
            "updated_at": guidance.updated_at,
        }

    def gemini_enabled(self) -> bool:
        return self._gemini.enabled

    def update_config(self, **kwargs: object) -> VisionConfig:
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._config, key) and value is not None:
                    setattr(self._config, key, value)
            self._expression.set_active(self._config.show_expressions)
            if not self._config.show_expressions:
                self._micro_tracker.reset()
            return VisionConfig(
                show_objects=self._config.show_objects,
                show_pose=self._config.show_pose,
                show_face=self._config.show_face,
                show_hands=self._config.show_hands,
                show_gemini=self._config.show_gemini,
                show_expressions=self._config.show_expressions,
                confidence=self._config.confidence,
            )

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
            )

    def get_snapshot(self) -> tuple[bytes, dict] | None:
        jpeg = self.get_latest_jpeg()
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

    def _open_camera(self) -> cv2.VideoCapture:
        if self._settings.camera_source:
            cap = cv2.VideoCapture(self._settings.camera_source)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(
                    f"Could not open video source: {self._settings.camera_source}"
                )
            return cap

        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        result: dict[str, cv2.VideoCapture | None] = {"cap": None}

        settings = self._settings

        def opener() -> None:
            cap = cv2.VideoCapture(settings.camera_index, backend)
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
        return cap

    def _quick_reopen_camera(self) -> cv2.VideoCapture | None:
        if self._settings.camera_source:
            return None
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self._settings.camera_index, backend)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._settings.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._settings.camera_height)
        return cap

    def start_async(self) -> None:
        with self._lock:
            if self._running or self._starting:
                return
            self._starting = True
            self._stats = VisionStats(
                state="starting",
                startup_message="Preparing vision pipeline…",
            )

        self._bootstrap_thread = threading.Thread(
            target=self._bootstrap, daemon=True
        )
        self._bootstrap_thread.start()

    def _still_starting(self) -> bool:
        with self._lock:
            return self._starting

    def _bootstrap(self) -> None:
        cap = None
        pose = None
        face = None
        hand = None
        yolo = None
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

            self._set_startup_message("Opening webcam…")
            cap = self._open_camera()

            if not self._still_starting():
                cap.release()
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
                self._release_models(cap, pose, face, hand)
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
                    self._release_models(cap, pose, face, hand)
                    return
                self._cap = cap
                self._pose = pose
                self._face = face
                self._hand = hand
                self._yolo = yolo
                self._tracker = self._make_tracker()
                self._running = True
                self._starting = False
                self._latest_jpeg = None
                self._live_since = time.time()
                self._frame_index = 0
                self._stats = VisionStats(state="live", degraded=degraded)

            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            self._gemini.start()
            self._expression.set_active(self._config.show_expressions)
        except Exception as exc:
            self._release_models(cap, pose, face, hand)
            with self._lock:
                self._starting = False
                self._running = False
                self._stats = VisionStats(state="error", error=str(exc))

    @staticmethod
    def _release_models(cap, pose, face, hand) -> None:
        if cap is not None:
            cap.release()
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
        self._expression.set_active(False)
        self._micro_tracker.reset()

        if self._bootstrap_thread and self._bootstrap_thread.is_alive():
            self._bootstrap_thread.join(timeout=3.0)
        self._bootstrap_thread = None

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

        with self._lock:
            self._release_models(self._cap, self._pose, self._face, self._hand)
            self._cap = None
            self._pose = None
            self._face = None
            self._hand = None
            self._yolo = None
            self._latest_jpeg = None
            self._live_since = None
            self._frame_index = 0
            self._stats = VisionStats(state="idle")

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

    def _loop(self) -> None:
        timestamp_ms = 0
        read_failures = 0
        fps_clock = time.perf_counter()
        fps_value = 0.0
        latency_value = 0.0
        frame_index = 0
        stride = max(1, self._settings.yolo_stride)
        last_detections = sv.Detections.empty()
        writer: cv2.VideoWriter | None = None

        with self._lock:
            degraded = list(self._stats.degraded)

        while True:
            with self._lock:
                if not self._running:
                    break
                cap = self._cap
                pose = self._pose
                face = self._face
                hand = self._hand
                yolo = self._yolo
                config = VisionConfig(
                    show_objects=self._config.show_objects,
                    show_pose=self._config.show_pose,
                    show_face=self._config.show_face,
                    show_hands=self._config.show_hands,
                    show_gemini=self._config.show_gemini,
                    show_expressions=self._config.show_expressions,
                    confidence=self._config.confidence,
                )

            if cap is None:
                break

            ok, frame = cap.read()
            if not ok:
                read_failures += 1
                if self._settings.camera_source and read_failures < 30:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                    continue
                if (
                    not self._settings.camera_source
                    and read_failures == 12
                ):
                    cap.release()
                    new_cap = self._quick_reopen_camera()
                    if new_cap is not None:
                        with self._lock:
                            self._cap = new_cap
                        cap = new_cap
                        read_failures = 0
                        continue
                if read_failures >= 30:
                    with self._lock:
                        self._running = False
                        self._stats = VisionStats(state="error", error=CAMERA_ERROR)
                    break
                time.sleep(0.03)
                continue

            read_failures = 0
            frame = downscale_frame(frame, self._settings.processing_max_width)
            height, width = frame.shape[:2]

            if self._gemini.enabled:
                analysis_width = analysis_width_for_frame(width, self._settings)
                analysis_jpeg = encode_frame_jpeg(
                    frame,
                    max_width=analysis_width,
                    quality=self._settings.gemini_jpeg_quality,
                )
                if analysis_jpeg:
                    self._gemini.push_frame(analysis_jpeg)

            yolo_confidence = effective_object_confidence(
                config.confidence,
                self._gemini.get_insight(),
                self._settings.gemini_box_max_age_sec,
            )

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms += 33
            infer_start = time.perf_counter()
            need_face = config.show_face or config.show_expressions
            pose_result = pose.detect_for_video(mp_image, timestamp_ms) if pose else None
            face_result = (
                face.detect_for_video(mp_image, timestamp_ms)
                if face and need_face
                else None
            )
            hand_result = hand.detect_for_video(mp_image, timestamp_ms) if hand else None

            micro_events: list = []
            if config.show_expressions and face_result and face_result.face_blendshapes:
                for idx, shapes in enumerate(face_result.face_blendshapes):
                    micro_events.extend(self._micro_tracker.update(idx, shapes))

            if (
                config.show_expressions
                and face_result
                and face_result.face_landmarks
            ):
                face_jpeg = crop_face_jpeg(
                    frame,
                    face_result.face_landmarks[0],
                    self._settings.gemini_analysis_scale,
                    self._settings.gemini_jpeg_quality,
                )
                if face_jpeg:
                    self._expression.push_face_frame(face_jpeg)

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
            if config.show_objects and not detections.is_empty():
                labels = build_detection_labels(detections, yolo.names)
                annotated = self._box_annotator.annotate(
                    scene=annotated,
                    detections=detections,
                )
                annotated = self._box_corner_annotator.annotate(
                    scene=annotated,
                    detections=detections,
                )
                annotated = self._label_annotator.annotate(
                    scene=annotated,
                    detections=detections,
                    labels=labels,
                )
            if config.show_pose:
                annotated = self._pose_edge.annotate(scene=annotated, key_points=pose_kp)
            if config.show_expressions and face_result and face_result.face_landmarks:
                guidance = self._expression.get_guidance()
                annotated = draw_expression_overlay(
                    annotated,
                    face_result.face_landmarks,
                    guidance,
                    micro_events,
                )
            elif config.show_face:
                annotated = self._face_edge.annotate(scene=annotated, key_points=face_kp)
                annotated = self._face_vertex.annotate(scene=annotated, key_points=face_kp)
            if config.show_hands:
                annotated = self._hand_edge.annotate(scene=annotated, key_points=hand_kp)
                annotated = self._hand_vertex.annotate(scene=annotated, key_points=hand_kp)

            gemini_insight = self._gemini.get_insight()
            if config.show_gemini and self._gemini.enabled:
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
            track_items = list_tracked_objects(detections, class_names)
            if self._gemini.enabled:
                track_items = merge_insight_into_tracks(
                    track_items,
                    detections,
                    gemini_insight,
                    width,
                    height,
                    self._settings.gemini_box_max_age_sec,
                )
            alert_items = self._fire_alerts(object_summary)
            with self._lock:
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
                    object_count=len(detections),
                    fps=round(fps_value, 1),
                    latency_ms=round(latency_value, 1),
                    objects=object_summary,
                    tracks=track_items,
                    degraded=degraded,
                    recording=writer is not None,
                    alerts=alert_items,
                )

            time.sleep(0.001)

        if writer is not None:
            writer.release()
