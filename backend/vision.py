from __future__ import annotations

import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import supervision as sv
from ultralytics import YOLO

from backend.config import Settings, get_settings

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
    "Could not access the webcam. On macOS, grant camera permission under "
    "System Settings → Privacy & Security → Camera for Terminal or your Python runtime."
)
def ensure_models(models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in MODEL_URLS.items():
        dest = models_dir / filename
        if dest.exists():
            continue
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, dest)
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


@dataclass
class VisionConfig:
    show_objects: bool = True
    show_pose: bool = True
    show_face: bool = True
    show_hands: bool = True
    confidence: float = 0.35


@dataclass
class VisionStats:
    state: str = "idle"
    face_count: int = 0
    pose_count: int = 0
    hand_count: int = 0
    object_count: int = 0
    fps: float = 0.0
    objects: list[dict[str, float | int | str]] = field(default_factory=list)
    tracks: list[dict[str, float | int | str | None]] = field(default_factory=list)
    startup_message: str | None = None
    error: str | None = None


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
        self._cap: cv2.VideoCapture | None = None
        self._pose: mp.tasks.vision.PoseLandmarker | None = None
        self._face: mp.tasks.vision.FaceLandmarker | None = None
        self._hand: mp.tasks.vision.HandLandmarker | None = None
        self._yolo: YOLO | None = None
        self._tracker = sv.ByteTrack(minimum_consecutive_frames=2)
        self._palette = sv.ColorPalette.from_hex([
            "#FFB020", "#22C55E", "#4DA3FF", "#FF6B6B", "#A78BFA",
            "#F472B6", "#2DD4BF", "#FB923C", "#818CF8", "#E879F9",
        ])
        self._pose_edge = sv.EdgeAnnotator(color=sv.Color.GREEN, thickness=2)
        self._face_edge = sv.EdgeAnnotator(color=sv.Color.from_hex("#FF6B6B"), thickness=1)
        self._hand_edge = sv.EdgeAnnotator(
            color=sv.Color.from_hex("#4DA3FF"), thickness=2, edges=HAND_CONNECTIONS
        )
        self._hand_vertex = sv.VertexAnnotator(
            color=sv.Color.from_hex("#4DA3FF"), radius=3
        )
        self._box_annotator = sv.BoxCornerAnnotator(
            color=self._palette,
            thickness=2,
            corner_length=14,
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
                confidence=self._config.confidence,
            )

    def update_config(self, **kwargs: object) -> VisionConfig:
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._config, key) and value is not None:
                    setattr(self._config, key, value)
            return VisionConfig(
                show_objects=self._config.show_objects,
                show_pose=self._config.show_pose,
                show_face=self._config.show_face,
                show_hands=self._config.show_hands,
                confidence=self._config.confidence,
            )

    def get_stats(self) -> VisionStats:
        with self._lock:
            return VisionStats(
                state=self._stats.state,
                face_count=self._stats.face_count,
                pose_count=self._stats.pose_count,
                hand_count=self._stats.hand_count,
                object_count=self._stats.object_count,
                fps=self._stats.fps,
                objects=list(self._stats.objects),
                tracks=list(self._stats.tracks),
                startup_message=self._stats.startup_message,
                error=self._stats.error,
            )

    def _set_startup_message(self, message: str) -> None:
        with self._lock:
            self._stats = VisionStats(state="starting", startup_message=message)

    def _open_camera(self) -> cv2.VideoCapture:
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

    def _bootstrap(self) -> None:
        cap = None
        pose = None
        face = None
        hand = None
        try:
            models_dir = self._settings.models_dir
            self._set_startup_message("Checking MediaPipe models…")
            ensure_models(models_dir)

            with self._lock:
                if not self._starting:
                    return

            self._set_startup_message("Opening webcam…")
            cap = self._open_camera()

            with self._lock:
                if not self._starting:
                    cap.release()
                    return

            self._set_startup_message("Loading pose, face, and hand models…")
            base = mp.tasks.BaseOptions
            vision = mp.tasks.vision
            running_mode = mp.tasks.vision.RunningMode.VIDEO

            pose = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=base(
                        model_asset_path=str(models_dir / "pose_landmarker_lite.task")
                    ),
                    running_mode=running_mode,
                    num_poses=2,
                )
            )
            face = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=base(
                        model_asset_path=str(models_dir / "face_landmarker.task")
                    ),
                    running_mode=running_mode,
                    num_faces=2,
                )
            )
            hand = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=base(
                        model_asset_path=str(models_dir / "hand_landmarker.task")
                    ),
                    running_mode=running_mode,
                    num_hands=2,
                )
            )

            with self._lock:
                if not self._starting:
                    self._release_models(cap, pose, face, hand)
                    return

            self._set_startup_message("Loading object detection model…")
            yolo = YOLO(self._settings.yolo_model)

            with self._lock:
                if not self._starting:
                    self._release_models(cap, pose, face, hand)
                    return
                self._cap = cap
                self._pose = pose
                self._face = face
                self._hand = hand
                self._yolo = yolo
                self._tracker = sv.ByteTrack(minimum_consecutive_frames=2)
                self._running = True
                self._starting = False
                self._latest_jpeg = None
                self._stats = VisionStats(state="live")

            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
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
            self._stats = VisionStats(state="idle")

    def _detect_objects(
        self, frame: np.ndarray, yolo: YOLO, confidence: float
    ) -> sv.Detections:
        results = yolo(
            frame,
            imgsz=self._settings.yolo_imgsz,
            conf=confidence,
            iou=0.5,
            max_det=30,
            verbose=False,
        )[0]
        detections = sv.Detections.from_ultralytics(results)
        return self._tracker.update_with_detections(detections)

    def _loop(self) -> None:
        timestamp_ms = 0
        read_failures = 0
        fps_clock = time.perf_counter()
        fps_value = 0.0

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
                    confidence=self._config.confidence,
                )

            if cap is None or pose is None or face is None or hand is None or yolo is None:
                break

            ok, frame = cap.read()
            if not ok:
                read_failures += 1
                if read_failures >= 30:
                    with self._lock:
                        self._running = False
                        self._stats = VisionStats(state="error", error=CAMERA_ERROR)
                    break
                time.sleep(0.03)
                continue

            read_failures = 0
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms += 33
            pose_result = pose.detect_for_video(mp_image, timestamp_ms)
            face_result = face.detect_for_video(mp_image, timestamp_ms)
            hand_result = hand.detect_for_video(mp_image, timestamp_ms)

            detections = (
                self._detect_objects(frame, yolo, config.confidence)
                if config.show_objects
                else sv.Detections.empty()
            )

            resolution = (width, height)
            pose_kp = sv.KeyPoints.from_mediapipe(pose_result, resolution)
            face_kp = sv.KeyPoints.from_mediapipe(face_result, resolution)
            hand_kp = keypoints_from_hands(hand_result, resolution)

            annotated = frame.copy()
            if config.show_objects and not detections.is_empty():
                labels = build_detection_labels(detections, yolo.names)
                annotated = self._box_annotator.annotate(
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
            if config.show_face:
                annotated = self._face_edge.annotate(scene=annotated, key_points=face_kp)
            if config.show_hands:
                annotated = self._hand_edge.annotate(scene=annotated, key_points=hand_kp)
                annotated = self._hand_vertex.annotate(scene=annotated, key_points=hand_kp)

            ok, jpeg = cv2.imencode(
                ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            )
            if not ok:
                continue

            now = time.perf_counter()
            elapsed = now - fps_clock
            fps_clock = now
            if elapsed > 0:
                fps_value = 0.85 * fps_value + 0.15 * (1.0 / elapsed)

            object_summary = summarize_objects(detections, yolo.names)
            track_items = list_tracked_objects(detections, yolo.names)
            with self._lock:
                self._latest_jpeg = jpeg.tobytes()
                self._stats = VisionStats(
                    state="live",
                    face_count=len(face_kp.xy) if config.show_face else 0,
                    pose_count=len(pose_kp.xy) if config.show_pose else 0,
                    hand_count=len(hand_kp.xy) if config.show_hands else 0,
                    object_count=len(detections),
                    fps=round(fps_value, 1),
                    objects=object_summary,
                    tracks=track_items,
                )

            time.sleep(0.001)
