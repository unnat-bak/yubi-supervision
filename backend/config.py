"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent / "models"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "YUBI Supervision"
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False

    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_timeout_sec: int = 12
    # Default input source. Overrides camera_index when set. May be a local video
    # file path, an RTSP/HTTP(S) stream URL, a YouTube URL, or a webcam index ("0").
    # The /api/start endpoint can override this per-session.
    camera_source: Optional[str] = None  # noqa: UP045
    # Loop file/clip sources when they reach the end (continuous feed for analysis).
    source_loop: bool = True
    # Preferred yt-dlp format selector when resolving a YouTube/stream URL.
    youtube_format: str = "best[ext=mp4][height<=1080]/best[height<=1080]/best"

    yolo_model: str = "yolov8s.pt"
    yolo_imgsz: int = 640
    yolo_iou: float = Field(default=0.5, ge=0.1, le=0.9)
    yolo_max_det: int = Field(default=30, ge=1, le=300)
    # Run YOLO every N frames, reusing tracked detections in between (FPS boost).
    yolo_stride: int = Field(default=2, ge=1, le=5)
    default_confidence: float = Field(default=0.35, ge=0.1, le=0.95)

    models_dir: Path = DEFAULT_MODELS_DIR
    recordings_dir: Path = ROOT_DIR / "recordings"

    # Comma-separated class labels that trigger alerts (e.g. "person,cell phone")
    alert_classes: str = ""
    alert_webhook_url: Optional[str] = None  # noqa: UP045
    alert_cooldown_sec: int = Field(default=10, ge=1, le=3600)

    gemini_api_key: str = ""
    gemini_enabled: bool = True
    gemini_model: str = "gemini-2.5-flash"
    gemini_interval_sec: float = Field(default=3.0, ge=2.0, le=30.0)
    gemini_max_objects: int = Field(default=8, ge=1, le=25)
    # Fraction of processing frame width sent to v3.0 (0.6 = 60% — cheaper, still accurate).
    gemini_analysis_scale: float = Field(default=0.6, ge=0.3, le=1.0)
    gemini_analysis_max_width: int = Field(default=0, ge=0, le=1280)
    gemini_jpeg_quality: int = Field(default=70, ge=50, le=95)
    gemini_box_max_age_sec: float = Field(default=6.0, ge=2.0, le=30.0)
    # Local detections below this confidence are sent to v3.0 for verify/correct.
    gemini_verify_below: float = Field(default=0.45, ge=0.1, le=0.9)
    # Confirmed v3.0 labels stick on a tracker_id without re-querying the API.
    gemini_label_cache_sec: float = Field(default=45.0, ge=5.0, le=300.0)
    # YUBI v3.0 enrichment passes for post-session markdown (1–3).
    session_report_passes: int = Field(default=3, ge=1, le=3)

    # Identity layer (YUBI v3.0 vision): name/jersey/descriptor per tracked person.
    identity_enabled: bool = True
    identity_interval_sec: float = Field(default=4.0, ge=2.0, le=30.0)
    identity_max_persons: int = Field(default=6, ge=1, le=20)
    # Confirmed identities stick on a tracker_id this long without re-querying.
    identity_label_cache_sec: float = Field(default=90.0, ge=10.0, le=600.0)

    # Default overlay layers (UI can toggle live; these set the initial state).
    show_masks_default: bool = False
    show_pose_labels_default: bool = False
    show_identity_default: bool = False

    # Downscale camera frames before inference (major FPS win on 1080p webcams).
    processing_max_width: int = Field(default=1280, ge=640, le=1920)
    stream_jpeg_quality: int = Field(default=80, ge=50, le=95)

    @property
    def gemini_active(self) -> bool:
        return bool(self.gemini_api_key) and self.gemini_enabled

    @property
    def frontend_dir(self) -> Path:
        return ROOT_DIR / "frontend"


@lru_cache
def get_settings() -> Settings:
    return Settings()
