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
    # Optional video file path; overrides camera_index (testing/CI without a webcam).
    camera_source: Optional[str] = None  # noqa: UP045

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

    @property
    def frontend_dir(self) -> Path:
        return ROOT_DIR / "frontend"


@lru_cache
def get_settings() -> Settings:
    return Settings()
