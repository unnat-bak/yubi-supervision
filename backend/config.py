"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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

    yolo_model: str = "yolov8s.pt"
    yolo_imgsz: int = 640
    default_confidence: float = Field(default=0.35, ge=0.1, le=0.95)

    models_dir: Path = DEFAULT_MODELS_DIR

    @property
    def frontend_dir(self) -> Path:
        return ROOT_DIR / "frontend"


@lru_cache
def get_settings() -> Settings:
    return Settings()
