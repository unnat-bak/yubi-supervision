"""Pydantic schemas for API request and response bodies."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ConfigUpdate(BaseModel):
    show_objects: Optional[bool] = None
    show_pose: Optional[bool] = None
    show_face: Optional[bool] = None
    show_hands: Optional[bool] = None
    show_gemini: Optional[bool] = None
    show_expressions: Optional[bool] = None
    show_masks: Optional[bool] = None
    show_pose_labels: Optional[bool] = None
    show_identity: Optional[bool] = None
    confidence: Optional[float] = Field(default=None, ge=0.1, le=0.95)


class StartRequest(BaseModel):
    """Optional per-session input source.

    `source` may be a local video path, an RTSP/HTTP(S) stream URL, a YouTube
    URL, or a webcam index ("0"). When omitted, the configured default
    (CAMERA_SOURCE or webcam) is used.
    """

    source: Optional[str] = None


class GeminiObjectResponse(BaseModel):
    label: str
    confidence: float
    box_2d: list[int]


class GeminiStatusResponse(BaseModel):
    enabled: bool
    state: str
    scene_summary: str = ""
    objects: list[GeminiObjectResponse] = Field(default_factory=list)
    error: Optional[str] = None
    updated_at: Optional[float] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    app: str
    vision_state: str


class StartResponse(BaseModel):
    state: str
    source_label: str = ""


class SessionReportRequest(BaseModel):
    draft_markdown: str = Field(min_length=1)
    session: dict[str, Any] = Field(default_factory=dict)


class SessionReportResponse(BaseModel):
    markdown: str
    passes_completed: int = 0
    enriched: bool = False


class StatusResponse(BaseModel):
    state: str
    face_count: int = 0
    pose_count: int = 0
    hand_count: int = 0
    object_count: int = 0
    fps: float = 0.0
    latency_ms: float = 0.0
    objects: list[dict[str, Any]] = Field(default_factory=list)
    tracks: list[dict[str, Any]] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)
    recording: bool = False
    alerts: list[dict[str, Any]] = Field(default_factory=list)
    startup_message: Optional[str] = None
    config: dict[str, Any] = Field(default_factory=dict)
    gemini: dict[str, Any] = Field(default_factory=dict)
    expressions: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    session_id: str = ""
    frame_index: int = 0
    uptime_sec: float = 0.0
    source_label: str = ""
    source_kind: str = ""
