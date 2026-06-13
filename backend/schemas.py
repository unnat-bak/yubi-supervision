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
    confidence: Optional[float] = Field(default=None, ge=0.1, le=0.95)


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
