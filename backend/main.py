from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.vision import VisionEngine

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"

app = FastAPI(title="YUBI Supervision")
engine = VisionEngine()


class ConfigUpdate(BaseModel):
    show_objects: Optional[bool] = None
    show_pose: Optional[bool] = None
    show_face: Optional[bool] = None
    show_hands: Optional[bool] = None
    confidence: Optional[float] = Field(default=None, ge=0.1, le=0.95)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/config")
async def get_config() -> dict[str, Any]:
    config = engine.get_config()
    return config.__dict__


@app.post("/api/config")
async def update_config(payload: ConfigUpdate) -> dict[str, Any]:
    config = engine.update_config(**payload.model_dump(exclude_none=True))
    return config.__dict__


@app.post("/api/start")
async def start_vision() -> dict:
    if engine.is_running:
        return {"state": "live"}
    if engine.is_starting:
        return {"state": "starting"}
    engine.start_async()
    return {"state": "starting"}


@app.post("/api/stop")
async def stop_vision() -> dict:
    engine.stop()
    return {"state": "idle"}


@app.get("/api/status")
async def status() -> dict:
    stats = engine.get_stats()
    config = engine.get_config()
    return {
        "state": stats.state,
        "face_count": stats.face_count,
        "pose_count": stats.pose_count,
        "hand_count": stats.hand_count,
        "object_count": stats.object_count,
        "fps": stats.fps,
        "objects": stats.objects,
        "tracks": stats.tracks,
        "startup_message": stats.startup_message,
        "config": config.__dict__,
        "error": stats.error,
    }


async def mjpeg_stream():
    while engine.is_running or engine.is_starting:
        if engine.is_running:
            jpeg = engine.get_latest_jpeg()
            if jpeg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
        await asyncio.sleep(0.033)


@app.get("/api/stream")
async def stream() -> StreamingResponse:
    return StreamingResponse(
        mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
