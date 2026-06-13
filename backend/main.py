from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.config import get_settings
from backend.schemas import (
    ConfigUpdate,
    GeminiStatusResponse,
    HealthResponse,
    StartResponse,
    StatusResponse,
)
from backend.vision import VisionEngine

settings = get_settings()
engine = VisionEngine(settings=settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await asyncio.to_thread(engine.stop)


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(settings.frontend_dir / "index.html")


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    stats = engine.get_stats()
    return HealthResponse(
        app=settings.app_name,
        vision_state=stats.state,
    )


@app.get("/api/config")
async def get_config() -> dict:
    config = engine.get_config()
    return config.__dict__


@app.post("/api/config")
async def update_config(payload: ConfigUpdate) -> dict:
    config = engine.update_config(**payload.model_dump(exclude_none=True))
    return config.__dict__


@app.post("/api/start", response_model=StartResponse)
async def start_vision() -> StartResponse:
    if engine.is_running:
        return StartResponse(state="live")
    if engine.is_starting:
        return StartResponse(state="starting")
    engine.start_async()
    return StartResponse(state="starting")


@app.post("/api/stop", response_model=StartResponse)
async def stop_vision() -> StartResponse:
    await asyncio.to_thread(engine.stop)
    return StartResponse(state="idle")


@app.get("/api/gemini", response_model=GeminiStatusResponse)
async def gemini_status() -> GeminiStatusResponse:
    insight = engine.get_gemini_insight()
    return GeminiStatusResponse(
        enabled=engine.gemini_enabled(),
        state=insight.state,
        scene_summary=insight.scene_summary,
        objects=[
            {"label": obj.label, "confidence": obj.confidence, "box_2d": obj.box_2d}
            for obj in insight.objects
        ],
        error=insight.error,
        updated_at=insight.updated_at,
    )


@app.get("/api/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    stats = engine.get_stats()
    config = engine.get_config()
    insight = engine.get_gemini_insight()
    return StatusResponse(
        state=stats.state,
        face_count=stats.face_count,
        pose_count=stats.pose_count,
        hand_count=stats.hand_count,
        object_count=stats.object_count,
        fps=stats.fps,
        latency_ms=stats.latency_ms,
        objects=stats.objects,
        tracks=stats.tracks,
        degraded=stats.degraded,
        recording=stats.recording,
        alerts=stats.alerts,
        startup_message=stats.startup_message,
        config=config.__dict__,
        gemini={
            "enabled": engine.gemini_enabled(),
            "state": insight.state,
            "scene_summary": insight.scene_summary,
            "objects": [
                {
                    "label": obj.label,
                    "confidence": obj.confidence,
                    "box_2d": obj.box_2d,
                }
                for obj in insight.objects
            ],
            "error": insight.error,
            "updated_at": insight.updated_at,
        },
        expressions=engine.get_expression_status(),
        error=stats.error,
    )


@app.get("/api/snapshot")
async def snapshot() -> Response:
    result = await asyncio.to_thread(engine.get_snapshot)
    if result is None:
        raise HTTPException(status_code=409, detail="Vision is not live")
    png, _ = result
    return Response(
        content=png,
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=snapshot.png"},
    )


@app.get("/api/snapshot/json")
async def snapshot_json() -> dict:
    result = await asyncio.to_thread(engine.get_snapshot)
    if result is None:
        raise HTTPException(status_code=409, detail="Vision is not live")
    _, payload = result
    return payload


@app.post("/api/record/start")
async def record_start() -> dict:
    if not engine.request_recording("start"):
        raise HTTPException(status_code=409, detail="Vision is not live")
    return {"recording": True}


@app.post("/api/record/stop")
async def record_stop() -> dict:
    engine.request_recording("stop")
    return {"recording": False}


async def mjpeg_stream():
    while engine.is_running or engine.is_starting:
        if engine.is_running:
            jpeg = engine.get_latest_jpeg()
            if jpeg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
        await asyncio.sleep(0.016)


@app.get("/api/stream")
async def stream() -> StreamingResponse:
    return StreamingResponse(
        mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


app.mount("/static", StaticFiles(directory=settings.frontend_dir), name="static")
