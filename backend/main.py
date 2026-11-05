from __future__ import annotations

import asyncio
import json
import threading
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.config import get_settings
from backend.profiles import PROFILES
from backend.schemas import (
    ConfigUpdate,
    GeminiStatusResponse,
    HealthResponse,
    SessionReportRequest,
    SessionReportResponse,
    StartRequest,
    StartResponse,
    StatusResponse,
)
from backend.session_report import (
    PASS_LABELS,
    enrich_session_report_events,
    pass_plan,
    render_analysis_markdown,
)
from backend.sources import get_youtube_cache_dir, prune_youtube_cache
from backend.vision import VisionEngine

settings = get_settings()
engine = VisionEngine(settings=settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    cache_dir = get_youtube_cache_dir(settings.youtube_cache_dir)
    prune_youtube_cache(cache_dir, float(settings.youtube_cache_max_age_hours))
    yield
    await asyncio.to_thread(engine.stop)


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(settings.frontend_dir / "index.html")


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    stats = engine.get_stats()
    cache_dir = get_youtube_cache_dir(settings.youtube_cache_dir)
    return HealthResponse(
        app=settings.app_name,
        vision_state=stats.state,
        youtube_cache_dir=str(cache_dir),
        profiles=[profile.as_dict() for profile in PROFILES.values()],
        default_profile=settings.default_profile,
        report_steps=[PASS_LABELS[step] for step in pass_plan(settings.session_report_passes)],
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
async def start_vision(
    payload: StartRequest = Body(default_factory=StartRequest),
) -> StartResponse:
    if engine.is_running:
        return StartResponse(state="live", source_label=engine.source_label)
    if engine.is_starting:
        return StartResponse(state="starting", source_label=engine.source_label)
    source = (payload.source or "").strip() or None
    engine.start_async(source, payload.profile)
    return StartResponse(state="starting", profile=engine.get_stats().profile)


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
        insights=engine.get_intelligence_status(),
        ai_usage=engine.get_ai_usage(),
        error=stats.error,
        session_id=stats.session_id,
        frame_index=stats.frame_index,
        uptime_sec=stats.uptime_sec,
        source_label=stats.source_label,
        source_kind=stats.source_kind,
        source_is_live=stats.source_is_live,
        stream_frame_seq=stats.stream_frame_seq,
        stream_fps=stats.stream_fps,
        profile=stats.profile,
        media_position_sec=stats.media_position_sec,
        media_duration_sec=stats.media_duration_sec,
        processing_speed=stats.processing_speed,
    )


@app.get("/api/analysis")
async def analysis() -> dict:
    return await asyncio.to_thread(engine.get_analysis)


@app.get("/api/analysis/export")
async def analysis_export() -> Response:
    data = await asyncio.to_thread(engine.get_analysis)
    session_id = (data.get("session") or {}).get("session_id") or "session"
    return Response(
        content=json.dumps(data, ensure_ascii=False, indent=2),
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="yubi-analysis-{session_id}.json"'
        },
    )


@app.get("/api/analysis/report.md", response_class=PlainTextResponse)
async def analysis_markdown() -> PlainTextResponse:
    data = await asyncio.to_thread(engine.get_analysis)
    return PlainTextResponse(render_analysis_markdown(data), media_type="text/markdown")


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


def _report_events(payload: SessionReportRequest):
    return enrich_session_report_events(
        payload.draft_markdown,
        payload.session,
        settings,
        ai=engine.ai,
        analysis=engine.get_analysis(),
    )


@app.post("/api/session-report", response_model=SessionReportResponse)
async def session_report(payload: SessionReportRequest) -> SessionReportResponse:
    def run() -> SessionReportResponse:
        result_markdown = payload.draft_markdown
        passes_completed = 0
        enriched = False
        for event in _report_events(payload):
            if event.get("phase") == "done":
                result_markdown = str(event.get("markdown", result_markdown))
                passes_completed = int(event.get("passes_completed", 0))
                enriched = bool(event.get("enriched", False))
        return SessionReportResponse(
            markdown=result_markdown,
            passes_completed=passes_completed,
            enriched=enriched,
        )

    return await asyncio.to_thread(run)


async def session_report_stream(payload: SessionReportRequest):
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    def worker() -> None:
        try:
            for event in _report_events(payload):
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as exc:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {
                    "phase": "error",
                    "message": "Session report failed — exporting raw log.",
                    "detail": str(exc),
                },
            )
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {
                    "phase": "done",
                    "message": "Session report ready.",
                    "markdown": payload.draft_markdown,
                    "passes_completed": 0,
                    "enriched": False,
                },
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=worker, daemon=True).start()

    while True:
        event = await queue.get()
        if event is None:
            break
        line = json.dumps(event, ensure_ascii=False) + "\n"
        yield line.encode("utf-8")


@app.post("/api/session-report/stream")
async def session_report_stream_endpoint(
    payload: SessionReportRequest,
) -> StreamingResponse:
    return StreamingResponse(
        session_report_stream(payload),
        media_type="application/x-ndjson",
    )


async def mjpeg_stream():
    last_stream_seq = 0
    while engine.is_running or engine.is_starting:
        if engine.is_running:
            jpeg = engine.get_latest_jpeg()
            if jpeg:
                seq = engine.get_stream_frame_seq()
                if seq and seq == last_stream_seq:
                    await asyncio.sleep(0.016)
                    continue
                if seq:
                    last_stream_seq = seq
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
