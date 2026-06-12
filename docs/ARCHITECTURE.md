# Architecture — YUBI Supervision

## Overview

```
Browser                    FastAPI (async)              Background threads
────────                   ───────────────              ──────────────────
index.html ──POST /start──► main.py ──start_async()──►  _bootstrap thread
app.js polls /status       (returns immediately)         ├─ download models
img ◄── GET /stream ◄──    mjpeg_stream()                ├─ open camera
                           reads latest_jpeg              ├─ load MediaPipe + YOLO
                                                        └─ spawn _loop thread
                                                           ├─ cap.read()
                                                           ├─ detect + annotate
                                                           └─ encode JPEG
```

## API contract

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | SPA shell |
| GET | `/api/health` | Liveness + vision state |
| GET | `/api/config` | Layer toggles, confidence |
| POST | `/api/config` | Update runtime config |
| POST | `/api/start` | Begin async bootstrap → `starting` |
| POST | `/api/stop` | Release camera and models → `idle` |
| GET | `/api/status` | Counts, fps, tracks, `startup_message`, `error` |
| GET | `/api/stream` | MJPEG multipart stream |

### State machine

```
idle ──POST /start──► starting ──success──► live
  ▲                      │
  │                      └── error (camera, model load)
  └── POST /stop ───── live
```

Frontend must poll `/api/status` while `starting`; never assume `POST /start` returns `live`.

## Vision pipeline (per frame)

1. BGR frame from OpenCV
2. MediaPipe VIDEO mode: pose, face, hand (`detect_for_video`)
3. YOLOv8 → `sv.Detections.from_ultralytics` → ByteTrack
4. Annotate (order): objects (boxes/labels) → pose edges → face edges → hand edges/vertices
5. JPEG encode → `_latest_jpeg`

## Configuration

Environment variables (see `.env.example`) load via `backend/config.py`:

- `CAMERA_INDEX`, `CAMERA_WIDTH`, `CAMERA_HEIGHT`
- `YOLO_MODEL`, `YOLO_IMGSZ`, `DEFAULT_CONFIDENCE`
- `HOST`, `PORT` (for `scripts/dev.sh`)

## Scaling path (planned)

Current code is intentionally monolithic. Recommended split when adding features:

```
backend/
  api/
    routes.py          # FastAPI routers
    deps.py            # VisionEngine dependency
  vision/
    engine.py          # VisionEngine orchestration
    detectors/
      objects.py       # YOLO + ByteTrack
      pose.py
      face.py
      hands.py
    annotators.py      # supervision annotator setup
    models.py          # MediaPipe / YOLO download helpers
  config.py
  schemas.py
```

Add structured logging (`structlog` or stdlib), health checks per subsystem, and graceful degradation (e.g. objects-only mode if MediaPipe fails).

## Deployment notes (future)

- **Docker:** install `opencv-python-headless`, expose port 8000, pass `/dev/video0` on Linux.
- **macOS dev:** camera permission on host process; not suitable for naive Docker-on-Mac webcam passthrough.
- **Stream latency:** MJPEG is simple but high latency; WebRTC or HLS for production.
- **Secrets:** no secrets today; use env vars when adding API keys or auth.

## Dependencies

- **supervision** — `KeyPoints`, `Detections`, annotators, ByteTrack
- **mediapipe** — Tasks API (`.task` models)
- **ultralytics** — YOLOv8 weights auto-download
- **opencv-python** — capture and encode (use headless in CI/server)
