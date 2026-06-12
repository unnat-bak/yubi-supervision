# YUBI Supervision

Minimal full-screen webcam vision app using MediaPipe, YOLOv8, and [supervision](https://github.com/roboflow/supervision) for live detection and keypoint visualization.

**Repo:** https://github.com/unnat-bak/yubi-supervision

## Capabilities

- **Objects** — YOLOv8 with ByteTrack tracking, per-class corner boxes and labels (80 COCO classes)
- **Body pose** — MediaPipe pose skeleton (green), up to 2 people
- **Face mesh** — MediaPipe face landmarks (coral), up to 2 faces
- **Hands** — MediaPipe hand landmarks (blue), up to 2 hands

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"          # optional: ruff, pytest

cp .env.example .env             # optional: tune camera / YOLO settings
./scripts/dev.sh
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), click **Start Vision**, and press `Q` or `Esc` to stop.

On first run:

- MediaPipe `.task` models download into `backend/models/`
- YOLO weights (default `yolov8s.pt`) download via Ultralytics

## API smoke test

```bash
curl http://127.0.0.1:8000/api/health
curl -X POST http://127.0.0.1:8000/api/start
curl http://127.0.0.1:8000/api/status
curl -X POST http://127.0.0.1:8000/api/stop

./scripts/smoke.sh                       # full lifecycle integration test
```

`/api/status` reports `fps`, `latency_ms` (inference EMA), and `degraded` — a list
of subsystems (`pose`, `face`, `hands`, `objects`) that failed to load. The engine
goes live with whatever models loaded; it errors only if none did.

To run without a webcam (testing/CI), set `CAMERA_SOURCE=path/to/video.mp4` —
the file loops as the camera feed.

## Snapshot, recording, and alerts

| Endpoint | Effect |
|----------|--------|
| `GET /api/snapshot` | PNG of the latest annotated frame |
| `GET /api/snapshot/json` | Detections bundle (objects, tracks, fps, latency) |
| `POST /api/record/start` | Begin writing annotated MP4 to `recordings/` |
| `POST /api/record/stop` | Finish the clip |

Alerts fire when a watched class appears (with cooldown), shown as a UI banner +
sound and optionally POSTed as JSON to a webhook:

```bash
ALERT_CLASSES="person,cell phone"        # comma-separated COCO labels
ALERT_WEBHOOK_URL=https://example.com/hook   # optional
ALERT_COOLDOWN_SEC=10
```

Recent alerts appear in `/api/status` under `alerts`; `recording` reflects clip state.

## Detection tuning

| Env | Default | Effect |
|-----|---------|--------|
| `YOLO_MODEL` | `yolov8s.pt` | `yolov8n.pt` is ~2× faster, less accurate; `yolov8m.pt` more accurate, slower |
| `YOLO_STRIDE` | `2` | Run YOLO every N frames, reusing tracked boxes in between (FPS boost) |
| `YOLO_IOU` | `0.5` | NMS overlap threshold |
| `YOLO_MAX_DET` | `30` | Max detections per frame |

## macOS camera permission

Webcam capture runs **server-side** in OpenCV — not in the browser. If the camera fails to open, grant access to Terminal (or your Python runtime) under **System Settings → Privacy & Security → Camera**.

## Project layout

| Path | Purpose |
|------|---------|
| [AGENTS.md](AGENTS.md) | Canonical instructions for AI agents |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design and API contract |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Production evolution plan |
| `backend/config.py` | Environment-based settings |
| `backend/vision.py` | Vision pipeline (split later per roadmap) |
| `frontend/` | Static SPA (no build step) |

See [CONTRIBUTING.md](CONTRIBUTING.md) for lint/test workflow.

## Configuration

Environment variables (see [.env.example](.env.example)):

| Variable | Default | Description |
|----------|---------|-------------|
| `CAMERA_INDEX` | `0` | Webcam device index |
| `YOLO_MODEL` | `yolov8s.pt` | Ultralytics weights |
| `DEFAULT_CONFIDENCE` | `0.35` | Object detection threshold |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Dev server bind |

## Local supervision checkout

To develop against a local clone of supervision:

```bash
pip install -e /path/to/supervision
```

Or:

```bash
PYTHONPATH=/path/to/supervision/src uvicorn backend.main:app --reload
```
