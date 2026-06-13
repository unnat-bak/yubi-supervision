# YUBI Supervision

Full-screen **vision command center**: live webcam feed with YOLOv8 object tracking, MediaPipe pose/face/hands, **micro-expression tracking**, and **YUBI v3.0** semantic intelligence — rendered with [supervision](https://github.com/roboflow/supervision) and a Palantir-style HUD.

**Repo:** https://github.com/unnat-bak/yubi-supervision

## What it does

| Capability | Description |
|------------|-------------|
| **Objects** | YOLOv8 + ByteTrack; corner boxes, class labels, tracker IDs; confidence gate in UI |
| **Pose / face / hands** | MediaPipe skeleton overlays (toggle per layer) |
| **Expressions** | High-precision facial mesh + live tracking dots; micro-movement detection from blendshapes; dedicated right-rail dock with region meters and live signals |
| **YUBI v3.0** | Semantic scene analysis, object verification for uncertain detections, sticky labels on tracked objects, optional overlay boxes |
| **Command HUD** | Session ID, frame index, uptime, FPS/latency sparkline, pipeline status LEDs |
| **Intel feed** | Real-time event chronology (config changes, v3.0, objects, expressions, alerts) |
| **Alerts** | Watchlist classes → on-screen banner + optional webhook |
| **Capture / record** | PNG snapshot + JSON bundle; annotated MP4 to `recordings/` |
| **Session report** | After terminate: download markdown session log; optional multi-pass YUBI v3.0 enrichment with progress UI |

Webcam capture runs **in the Python process** (OpenCV), not in the browser. On macOS, grant camera access to Terminal or your IDE.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"          # optional: ruff, pytest

cp .env.example .env             # optional: camera, YOLO, YUBI v3.0 key
./scripts/dev.sh
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), click **Initialize**.

**Keyboard:** `Space` initialize · `S` capture · `Q` / `Esc` terminate

**First run** (~30s cold start): MediaPipe `.task` files → `backend/models/`; YOLO weights → project root (`yolov8s.pt` by default).

### YUBI v3.0 setup

Add a Google AI Studio API key to `.env` (powers YUBI v3.0 — not shown in the UI by vendor name):

```bash
GEMINI_API_KEY=your_key_here
GEMINI_ENABLED=true
```

Without a key, local vision still works; v3.0 panels and report enrichment fall back to raw data.

## API smoke test

```bash
curl http://127.0.0.1:8000/api/health
curl -X POST http://127.0.0.1:8000/api/start
curl http://127.0.0.1:8000/api/status
curl -X POST http://127.0.0.1:8000/api/stop

./scripts/smoke.sh
```

`/api/status` includes `fps`, `latency_ms`, `degraded` subsystems, `session_id`, `gemini` (v3.0), and `expressions` blocks.

**No webcam:** set `CAMERA_SOURCE=path/to/video.mp4` (loops as feed).

## UI layout

```
┌─────────────────────────────────────────────────────────────┐
│  Command bar — telemetry · pipeline rail · status           │
├──────────────┬──────────────────────────────┬───────────────┤
│ Object       │                              │ YUBI v3.0     │
│ registry     │      Live MJPEG feed         │ + Expression  │
│ Pipeline     │      + HUD overlay           │ dock (when   │
│ layers       │                              │ enabled)      │
│ Intel feed   │                              │               │
├──────────────┴──────────────────────────────┴───────────────┤
│  Initialize · Terminate · Capture · Record                  │
└─────────────────────────────────────────────────────────────┘
```

Layer toggles live in the **left sidebar** (not the bottom bar).

## Endpoints (summary)

| Endpoint | Effect |
|----------|--------|
| `GET /api/stream` | MJPEG annotated video |
| `GET /api/snapshot` | PNG frame |
| `GET /api/snapshot/json` | Detections + tracks JSON |
| `POST /api/record/start` / `stop` | Annotated MP4 |
| `POST /api/session-report` | Enriched markdown (blocking) |
| `POST /api/session-report/stream` | NDJSON enrichment progress |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full contracts.

## Configuration highlights

| Variable | Default | Effect |
|----------|---------|--------|
| `YOLO_MODEL` | `yolov8s.pt` | Speed vs accuracy (`yolov8n` faster) |
| `YOLO_STRIDE` | `2` | Run YOLO every N frames |
| `PROCESSING_MAX_WIDTH` | `1280` | Downscale before inference |
| `GEMINI_VERIFY_BELOW` | `0.45` | Low-confidence labels → v3.0 verify |
| `GEMINI_LABEL_CACHE_SEC` | `45` | Sticky label on tracker ID |
| `SESSION_REPORT_PASSES` | `3` | YUBI v3.0 passes on session export |

Full list: [.env.example](.env.example)

### Alerts

```bash
ALERT_CLASSES="person,cell phone"
ALERT_WEBHOOK_URL=https://example.com/hook   # optional
ALERT_COOLDOWN_SEC=10
```

## Project docs

| Path | Purpose |
|------|---------|
| [AGENTS.md](AGENTS.md) | Canonical instructions for AI agents |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, API, threading, features |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Production evolution |
| [backend/CLAUDE.md](backend/CLAUDE.md) | Backend module map |

## Development

```bash
ruff check backend tests
pytest
node --check frontend/app.js
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Local supervision checkout

```bash
pip install -e /path/to/supervision
# or
PYTHONPATH=/path/to/supervision/src uvicorn backend.main:app --reload
```
