# YUBI Supervision

Full-screen **vision command center**: live webcam feed with YOLOv8 object tracking, MediaPipe pose/face/hands, **micro-expression tracking**, and **YUBI v3.0** semantic intelligence — rendered with [supervision](https://github.com/roboflow/supervision) and a Palantir-style HUD.

**Repo:** https://github.com/unnat-bak/yubi-supervision

## What it does

| Capability | Description |
|------------|-------------|
| **Input source** | Webcam, local video file, RTSP/CCTV, or YouTube URL; files play at 1× on media time |
| **Analysis profile** | General / Interview / Security presets picked on the idle screen (Interview plays files once and ends with a behavioral report) |
| **Objects** | YOLOv8 + ByteTrack; corner boxes, class labels, tracker IDs; confidence gate in UI |
| **Pose / face / hands** | MediaPipe skeleton overlays (toggle per layer) |
| **Expressions** | FACS engine: 22 action units from face blendshapes, per-subject adaptive baselines, micro (≤500 ms) vs macro episodes, EMFACS emotions incl. contempt, speech-aware gating, blink / gaze / head / self-touch, experimental pulse; dock with emotion bars, AU chips, activation timeline, and episode list |
| **Behavioral insights** | Notable episodes (micro-expressions, masking/leakage cues, strong negative affect) wake YUBI v3.0 for hedged interpretations and a running summary — never per frame |
| **AI spend** | Live spend meter vs. a per-session budget; calls skipped by change gating; clean pause when credits run out |
| **YUBI v3.0** | Semantic scene analysis (only when the scene changes), object verification for uncertain detections, sticky labels on tracked objects, optional overlay boxes |
| **Command HUD** | Session ID, frame index, uptime, source, media position, profile, FPS/latency sparkline, pipeline status LEDs |
| **Intel feed** | Real-time event chronology (config changes, v3.0, objects, notable episodes, insights, alerts) |
| **Alerts** | Watchlist classes → on-screen banner + optional webhook |
| **Capture / record** | PNG snapshot + JSON bundle; annotated MP4 to `recordings/` |
| **Session report** | After terminate: markdown log with a measured **Behavioral analysis** section, JSON episode timeline export, optional analyze → compose → audit YUBI v3.0 enrichment |

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

### AI cost controls

Every YUBI v3.0 call goes through one budgeted client, so a long session can't quietly run up a bill:

- **Change-gated scene analysis** — a cheap thumbnail diff skips calls while the scene is static; the refresh interval backs off up to `GEMINI_MAX_INTERVAL_SEC`.
- **Event-driven insights** — interpretation runs only after notable episodes (at most every `INSIGHT_MIN_INTERVAL_SEC`, plus a digest every `INSIGHT_DIGEST_INTERVAL_SEC`), with a few apex face crops at low media resolution.
- **Hard caps** — `AI_SESSION_BUDGET_USD` and `AI_MAX_CALLS_PER_MIN`; when a cap is hit or the key runs out of credits, AI pauses with a clear reason while measured analysis keeps running and is kept for the report.

## Prerecorded video & interviews

1. Type a file path or YouTube URL into **Input source** and pick the **Interview** profile.
2. **Initialize** — the file plays once at 1× (`SOURCE_REALTIME_PACING`), all timestamps are media time, and the session ends at EOF.
3. The first `EXPRESSION_CALIBRATION_SEC` per face builds a personal baseline; episodes are measured against it, not a population average.
4. On **Session complete**: **Download raw log** (chronology + behavioral analysis), **Export timeline (JSON)** (every episode, AU, cue, and timeline sample), or **Enrich with YUBI v3.0**.

What it surfaces that's hard to catch by eye: sub-500 ms micro-expressions, one-sided (contempt) movements, smiles without eye involvement, lip compression / suck, blink-rate shifts and bursts, gaze aversion, head nods vs. shakes, self-touch, and how affect shifts over time per subject. Brow movements timed with speech are labeled **speech emphasis** rather than emotion. All signals are observational indicators, not evidence of deception or intent.

## API smoke test

```bash
curl http://127.0.0.1:8000/api/health
curl -X POST http://127.0.0.1:8000/api/start
curl http://127.0.0.1:8000/api/status
curl -X POST http://127.0.0.1:8000/api/stop

./scripts/smoke.sh
```

`/api/status` includes `fps`, `latency_ms`, `degraded` subsystems, `session_id`, media position, `profile`, and `gemini` (v3.0), `expressions`, `insights`, and `ai_usage` blocks.

**Input source (not just the webcam):** type a source into the **Input source** box
on the idle screen — or set `CAMERA_SOURCE` — to analyze something other than the
local camera. Supported:

| Source | Example |
|--------|---------|
| Webcam | *(blank)* or `0` |
| Local video file | `clips/traffic.mp4` (loops; plays once under the Interview profile) |
| CCTV / IP camera | `rtsp://user:pass@10.0.0.5:554/stream1` |
| YouTube | `https://youtu.be/XXXXXXXXXXX` (resolved via `yt-dlp`) |

**Segmentation masks:** set `YOLO_MODEL=yolov8s-seg.pt` and toggle the **Masks**
layer for per-instance colored masks. **Joint tags** labels pose keypoints;
**Identity** (needs `GEMINI_API_KEY`) tags tracked people with a name / jersey
number / descriptor.

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
| `GET /api/analysis` | Measured behavioral analysis (subjects, episodes, insights) |
| `GET /api/analysis/report.md` | Behavioral analysis as markdown |
| `GET /api/analysis/export` | Full episode timeline JSON download |
| `POST /api/session-report` | Enriched markdown (blocking) |
| `POST /api/session-report/stream` | NDJSON enrichment progress |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full contracts.

## Configuration highlights

| Variable | Default | Effect |
|----------|---------|--------|
| `YOLO_MODEL` | `yolov8s.pt` | Speed vs accuracy (`yolov8n` faster) |
| `YOLO_STRIDE` | `2` | Run YOLO every N frames |
| `PROCESSING_MAX_WIDTH` | `1280` | Downscale before inference |
| `DEFAULT_PROFILE` | `general` | Preselected analysis profile |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Perception model (scene, verify, identity) |
| `GEMINI_INSIGHT_MODEL` | `gemini-3.8-flash` | Interpretation model (insights, report) |
| `GEMINI_INTERVAL_SEC` / `GEMINI_MAX_INTERVAL_SEC` | `4` / `60` | Scene refresh floor / backoff ceiling |
| `GEMINI_VERIFY_BELOW` | `0.45` | Low-confidence labels → v3.0 verify |
| `GEMINI_LABEL_CACHE_SEC` | `45` | Sticky label on tracker ID |
| `AI_SESSION_BUDGET_USD` | `1.0` | Hard spend cap per session |
| `AI_MAX_CALLS_PER_MIN` | `12` | Rate cap across all AI features |
| `EXPRESSION_SENSITIVITY` | `1.0` | Episode onset sensitivity (higher = more episodes) |
| `SESSION_REPORT_PASSES` | `2` | 1 = compose, 2 = analyze + compose, 3 = + audit |

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
