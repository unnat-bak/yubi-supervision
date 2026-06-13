# Architecture — YUBI Supervision

## Overview

YUBI Supervision is a **single-process** vision command center: one FastAPI app, one `VisionEngine`, background inference threads, and a static SPA that polls status and displays an MJPEG feed with glass HUD overlays.

```
Browser                         FastAPI (async)                    Background threads
────────                        ───────────────                    ──────────────────
index.html / app.js             main.py                            _bootstrap thread
  ├─ poll GET /api/status       ├─ routes only                     ├─ download .task / YOLO
  ├─ POST /api/config           ├─ VisionEngine singleton          ├─ open VideoCapture
  ├─ POST /api/start/stop       └─ StreamingResponse /api/stream   └─ spawn _loop thread
  └─ img GET /api/stream              reads _latest_jpeg                 ├─ read frame
                                                                      ├─ MediaPipe + YOLO
                                                                      ├─ v3.0 reconcile
                                                                      ├─ micro-expressions
                                                                      └─ encode JPEG

YUBI v3.0 (gemini_vision.py)    ExpressionEnricher (expression_vision.py)
  └─ periodic GenAI calls         └─ periodic face-crop GenAI calls
     in GeminiEnricher thread        + MicroExpressionTracker (local)
```

**Webcam capture is server-side** (OpenCV). The browser only displays JPEGs from `/api/stream`.

## Feature matrix

| Layer | Runtime toggle (`VisionConfig`) | Inference | Overlay / UI |
|-------|----------------------------------|-----------|--------------|
| Objects | `show_objects` | YOLOv8 + ByteTrack (optional stride) | Corner boxes + labels |
| Pose | `show_pose` | MediaPipe PoseLandmarker | Edge annotator |
| Face | `show_face` | MediaPipe FaceLandmarker | Edges + vertices (disabled when expressions on) |
| Hands | `show_hands` | MediaPipe HandLandmarker | Edges + vertices |
| Expressions | `show_expressions` | Face landmarks + blendshapes | Custom mesh + tracking dots (`draw_expression_overlay`) |
| YUBI v3.0 | `show_gemini` | Periodic full-frame (scaled JPEG) GenAI | Semantic boxes + scene summary in panel |

**Graceful degradation:** Subsystems that fail to load are listed in `degraded` on `/api/status`. The engine goes live if at least one subsystem succeeds.

## API contract

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | SPA shell |
| GET | `/api/health` | Liveness + `vision_state` |
| GET | `/api/config` | Layer toggles + confidence |
| POST | `/api/config` | Update runtime `VisionConfig` |
| POST | `/api/start` | Async bootstrap → `starting` |
| POST | `/api/stop` | Release camera/models → `idle` |
| GET | `/api/status` | Full telemetry (see below) |
| GET | `/api/gemini` | YUBI v3.0 insight snapshot (API name legacy) |
| GET | `/api/stream` | MJPEG multipart stream |
| GET | `/api/snapshot` | PNG of latest annotated frame (requires `live`) |
| GET | `/api/snapshot/json` | Detections + tracks bundle (requires `live`) |
| POST | `/api/record/start` | Start annotated MP4 (`recordings/`) |
| POST | `/api/record/stop` | Stop recording |
| POST | `/api/session-report` | Blocking enriched markdown (YUBI v3.0 passes) |
| POST | `/api/session-report/stream` | NDJSON progress stream for report enrichment |

### Vision state machine

```
idle ──POST /start──► starting ──success──► live
  ▲                      │
  │                      └── error (camera, all models failed)
  └── POST /stop ───── live
```

Frontend must poll `/api/status` while `starting`; `POST /api/start` does not return `live`.

### `/api/status` payload (high-signal fields)

| Field | Meaning |
|-------|---------|
| `state` | `idle` \| `starting` \| `live` \| `error` |
| `fps`, `latency_ms` | Processing performance (EMA) |
| `object_count`, `pose_count`, `face_count`, `hand_count` | Per-frame counts |
| `objects`, `tracks` | Grouped detections + per-track labels/confidence/`tracker_id` |
| `degraded` | Failed subsystems, e.g. `["hands"]` |
| `recording` | MP4 writer active |
| `alerts` | Recent watchlist hits |
| `startup_message` | Bootstrap progress text |
| `session_id`, `frame_index`, `uptime_sec` | Session telemetry |
| `config` | Current `VisionConfig` mirror |
| `gemini` | YUBI v3.0: `enabled`, `state`, `scene_summary`, `objects`, `error` |
| `expressions` | `enabled`, `state`, `events`, `micro_cues`, `structure_notes` |

### Session report stream (NDJSON)

Request body (`SessionReportRequest`):

```json
{
  "draft_markdown": "# Session log…",
  "session": { "id", "events", "finalStats", … }
}
```

Each line is a JSON object:

| `phase` | Meaning |
|---------|---------|
| `compile` | Client draft received |
| `pass` | YUBI v3.0 pass *n* starting (`pass`, `total_passes`, `message`) |
| `error` | Pass failed; best draft retained |
| `done` | Final `markdown`, `passes_completed`, `enriched` |

Enrichment runs in a **worker thread** feeding an `asyncio.Queue` so the event loop stays responsive. If `GEMINI_API_KEY` is unset or `GEMINI_ENABLED=false`, `done` returns the draft with `enriched: false`.

## Per-frame vision pipeline (`VisionEngine._loop`)

1. **Capture** BGR frame; optional downscale via `PROCESSING_MAX_WIDTH`.
2. **MediaPipe** (VIDEO mode): pose, face, hands when toggled or needed for expressions.
3. **YOLO** every `YOLO_STRIDE` frames → `sv.Detections.from_ultralytics` → ByteTrack.
4. **YUBI v3.0 reconcile** (`reconcile_tracked_objects`):
   - Merge GenAI semantic objects with local boxes.
   - Low-confidence YOLO labels below `GEMINI_VERIFY_BELOW` sent to v3.0 for verify/correct/suppress.
   - Confirmed labels stick on `tracker_id` for `GEMINI_LABEL_CACHE_SEC` without repeat API calls.
5. **Micro-expressions** (if `show_expressions`): `MicroExpressionTracker` on blendshape deltas; push face crop JPEG to `ExpressionEnricher` for structure guidance boxes.
6. **Annotate** (order): objects → pose → **expressions OR face** → hands → v3.0 semantic boxes (if enabled).
7. **Alerts** if watchlist class in reconciled detections (cooldown + optional webhook).
8. **JPEG encode** → `_latest_jpeg`; update `VisionStats`.

## YUBI v3.0 subsystem (`gemini_vision.py`)

User-facing name: **YUBI v3.0**. Implementation uses `google.genai` with settings from `GEMINI_*` env vars.

| Component | Role |
|-----------|------|
| `GeminiEnricher` | Background thread; periodic scene analysis; maintains `GeminiInsight` |
| `reconcile_tracked_objects` | Match v3.0 boxes to ByteTrack IDs; label aliases (e.g. dining table ↔ shelf) |
| `TrackLabelCache` | Sticky display labels per `tracker_id` |
| `collect_uncertain_hints` | Builds prompt hints for low-confidence local detections |
| `draw_gemini_boxes` | Optional overlay for v3.0 semantic boxes (aged out via `GEMINI_BOX_MAX_AGE_SEC`) |

States exposed to UI: `disabled`, `idle`, `thinking`, `ready`, `error`.

## Expression mode (`expression_vision.py`)

When `show_expressions` is true:

- **Face overlay path** in `vision.py` uses `draw_expression_overlay` instead of standard face keypoints.
- **Mesh:** MediaPipe tessellation + brow/eye/iris/nose connections (muted blend).
- **Tracking dots:** Always-on small dots on brow/eye/iris; brighten on micro-events.
- **`MicroExpressionTracker`:** Compares blendshape scores to rolling baseline; emits `MicroEvent` (brow vs under-eye regions).
- **`ExpressionEnricher`:** Sends cropped face JPEG to YUBI v3.0 for `ExpressionGuidance` (eyebrow/under-eye boxes); `refine_landmarks_with_guidance` nudges mesh toward AI boxes.

Frontend: right-rail **micro-expression dock** expands (`expressions-live` on `#video-wrap`) with live strip, region meters, and signal list.

## Frontend architecture

| Region | DOM / behavior |
|--------|----------------|
| Top **command bar** | Brand, telemetry strip, pipeline rail LEDs, status pill |
| Left **detections panel** | Pipeline layer toggles, confidence slider, object registry, intel feed |
| Right **rail** | YUBI v3.0 panel; expression dock (when enabled) |
| Bottom **command footer** | Initialize / Terminate / Capture / Record |
| **HUD overlay** | Grid, brackets, crosshair on live feed |
| **Session complete** | Post-terminate card → download report (stream + overlay) |
| **Polling** | `refreshStatus()` on interval while live |

Layer chips live in the **left sidebar** (`#layer-toggles`). Optimistic toggle with rollback if `POST /api/config` fails.

Client session log: events accumulated in `activeSession` during live poll; `buildSessionMarkdown()` on terminate; optional server enrichment.

## Configuration

Environment variables load via `backend/config.py` (see `.env.example`).

### Camera & processing

| Variable | Default | Notes |
|----------|---------|-------|
| `CAMERA_INDEX` | `0` | Device index |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT` | `1280` / `720` | Requested capture size |
| `CAMERA_SOURCE` | — | Video file path (CI/testing) |
| `CAMERA_TIMEOUT_SEC` | `12` | Open timeout |
| `PROCESSING_MAX_WIDTH` | `1280` | Downscale before inference |
| `STREAM_JPEG_QUALITY` | `80` | MJPEG encode quality |

### YOLO

| Variable | Default | Notes |
|----------|---------|-------|
| `YOLO_MODEL` | `yolov8s.pt` | `n` faster, `m` slower |
| `YOLO_STRIDE` | `2` | Run detector every N frames |
| `YOLO_IMGSZ`, `YOLO_IOU`, `YOLO_MAX_DET` | | Standard Ultralytics tuning |
| `DEFAULT_CONFIDENCE` | `0.35` | UI confidence gate default |

### Alerts & recording

| Variable | Default | Notes |
|----------|---------|-------|
| `ALERT_CLASSES` | — | Comma-separated COCO labels |
| `ALERT_WEBHOOK_URL` | — | Optional JSON POST |
| `ALERT_COOLDOWN_SEC` | `10` | Per-label cooldown |

### YUBI v3.0 (`GEMINI_*` env — internal names)

| Variable | Default | Notes |
|----------|---------|-------|
| `GEMINI_API_KEY` | — | Required for v3.0 features |
| `GEMINI_ENABLED` | `true` | Master switch |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model id (not shown in UI) |
| `GEMINI_INTERVAL_SEC` | `3` | Scene analysis interval |
| `GEMINI_MAX_OBJECTS` | `8` | Max semantic objects per pass |
| `GEMINI_ANALYSIS_SCALE` | `0.6` | JPEG scale for API |
| `GEMINI_VERIFY_BELOW` | `0.45` | Local conf below → v3.0 verify |
| `GEMINI_LABEL_CACHE_SEC` | `45` | Sticky label TTL on tracker |
| `SESSION_REPORT_PASSES` | `3` | Post-session markdown passes (1–3) |

## Threading & locks

| Thread | Work |
|--------|------|
| FastAPI event loop | HTTP, MJPEG read of buffer, stream report queue consumer |
| `_bootstrap` | One-time init |
| `_loop` | Per-frame capture + inference + encode |
| `GeminiEnricher` | Periodic GenAI scene calls |
| `ExpressionEnricher` | Periodic face-structure GenAI calls |
| Session report worker | Multi-pass markdown enrichment |

`VisionEngine` uses `self._lock` for `_stats`, `_latest_jpeg`, `_running`, `_config`, recording requests.

**Rule:** Never call `VideoCapture.read()`, YOLO inference, or GenAI on the asyncio event loop.

## Scaling path (planned)

Current code is modular at file level but monolithic in `vision.py`. Recommended split:

```
backend/
  api/routes.py
  vision/engine.py
  vision/detectors/{objects,pose,face,hands,expressions}.py
  intelligence/{v3_scene,v3_report,reconcile}.py
```

See [ROADMAP.md](ROADMAP.md) for production ops (Docker, WebRTC, metrics).

## Dependencies

- **supervision** — `Detections`, `KeyPoints`, annotators, ByteTrack
- **mediapipe** — Tasks API (`.task` models in `backend/models/`)
- **ultralytics** — YOLOv8 weights (project root `*.pt`)
- **google-genai** — YUBI v3.0 backend client
- **opencv-python** — capture and encode (headless in CI/server)

## Testing map

| File | Covers |
|------|--------|
| `tests/test_api.py` | Health, config, lifecycle start/stop |
| `tests/test_gemini_reconcile.py` | Label reconciliation + track cache |
| `tests/test_session_report.py` | Report enrichment offline + stream NDJSON |

Run `node --check frontend/app.js` after editing `app.js` — a top-level syntax error prevents the entire script from loading (Initialize button dead).
