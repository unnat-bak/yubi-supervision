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
  └─ img GET /api/stream              reads stream/preview JPEG           ├─ read frame (capture thread)
                                                                      ├─ stream publisher (remote live)
                                                                      ├─ MediaPipe + YOLO (throttled on live)
                                                                      ├─ v3.0 reconcile
                                                                      ├─ ExpressionAnalyzer (local FACS engine)
                                                                      └─ encode annotated JPEG (not MJPEG on live remote)

                     AIClient + AIBudget (one per engine — every model call is admitted + metered)
                       ├─ GeminiEnricher   change-gated scene passes      (perception tier)
                       ├─ IdentityEnricher per-track identity, cached      (perception tier)
                       ├─ InsightEngine    event-driven interpretation     (reasoning tier)
                       └─ session report   analyze → compose → audit       (reasoning tier)
```

**Cost model:** the camera loop never streams frames to the model. Everything per-frame (objects, keypoints, FACS action units, blinks, gaze, head pose, pulse) is measured locally. YUBI v3.0 is called only when something changed (scene gate), when a new low-confidence track needs verifying (once per track), or when the local engine found a notable behavioral episode (insight engine). All calls share one budget with a session USD cap, a per-minute cap, and a circuit breaker that pauses — instead of retrying — on billing/quota errors.

**Webcam capture is server-side** (OpenCV). The browser only displays JPEGs from `/api/stream`.

### Remote live (YouTube / RTSP)

For `source_kind` of `youtube` or `stream` **when `source_is_live` is true**, MJPEG is **decoupled from inference**:

- **Capture thread** publishes frames into a `LatestFrameSlot` (sequence id + BGR buffer).
- **Stream publisher thread** encodes fresh JPEGs at `LIVE_STREAM_FPS` into `_stream_jpeg`; `/api/stream` always serves this buffer.
- **Inference thread** snapshots frames at `LIVE_INFERENCE_FPS` (default 2 fps); annotated JPEGs update `/api/status` only.
- YouTube live HLS prefers an **ffmpeg pipe** (`backend/live_capture.py`) with OpenCV fallback.

Frontend stall recovery uses `stream_frame_seq` / `stream_fps`, not inference `frame_index`.

## Feature matrix

| Layer | Runtime toggle (`VisionConfig`) | Inference | Overlay / UI |
|-------|----------------------------------|-----------|--------------|
| Objects | `show_objects` | YOLOv8 + ByteTrack (optional stride) | Corner boxes + labels |
| Pose | `show_pose` | MediaPipe PoseLandmarker | Edge annotator |
| Face | `show_face` | MediaPipe FaceLandmarker | Edges + vertices (disabled when expressions on) |
| Hands | `show_hands` | MediaPipe HandLandmarker | Edges + vertices |
| Expressions | `show_expressions` | Face landmarks + blendshapes + transformation matrix → FACS engine | Smoothed mesh, region heat, subject tag, micro flash (`draw_expression_overlay`); dock with affect, AUs, vitals, episodes |
| YUBI v3.0 | `show_gemini` | Change-gated scene passes; event-driven insights | Semantic boxes (with Objects) + scene summary, insights, spend meter |

### Analysis profiles (`backend/profiles.py`)

`POST /api/start {"source", "profile"}` applies a layer preset and session behavior:

| Profile | Layers | Behavior |
|---------|--------|----------|
| `general` | Objects, pose, face, hands, v3.0 | Default; files loop per `SOURCE_LOOP` |
| `interview` | Expressions, hands, v3.0 (objects/pose/face off) | Files play **once** (session ends at EOF → report), insights on, no remote-live lite mode |
| `security` | Objects, identity, v3.0 | Insights off |

### File playback

File sources are stamped with **media time** (`frame_index / fps` + loop offset) rather than wall clock, so episode times in the UI, export, and report match the video. With `SOURCE_REALTIME_PACING=true` the loop sleeps to hold 1× playback; `/api/status` exposes `media_position_sec`, `media_duration_sec`, and `processing_speed`.

**Graceful degradation:** Subsystems that fail to load are listed in `degraded` on `/api/status`. The engine goes live if at least one subsystem succeeds.

## API contract

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | SPA shell |
| GET | `/api/health` | Liveness + `vision_state`, `profiles`, `default_profile`, `report_steps` |
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
| GET | `/api/analysis` | Session analysis: per-subject summaries, episodes, affect timeline, insights, AI usage (kept after stop) |
| GET | `/api/analysis/export` | Same payload as a JSON download |
| GET | `/api/analysis/report.md` | Measured "Behavioral analysis" markdown section (no AI) |
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
| `gemini` | YUBI v3.0 scene: `enabled`, `state` (`idle`/`thinking`/`ready`/`paused`), `scene_summary`, `objects`, `error` |
| `expressions` | `enabled`, `state` (`off`/`no_face`/`calibrating`/`live`), `subjects[]` (emotions, dominant, valence/activation, regions, top AUs, blink, gaze, head, speaking, self-touch, pulse), `timeline`, `episodes` (latest 14), `counts` |
| `insights` | `state` (`off`/`idle`/`watching`/`thinking`/`paused`), `insights[]`, `running_summary`, `pending`, `scene_refresh_sec` |
| `ai_usage` | Session ledger: `calls`, `skipped` (avoided by gating), tokens, `cost_usd`, `budget_usd`, `paused`, `paused_reason`, per-feature breakdown — never model ids |
| `profile`, `media_position_sec`, `media_duration_sec`, `processing_speed` | Profile + file playback telemetry |

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
| `compile` | Draft received; `steps` lists the pass labels the UI should show |
| `pass` | Pass *n* starting (`pass`, `total_passes`, `step`, `label`, `message`) |
| `audit` | Evidence audit applied `fixes_applied` corrections |
| `error` | Pass failed; best draft retained |
| `done` | Final `markdown`, `passes_completed`, `enriched`, `cost_usd` |

The server first inserts the measured **Behavioral analysis** section (from the engine's retained analysis) ahead of the event chronology. Passes (`SESSION_REPORT_PASSES`):

1. **Analyze** (2–3) — reasoning tier reads a compact evidence payload (subject summaries, ≤150 episodes, downsampled affect timeline, insights) and returns structured findings as JSON.
2. **Compose** (1–3) — writes the intelligence brief from draft + findings. Uses the cheap perception tier when findings exist (they already carry the reasoning).
3. **Audit** (3) — checks claims against evidence and returns exact find/replace fixes; the server applies them instead of regenerating the document.

Report calls are `essential` (they bypass the USD/rate caps, not the billing breaker). Enrichment runs in a **worker thread** feeding an `asyncio.Queue`. Without an API key, `done` returns draft + analysis with `enriched: false`.

## Per-frame vision pipeline (`VisionEngine._loop`)

1. **Capture** BGR frame + `FrameMeta` (index, loop); optional downscale via `PROCESSING_MAX_WIDTH`. `_PlaybackClock` converts to analysis/media time.
2. **MediaPipe** (VIDEO mode, monotonic timestamps): pose, face (blendshapes + transformation matrix), hands when toggled or needed for expressions.
3. **YOLO** every `YOLO_STRIDE` frames → `sv.Detections.from_ultralytics` → ByteTrack.
4. **YUBI v3.0 reconcile** (`reconcile_tracked_objects`):
   - Merge GenAI semantic objects with local boxes.
   - Low-confidence YOLO labels below `GEMINI_VERIFY_BELOW` force **one** verification pass per tracker (`_verify_requested`); otherwise `_maybe_request_scene` asks the `SceneChangeGate`.
   - Confirmed labels stick on `tracker_id` for `GEMINI_LABEL_CACHE_SEC` without repeat API calls.
5. **Expressions** (if `show_expressions`): `ExpressionAnalyzer.process(face_result, frame, analysis_t, media_sec, hand_points)` → per-subject AUs, episodes, vitals; new episodes go to `InsightEngine.push_episodes` (with apex crops only when insights want images).
6. **Annotate** (order): objects → pose → **expressions OR face** → hands → v3.0 semantic boxes (only with Objects on).
7. **Alerts** if watchlist class in reconciled detections (cooldown + optional webhook).
8. **JPEG encode** → `_latest_jpeg`; update `VisionStats`; pace to 1× for files.

## YUBI v3.0 subsystem (`gemini_vision.py`)

User-facing name: **YUBI v3.0**. Implementation uses `google.genai` with settings from `GEMINI_*` env vars.

| Component | Role |
|-----------|------|
| `AIClient` (`ai_client.py`) | Single entry point for model calls: admission via `AIBudget`, JSON mode, per-family reasoning control (`thinking_budget` on 2.x, `thinkingLevel` on 3.x+), low media resolution for perception images |
| `AIBudget` (`ai_budget.py`) | Token ledger + cost estimates per feature, session USD cap, per-minute cap, error cooldowns; billing/quota errors pause all features (vendor-neutral message) |
| `SceneChangeGate` | 32×18 thumbnail diff + confident-class signature; unchanged views back off ×2 per refresh to `GEMINI_MAX_INTERVAL_SEC` |
| `GeminiEnricher` | Gated scene passes in a worker thread (one in flight); maintains `GeminiInsight` |
| `reconcile_tracked_objects` | Match v3.0 boxes to ByteTrack IDs; label aliases (e.g. dining table ↔ shelf) |
| `TrackLabelCache` | Sticky display labels per `tracker_id` |
| `collect_uncertain_hints` | Builds prompt hints for low-confidence local detections |
| `draw_gemini_boxes` | Optional overlay for v3.0 semantic boxes (aged out via `GEMINI_BOX_MAX_AGE_SEC`) |

States exposed to UI: `disabled`, `idle`, `thinking`, `ready`, `paused`, `error`.

## Expression engine (local, no API cost)

| Module | Role |
|--------|------|
| `facs.py` | 22 FACS action units from MediaPipe blendshapes (with L/R sides), EMFACS emotion prototypes (joy needs AU6+AU12 for Duchenne; contempt from unilateral AU12/AU14), cue library with meanings |
| `signal_filters.py` | One Euro filter (landmarks + AUs) — removes jitter without blunting 100–500 ms movements |
| `micro_expressions.py` | `ExpressionAnalyzer` — per-subject tracking, baselines, episodes, vitals, live status, session analysis |
| `pulse.py` | Experimental rPPG pulse (POS method) on forehead/cheek skin; hidden below confidence 0.3 |
| `expression_vision.py` | Draws `SubjectRender`: smoothed mesh, region heat, subject tag, micro flash |

**Episodes.** Each AU keeps an adaptive baseline + noise spread learned during `EXPRESSION_CALIBRATION_SEC`; an onset fires at z > 3 (scaled by `EXPRESSION_SENSITIVITY`) and is tracked through apex to offset with hysteresis. AU onsets within 200 ms form one cluster → **micro** (≤ 500 ms) or **macro** episode, matched to emotion prototypes and annotated with cues (Duchenne/social smile, unilateral smirk, masking, suppression, leakage, lip compression, …). Behavioral channels emit **behavior** episodes: blink bursts / long closures, gaze aversion, head nod/shake, self-touch (hand landmarks near the face).

**Speech gating.** Talking moves the lower face constantly, so while a subject is speaking (jaw-open oscillation): brief lower-face AU events are treated as articulation and dropped; lower-face context is excluded from emotion scoring; masking/suppression/leakage/lip-compression cues are withheld; brief brow raises or lowers with no emotion become `speech_emphasis` (a conversational baton, kept on the timeline but not "notable"); lower-face-only episodes with no emotion and no surviving cue are dropped as articulation. Clusters made only of speech-sensitive AUs wait 800 ms before finalizing so speech onset is detected first.

**Emotion attribution.** Already-active AUs count as context when scoring a cluster (a micro flash on top of a held smile), but an episode is only named after an emotion whose prototype includes at least one of the episode's own AUs — a lip pucker during a frown stays "Lip pucker", not "Anger".

**Notable** episodes (every micro, masking/leakage/suppression/smirk cues, strong negative emotion) wake the insight engine; everything else is still recorded for the timeline, export, and report.

## Insight engine (`insight_engine.py`)

Background worker that interprets locally measured episodes with the reasoning tier (`GEMINI_INSIGHT_MODEL`):

- Fires only when a notable episode is pending and `INSIGHT_MIN_INTERVAL_SEC` has passed, or as a digest every `INSIGHT_DIGEST_INTERVAL_SEC` when only non-notable episodes accumulated.
- Sends a compact text digest (episodes, subject summaries, scene summary, previous running summary) plus ≤ `INSIGHT_MAX_IMAGES` low-resolution apex face crops.
- Returns up to 3 insights (`title`, `observation`, `interpretation`, `category`, `confidence`, `at`, `evidence` episode ids) and an updated running summary — context accumulates without resending history.
- Prompts require hedged, evidence-cited language (no lie detection or diagnosis claims). On `AIUnavailable` the batch is re-queued and the state shows `paused`.

Frontend: the right-rail **micro-expression dock** expands (`expressions-live` on `#video-wrap`) with live affect bars, valence/activation timeline canvas (episode ticks), region meters, vitals (blink, gaze, head, pulse, speech, touch), action-unit chips, and the episode list. The YUBI v3.0 panel shows the spend meter, scene summary, and behavioral insights.

## Frontend architecture

| Region | DOM / behavior |
|--------|----------------|
| **Idle overlay** | Input source box + analysis profile picker (from `/api/health`; remembered in `localStorage`) |
| Top **command bar** | Brand, telemetry strip (incl. MEDIA position/speed for files, MODE profile), pipeline rail LEDs, status pill |
| Left **detections panel** | Pipeline layer toggles, confidence slider, object registry, intel feed (notable episodes, insights, AI pause/resume) |
| Right **rail** | YUBI v3.0 panel (spend meter, scene, insights); expression dock (when enabled) |
| Bottom **command footer** | Initialize / Terminate / Capture / Record |
| **HUD overlay** | Grid, brackets, crosshair on live feed |
| **Session complete** | Raw log (+ measured behavioral analysis), YUBI v3.0 report (dynamic pass list from `compile.steps`), timeline JSON export |
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
| `SOURCE_LOOP` | `true` | Loop files (Interview profile always plays once) |
| `SOURCE_REALTIME_PACING` | `true` | Hold files at 1× playback |
| `DEFAULT_PROFILE` | `general` | Profile when the request omits one |
| `PROCESSING_MAX_WIDTH` | `1280` | Downscale before inference |
| `STREAM_JPEG_QUALITY` | `80` | MJPEG encode quality |
| `LIVE_STREAM_FPS` | `20` | MJPEG publisher rate for YouTube/RTSP |
| `LIVE_INFERENCE_FPS` | `2.0` | Max inference rate for remote live |
| `LIVE_REMOTE_LITE` | `true` | Objects-only layers on remote live startup |

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
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Perception tier: scene, verify, identity (not shown in UI) |
| `GEMINI_INSIGHT_MODEL` | `gemini-3.8-flash` | Reasoning tier: insights, report analyze/audit |
| `GEMINI_MEDIA_RESOLUTION` | `low` | Image detail for perception calls |
| `GEMINI_INTERVAL_SEC` | `4` | Minimum seconds between scene passes |
| `GEMINI_MAX_INTERVAL_SEC` | `60` | Back-off ceiling for unchanged scenes |
| `GEMINI_SCENE_CHANGE_THRESHOLD` | `6` | Thumbnail mean pixel delta that counts as change |
| `GEMINI_MAX_OBJECTS` | `8` | Max semantic objects per pass |
| `GEMINI_ANALYSIS_SCALE` | `0.6` | JPEG scale for API |
| `GEMINI_VERIFY_BELOW` | `0.45` | Local conf below → v3.0 verify (once per track) |
| `GEMINI_LABEL_CACHE_SEC` | `45` | Sticky label TTL on tracker |
| `SESSION_REPORT_PASSES` | `2` | 1 compose · 2 analyze + compose · 3 + audit |
| `AI_SESSION_BUDGET_USD` | `1.0` | Estimated spend cap per session (0 = unlimited) |
| `AI_MAX_CALLS_PER_MIN` | `12` | Rate cap across features |
| `AI_REQUEST_TIMEOUT_SEC` | `30` | Per-call timeout |
| `INSIGHTS_ENABLED` | `true` | Event-driven behavioral interpretation |
| `INSIGHT_MIN_INTERVAL_SEC` | `20` | Minimum gap between insight calls |
| `INSIGHT_DIGEST_INTERVAL_SEC` | `90` | Digest cadence without notable episodes |
| `INSIGHT_MAX_IMAGES` | `2` | Apex face crops per call |
| `INSIGHT_THINKING` | `low` | Reasoning depth |

### Expression engine (local)

| Variable | Default | Notes |
|----------|---------|-------|
| `EXPRESSION_CALIBRATION_SEC` | `4` | Face time used to learn each subject's baseline |
| `EXPRESSION_SENSITIVITY` | `1.0` | >1 subtler movements, <1 only stronger |
| `EXPRESSION_PULSE_ENABLED` | `true` | Experimental rPPG pulse |

## Threading & locks

| Thread | Work |
|--------|------|
| FastAPI event loop | HTTP, MJPEG read of buffer, stream report queue consumer |
| `_bootstrap` | One-time init |
| `_loop` | Per-frame capture + inference + encode |
| `GeminiEnricher` / `IdentityEnricher` workers | Gated GenAI calls (one in flight each) |
| `InsightEngine` worker | Event-driven interpretation (wakes on notable episodes) |
| Session report worker | Multi-pass markdown enrichment |

`ExpressionAnalyzer`, `InsightEngine`, and `AIBudget` each guard their state with their own lock; `/api/analysis` reads run in `asyncio.to_thread`.

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
| `tests/test_api.py` | Health (profiles, report steps), config, lifecycle, analysis endpoints, no model ids in status |
| `tests/test_gemini_reconcile.py` | Label reconciliation + track cache |
| `tests/test_session_report.py` | Report enrichment offline + stream NDJSON |
| `tests/test_ai_budget.py` | Pricing, caps, essential bypass, 402 breaker, thinking/media settings, JSON parsing |
| `tests/test_signal_processing.py` | One Euro filter, pulse estimate, FACS mapping, emotion prototypes, cue detection |
| `tests/test_micro_expressions.py` | Synthetic-face episodes: micro vs macro, smirk side, blinks, neutral noise, speech gating + batons, emotion attribution |
| `tests/test_intelligence.py` | Scene gate back-off, insight engine triggering, analysis markdown, report pipeline (fake AI) |
| `tests/test_expression_vision.py` | Overlay drawing |

Run `node --check frontend/app.js` after editing `app.js` — a top-level syntax error prevents the entire script from loading (Initialize button dead).
