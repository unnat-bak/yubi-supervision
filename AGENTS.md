# YUBI Supervision — Agent Instructions

Full-screen **command-center** live webcam vision: YOLOv8 + ByteTrack, MediaPipe keypoints (pose/face/hands), **micro-expression tracking**, and **YUBI v3.0** semantic intelligence — drawn with [supervision](https://github.com/roboflow/supervision) annotators. FastAPI backend + vanilla frontend (no build step).

**Repo:** https://github.com/unnat-bak/yubi-supervision

## Product surface (what users see)

| Capability | UI / behavior |
|------------|----------------|
| **Input source** | Idle-screen **Input source** box: webcam, local video file, RTSP/CCTV stream, or YouTube URL (resolved via yt-dlp); files loop (except Interview), live feeds stay latest; SRC shown in command HUD; files paced at 1× with MEDIA position in the HUD |
| **Analysis profile** | Idle-screen picker: **General** (objects/skeleton/face/hands), **Interview** (expressions + insights; files play once, session ends with a report), **Security** (objects + identity) |
| **Objects** | Corner boxes, labels, tracker IDs; object registry + confidence gate in left sidebar |
| **Segmentation masks** | Per-instance colored masks when a YOLO `*-seg` model is configured; **Masks** layer chip |
| **Pose / face / hands** | Toggleable skeleton overlays (muted sage-teal palette); **Joint tags** chip labels curated pose keypoints |
| **Identity** | Per-tracked-person name / jersey number / descriptor via YUBI v3.0, cached on `tracker_id`; **Identity** layer chip (needs API key) |
| **Expressions** | Local FACS engine (22 action units, per-subject baselines, micro ≤ 500 ms vs macro episodes, EMFACS emotions, cue library, speech gating) + blink / gaze / head / self-touch / experimental pulse. Replaces face overlay with smoothed mesh + region heat; right **micro-expression dock** shows affect bars, valence/activation timeline, AU chips, vitals, episode list |
| **Behavioral insights** | YUBI v3.0 interprets locally detected notable episodes (text digest + ≤ 2 apex crops), hedged and evidence-cited; running summary in the YUBI v3.0 panel |
| **AI spend** | Spend meter in the YUBI v3.0 panel (calls, tokens, est. cost vs `AI_SESSION_BUDGET_USD`, calls avoided by gating); billing/quota errors pause cleanly with a neutral reason |
| **YUBI v3.0** | Change-gated scene summary + object list in right panel; overlay boxes with Objects on; one verify/correct pass per uncertain YOLO track |
| **Command HUD** | Top bar: UTC, session ID, frame index, uptime, source; pipeline rail (OBS/SKEL/FACE/HAND/EXPR/MASK/ID/V3.0); FPS/latency sparkline |
| **Intel feed** | Chronological event log in left sidebar (system, v3.0, objects, expressions, alerts) |
| **Alerts** | Watchlist classes → banner + optional webhook |
| **Capture / record** | PNG snapshot (+ JSON bundle), annotated MP4 to `recordings/` |
| **Session report** | On terminate / EOF: raw markdown log + measured **Behavioral analysis** section; optional YUBI v3.0 brief (analyze → compose → optional evidence audit) with loading overlay; **Export timeline (JSON)** of every episode |

**Branding:** User-facing strings, exports, and prompts say **YUBI v3.0** / **YUBI Supervision** — never vendor model names. Internal code/env may still use `gemini_*` and `/api/gemini` (legacy API path).

## Stack

| Layer | Technology |
|-------|------------|
| API | FastAPI, Uvicorn |
| Vision | OpenCV, MediaPipe Tasks, Ultralytics YOLOv8, supervision |
| Intelligence | YUBI v3.0 (`backend/gemini_vision.py`, `backend/session_report.py`) — Google GenAI client under the hood |
| Frontend | HTML, CSS, vanilla JS |
| Config | pydantic-settings (`.env`) |

Python **3.9+**. Package manager: **pip** + `requirements.txt` (see `pyproject.toml` for tooling).

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"          # optional: ruff, pytest, httpx

./scripts/dev.sh                   # start dev server
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

curl http://127.0.0.1:8000/api/health
curl -X POST http://127.0.0.1:8000/api/start
curl http://127.0.0.1:8000/api/status
curl -X POST http://127.0.0.1:8000/api/stop

ruff check backend tests
pytest
node --check frontend/app.js     # catch JS syntax errors before they break Initialize
```

Open http://127.0.0.1:8000 — click **Initialize**. Keyboard: `Space` start · `S` capture · `Q` / `Esc` terminate.

## Directory layout

```
backend/
  main.py              # FastAPI routes, MJPEG stream (thin layer)
  config.py            # Settings from environment
  schemas.py           # Pydantic request/response models
  vision.py            # VisionEngine — capture, inference, annotation orchestration
  sources.py           # Input source resolution (webcam / file / RTSP / YouTube via yt-dlp)
  profiles.py          # Analysis profiles (general / interview / security)
  ai_budget.py         # Spend governor: token ledger, cost estimates, caps, circuit breaker
  ai_client.py         # Shared YUBI v3.0 client: admission, JSON mode, reasoning/image detail per model
  gemini_vision.py     # YUBI v3.0 scene enricher + SceneChangeGate, reconciliation, track label cache
  identity_vision.py   # Per-track identity (name/jersey/descriptor) enricher + cache
  facs.py              # FACS action units, EMFACS emotion prototypes, cue library (pure)
  signal_filters.py    # One Euro filter
  pulse.py             # Experimental rPPG pulse (POS)
  micro_expressions.py # ExpressionAnalyzer — per-subject episodes, vitals, session analysis
  insight_engine.py    # Event-driven YUBI v3.0 interpretation of notable episodes
  expression_vision.py # Expression overlay drawing (mesh, region heat, tags, micro flash)
  session_report.py    # Behavioral analysis section + analyze / compose / audit passes
  models/              # Auto-downloaded MediaPipe .task files (gitignored)
frontend/
  index.html           # Command-center shell, sidebars, overlays
  styles.css           # Intel-panel HUD aesthetic
  app.js               # Polling, layer toggles, session log, report download stream
docs/
  ARCHITECTURE.md      # System design (read before large changes)
  ROADMAP.md           # Production evolution plan
tests/
  test_api.py
  test_gemini_reconcile.py
  test_session_report.py
  test_ai_budget.py
  test_intelligence.py
  test_micro_expressions.py
  test_signal_processing.py
  test_expression_vision.py
scripts/
  dev.sh
  smoke.sh
```

## Architecture (summary)

- **Async bootstrap:** `POST /api/start` returns `{state: "starting"}` immediately; model/camera init runs in `_bootstrap` thread. Frontend polls `GET /api/status` until `live` or `error`.
- **Processing thread (`_loop`):** Capture (+ media time for files) → MediaPipe + YOLO (stride) + ByteTrack → YUBI v3.0 reconcile labels → `ExpressionAnalyzer` → annotate → JPEG buffer → 1× pacing for files.
- **MJPEG stream:** `GET /api/stream` reads `_latest_jpeg`; must not block on inference.
- **AI cost model:** every model call goes through `AIClient` → `AIBudget` (session USD cap, per-minute cap, billing breaker). Never stream frames to the model; call it on change (scene gate), once per uncertain track, or on notable local evidence (insights).
- **YUBI v3.0 scene:** `SceneChangeGate` decides when a pass runs; results merged with local detections via `reconcile_tracked_objects`; sticky labels on `tracker_id`.
- **Expressions:** all measurement is local (`facs.py`, `micro_expressions.py`); overlay in `expression_vision.py`; notable episodes feed `InsightEngine`. Session analysis is retained after stop for `/api/analysis*` and the report.
- **Session report:** Client builds draft markdown locally → `POST /api/session-report/stream` (NDJSON progress) → server inserts measured analysis → analyze / compose / audit passes (`SESSION_REPORT_PASSES`) → download `.md`.
- **Webcam is server-side:** OpenCV on the machine running Python — **not** in the browser. On macOS, grant camera access to Terminal (or the uvicorn process).

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for API contracts, threading, reconciliation, and UI layout.

## Where to add code

| Change | Location |
|--------|----------|
| New API route | `backend/main.py` + schema in `backend/schemas.py` |
| Env / defaults | `backend/config.py` + `.env.example` |
| Input source (file / stream / YouTube) | `backend/sources.py` |
| Identity / jersey labels | `backend/identity_vision.py` |
| Object / pose / capture loop | `backend/vision.py` |
| YUBI v3.0 scene / boxes / reconcile | `backend/gemini_vision.py` |
| Any new model call | `AIClient.generate(feature=...)` in `backend/ai_client.py` (add pricing in `ai_budget.py` for new models) |
| Action units / emotions / cues | `backend/facs.py` |
| Episodes / vitals / speech gating | `backend/micro_expressions.py` |
| Behavioral interpretation | `backend/insight_engine.py` |
| Expression overlay drawing | `backend/expression_vision.py` |
| Analysis profiles | `backend/profiles.py` |
| Session report passes | `backend/session_report.py` |
| UI / polling / session log | `frontend/app.js`, `frontend/styles.css`, `frontend/index.html` |
| Agent context | this file (canonical) |

## Coding standards

- Keep `backend/main.py` thin — routes only; logic stays in vision modules.
- Never block the FastAPI event loop with `VideoCapture`, model load, or GenAI calls — use `VisionEngine` threads or `asyncio.to_thread` / worker threads for streams.
- Use **supervision** annotators — do not reinvent drawing.
- Match existing style: type hints, `from __future__ import annotations`, minimal comments.
- Frontend: no frameworks unless explicitly requested; preserve command-center / intel-panel aesthetic.
- Layer toggles in UI must **revert on failed** `POST /api/config` (`pushConfig` returns `null` on failure).
- Download/report flows must use `try/finally` so buttons and busy flags always reset.

## Testing

- API tests: `tests/test_api.py` (lifecycle smoke, analysis endpoints).
- Reconciliation: `tests/test_gemini_reconcile.py`.
- Session report offline path: `tests/test_session_report.py` (patch `settings` when API key present in env).
- AI spend governor: `tests/test_ai_budget.py`; scene gate / insights / report pipeline with a fake AI: `tests/test_intelligence.py` — never hit the real API in tests.
- Expression engine: `tests/test_micro_expressions.py` (synthetic face), `tests/test_signal_processing.py` (filters, pulse, FACS).
- Do not commit model weights (`*.pt`, `*.task`) or `.env`.
- Manual smoke: initialize → live + fps → toggle layers in **left sidebar** → expressions dock expands → terminate → download session report. For expressions, use the **Interview** profile with a talking-head video file (plays once, then the session-complete card appears).

## Prohibitions

- Do **not** commit `.venv/`, `*.pt`, `backend/models/*.task`, or secrets.
- Do **not** block the FastAPI event loop with synchronous model load or `VideoCapture`.
- Do **not** mention Gemini/Google in user-visible strings or exported markdown (YUBI v3.0 only).
- Do **not** add auth, database, or multi-page UI unless explicitly requested.
- Do **not** force-push to `main`.

## Production direction

See [docs/ROADMAP.md](docs/ROADMAP.md). Intended evolution: split `vision.py` into packages, structured logging, WebRTC option, Docker, CI deploy — without breaking the current demo.
