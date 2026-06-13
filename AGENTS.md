# YUBI Supervision — Agent Instructions

Full-screen **command-center** live webcam vision: YOLOv8 + ByteTrack, MediaPipe keypoints (pose/face/hands), **micro-expression tracking**, and **YUBI v3.0** semantic intelligence — drawn with [supervision](https://github.com/roboflow/supervision) annotators. FastAPI backend + vanilla frontend (no build step).

**Repo:** https://github.com/unnat-bak/yubi-supervision

## Product surface (what users see)

| Capability | UI / behavior |
|------------|----------------|
| **Objects** | Corner boxes, labels, tracker IDs; object registry + confidence gate in left sidebar |
| **Pose / face / hands** | Toggleable skeleton overlays (muted sage-teal palette) |
| **Expressions** | Replaces face overlay: facial mesh + live tracking dots; expanded right **micro-expression dock** (region meters, live signals) |
| **YUBI v3.0** | Semantic scene summary + object list in right panel; optional overlay boxes; label verify/correct for uncertain YOLO hits |
| **Command HUD** | Top bar: UTC, session ID, frame index, uptime; pipeline rail (OBS/SKEL/FACE/HAND/EXPR/V3.0); FPS/latency sparkline |
| **Intel feed** | Chronological event log in left sidebar (system, v3.0, objects, expressions, alerts) |
| **Alerts** | Watchlist classes → banner + optional webhook |
| **Capture / record** | PNG snapshot (+ JSON bundle), annotated MP4 to `recordings/` |
| **Session report** | On terminate: download markdown chronology; optional **3-pass YUBI v3.0 enrichment** with loading overlay |

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
  gemini_vision.py     # YUBI v3.0 enricher, label reconciliation, track label cache
  expression_vision.py # Micro-expression tracker, mesh overlay, face-structure guidance
  session_report.py    # Multi-pass post-session markdown enrichment (YUBI v3.0)
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
scripts/
  dev.sh
  smoke.sh
```

## Architecture (summary)

- **Async bootstrap:** `POST /api/start` returns `{state: "starting"}` immediately; model/camera init runs in `_bootstrap` thread. Frontend polls `GET /api/status` until `live` or `error`.
- **Processing thread (`_loop`):** Capture → MediaPipe + YOLO (stride) + ByteTrack → YUBI v3.0 reconcile labels → micro-expression blendshapes → annotate → JPEG buffer.
- **MJPEG stream:** `GET /api/stream` reads `_latest_jpeg`; must not block on inference.
- **YUBI v3.0:** Periodic scene analysis in background thread; results merged with local detections via `reconcile_tracked_objects`; sticky labels on `tracker_id`.
- **Expressions:** When `show_expressions` is on, face skeleton is replaced by expression mesh + dots; `MicroExpressionTracker` detects blendshape deltas; `ExpressionEnricher` sends cropped face JPEGs to v3.0 for structure boxes.
- **Session report:** Client builds draft markdown locally → `POST /api/session-report/stream` (NDJSON progress) → 1–3 YUBI v3.0 passes → download enriched `.md`.
- **Webcam is server-side:** OpenCV on the machine running Python — **not** in the browser. On macOS, grant camera access to Terminal (or the uvicorn process).

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for API contracts, threading, reconciliation, and UI layout.

## Where to add code

| Change | Location |
|--------|----------|
| New API route | `backend/main.py` + schema in `backend/schemas.py` |
| Env / defaults | `backend/config.py` + `.env.example` |
| Object / pose / capture loop | `backend/vision.py` |
| YUBI v3.0 scene / boxes / reconcile | `backend/gemini_vision.py` |
| Expression overlay / micro-signals | `backend/expression_vision.py` |
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

- API tests: `tests/test_api.py` (lifecycle smoke).
- Reconciliation: `tests/test_gemini_reconcile.py`.
- Session report offline path: `tests/test_session_report.py` (patch `settings` when API key present in env).
- Do not commit model weights (`*.pt`, `*.task`) or `.env`.
- Manual smoke: initialize → live + fps → toggle layers in **left sidebar** → expressions dock expands → terminate → download session report.

## Prohibitions

- Do **not** commit `.venv/`, `*.pt`, `backend/models/*.task`, or secrets.
- Do **not** block the FastAPI event loop with synchronous model load or `VideoCapture`.
- Do **not** mention Gemini/Google in user-visible strings or exported markdown (YUBI v3.0 only).
- Do **not** add auth, database, or multi-page UI unless explicitly requested.
- Do **not** force-push to `main`.

## Production direction

See [docs/ROADMAP.md](docs/ROADMAP.md). Intended evolution: split `vision.py` into packages, structured logging, WebRTC option, Docker, CI deploy — without breaking the current demo.
