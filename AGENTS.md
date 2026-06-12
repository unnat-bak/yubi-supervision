# YUBI Supervision — Agent Instructions

Full-screen live webcam vision: YOLOv8 object detection, MediaPipe keypoints (pose/face/hands), and [supervision](https://github.com/roboflow/supervision) annotators. FastAPI backend + vanilla frontend.

**Repo:** https://github.com/unnat-bak/yubi-supervision

## Stack

| Layer | Technology |
|-------|------------|
| API | FastAPI, Uvicorn |
| Vision | OpenCV, MediaPipe Tasks, Ultralytics YOLOv8, supervision |
| Frontend | HTML, CSS, vanilla JS (no build step) |
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
```

Open http://127.0.0.1:8000 — click **Start Vision**. Keyboard: `Space` start, `Q` / `Esc` stop.

## Directory layout

```
backend/
  main.py           # FastAPI routes, MJPEG stream (thin layer)
  config.py         # Settings from environment
  schemas.py        # Pydantic request/response models
  vision.py         # VisionEngine — capture, inference, annotation (split later)
  models/           # Auto-downloaded MediaPipe .task files (gitignored)
frontend/
  index.html        # Shell + overlays
  styles.css        # Glass-morphism UI
  app.js            # Start/stop, polling, layer toggles
docs/
  ARCHITECTURE.md   # System design (read before large changes)
  ROADMAP.md        # Production evolution plan
tests/
scripts/
  dev.sh
```

## Architecture (summary)

- **Async bootstrap:** `POST /api/start` returns `{state: "starting"}` immediately; model/camera init runs in a background thread. Frontend polls `GET /api/status` until `live` or `error`.
- **Processing thread:** Reads webcam → MediaPipe (pose/face/hands) + YOLO + ByteTrack → supervision annotators → JPEG buffer.
- **MJPEG stream:** `GET /api/stream` reads latest JPEG; must not block on inference.
- **Webcam is server-side:** OpenCV captures on the machine running Python — **not** in the browser. On macOS, grant camera access to Terminal (or the process running uvicorn).

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for API contracts, threading, and scaling path.

## Coding standards

- Keep `backend/main.py` thin — routes only; logic stays in `vision.py` or future `backend/vision/` package.
- Use **supervision** annotators (`EdgeAnnotator`, `BoxCornerAnnotator`, `KeyPoints.from_mediapipe`, `Detections.from_ultralytics`) — do not reinvent drawing.
- Match existing style: type hints, `from __future__ import annotations`, minimal comments.
- Frontend: no frameworks unless explicitly requested; preserve glass full-screen aesthetic.
- Prefer env-based config (`backend/config.py`) over hardcoded constants.

## Testing

- Add API tests in `tests/` using `fastapi.testclient.TestClient`.
- Do not commit model weights (`*.pt`, `*.task`) or `.env`.
- Manual smoke: start → status shows `live` + fps → stream renders → stop returns `idle`.

## Prohibitions

- Do **not** commit `.venv/`, `*.pt`, `backend/models/*.task`, or secrets.
- Do **not** block the FastAPI event loop with synchronous model load or `VideoCapture` — use background threads.
- Do **not** add auth, database, or multi-page UI unless explicitly requested.
- Do **not** force-push to `main`.

## Production direction

See [docs/ROADMAP.md](docs/ROADMAP.md). Intended evolution: split `vision.py` into packages, structured logging, WebRTC option, Docker, CI deploy — without breaking the current minimal demo.
