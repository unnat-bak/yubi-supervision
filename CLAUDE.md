@AGENTS.md

# Claude Code — YUBI Supervision

## Recommended model

Use **`/model fable`** (Claude Fable 5) for multi-file vision pipeline work, debugging startup, and UI verification. Requires Claude Code v2.1.170+.

## Session workflow

1. Read `docs/ARCHITECTURE.md` before structural changes.
2. Run `./scripts/dev.sh` or `uvicorn backend.main:app --reload`.
3. Verify with curl: `/api/health` → `POST /api/start` → poll `/api/status` → `POST /api/stop`.
4. For UI changes, open http://127.0.0.1:8000 and confirm Start Vision completes (not stuck on loading overlay).

## Known gotchas

- **Camera permission** is for the **Python/Terminal process**, not the browser.
- **Startup hang** usually means blocking init on the event loop — init must stay in `VisionEngine._bootstrap` thread.
- **Hand landmarks** are not covered by `KeyPoints.from_mediapipe` — use the custom helper in `vision.py`.
- Models download on first run (~30s cold start): MediaPipe `.task` → `backend/models/`, YOLO → `yolov8s.pt` in project root.

## Where to add code

| Change | Location |
|--------|----------|
| New API route | `backend/main.py` + schema in `backend/schemas.py` |
| Env / defaults | `backend/config.py` + `.env.example` |
| Inference / drawing | `backend/vision.py` (later: `backend/vision/`) |
| UI / polling | `frontend/app.js`, `frontend/styles.css` |
| Agent context | `AGENTS.md` (canonical), this file for Claude-only notes |

## Progressive disclosure

- Deep architecture → `docs/ARCHITECTURE.md`
- Production plan → `docs/ROADMAP.md`
- Backend-only rules → `backend/CLAUDE.md`
