@AGENTS.md

# Claude Code — YUBI Supervision

## Recommended model

Use **`/model fable`** (Claude Fable 5) for multi-file vision pipeline work, debugging startup, and UI verification. Requires Claude Code v2.1.170+.

## Session workflow

1. Read `docs/ARCHITECTURE.md` before structural changes.
2. Run `./scripts/dev.sh` or `uvicorn backend.main:app --reload`.
3. Verify with curl: `/api/health` → `POST /api/start` → poll `/api/status` → `POST /api/stop`.
4. For UI changes: open http://127.0.0.1:8000 → **Initialize** (not stuck on loading overlay).
5. After editing `frontend/app.js`, run `node --check frontend/app.js` — syntax errors break the entire script (Initialize dead).

## Known gotchas

- **Camera permission** is for the **Python/Terminal process**, not the browser.
- **Startup hang** = blocking init on the event loop — init must stay in `VisionEngine._bootstrap` thread.
- **Hand landmarks** are not covered by `KeyPoints.from_mediapipe` — custom helper in `vision.py`.
- **Expressions vs face** — mutually exclusive overlay paths when `show_expressions` is on.
- **YUBI v3.0** uses `GEMINI_*` env vars and `/api/gemini` internally; UI/export say **YUBI v3.0** only.
- **Layer toggles** are in the **left sidebar** (`#layer-toggles` inside detections panel).
- **Session report** uses `POST /api/session-report/stream` (NDJSON); worker thread in `main.py`.
- **All AI calls** go through `AIClient.generate` (`backend/ai_client.py`) so `AIBudget` can cap spend and pause on depleted credits — never call the GenAI SDK directly. Catch `AIUnavailable`.
- **Expressions** come from `backend/micro_expressions.py` (FACS engine) — overlay only draws in `expression_vision.py`. Brow movements during speech are `speech_emphasis`, not emotions; an episode's emotion must involve its own AUs.
- **Interview profile** plays files once and stops at EOF; other profiles loop files.
- **Overlay text is ASCII-only** (OpenCV Hershey fonts) — `·` / `—` render as `??`; use ` | ` and `-`.
- **Editing backend `.py` reloads uvicorn** and ends any live session — finish live smoke tests before backend edits.
- Models download on first run: MediaPipe `.task` → `backend/models/`; YOLO → `yolov8s.pt` in project root.

## Where to add code

| Change | Location |
|--------|----------|
| New API route | `backend/main.py` + `backend/schemas.py` |
| Env / defaults | `backend/config.py` + `.env.example` |
| Capture / YOLO / loop | `backend/vision.py` |
| YUBI v3.0 scene / reconcile | `backend/gemini_vision.py` |
| Expression engine (AUs, episodes, cues) | `backend/facs.py`, `backend/micro_expressions.py` |
| Expression overlay | `backend/expression_vision.py` |
| Behavioral insights | `backend/insight_engine.py` |
| AI spend / caps | `backend/ai_budget.py`, `backend/ai_client.py` |
| Analysis profiles | `backend/profiles.py` |
| Session report passes | `backend/session_report.py` |
| UI / polling / session log | `frontend/app.js`, `styles.css`, `index.html` |
| Agent context | `AGENTS.md` (canonical), this file for Claude-only notes |

## Progressive disclosure

- Deep architecture → `docs/ARCHITECTURE.md`
- Production plan → `docs/ROADMAP.md`
- Backend-only rules → `backend/CLAUDE.md`
