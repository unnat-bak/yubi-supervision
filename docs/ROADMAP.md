# Roadmap — YUBI Supervision

Production evolution plan. Items are ordered; earlier phases unblock later ones.

## Phase 1 — Foundation

- [x] FastAPI + MJPEG + async startup
- [x] AGENTS.md / CLAUDE.md / docs / config layer
- [x] CI: ruff + pytest smoke tests
- [x] Command-center HUD (telemetry, pipeline rail, intel feed)
- [x] Graceful degradation per subsystem (`degraded` on status)
- [x] Session telemetry (`session_id`, `frame_index`, `uptime_sec`)
- [ ] Fix startup reliability on all target machines (camera timeout, clear UI errors)

## Phase 2 — Reliability

- Structured logging with request IDs
- `/api/health` deep checks (camera available, models cached, v3.0 reachable)
- Integration test script (`scripts/smoke.sh`) expanded for expressions + report
- Graceful shutdown and restart without process kill
- Frontend error boundaries (ensure `app.js` syntax check in CI)

## Phase 3 — Code structure

- Split `vision.py` into `backend/vision/` package (see ARCHITECTURE.md)
- Rename internal `gemini_*` API paths to `v3` while keeping aliases
- Pydantic response models for all endpoints
- OpenAPI tags and versioned API (`/api/v1/`)

## Phase 4 — Product features

- [x] Clip recording (annotated MP4 export)
- [x] Class-based alerts (webhook when watchlist class detected)
- [x] Snapshot + JSON export
- [x] YUBI v3.0 semantic layer (scene summary, boxes, label reconciliation)
- [x] Micro-expression mode (blendshapes + facial mesh overlay)
- [x] Post-session markdown report with multi-pass v3.0 enrichment
- Line/zone counting (supervision `LineZone`, `PolygonZone`)
- Expression history timeline export (beyond current session log)

## Phase 5 — Production ops

- Docker + docker-compose
- Prometheus metrics (fps, inference ms, frame drops, v3.0 latency)
- Optional WebRTC low-latency stream
- Auth (API key or OAuth) if exposed beyond localhost
- Frontend build step (optional Vite) only if complexity warrants it

## Non-goals (unless requested)

- Multi-tenant SaaS
- Custom model training UI
- Mobile native apps
