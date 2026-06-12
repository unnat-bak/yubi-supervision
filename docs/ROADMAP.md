# Roadmap — YUBI Supervision

Production evolution plan. Items are ordered; earlier phases unblock later ones.

## Phase 1 — Foundation (current)

- [x] FastAPI + MJPEG + async startup
- [x] AGENTS.md / CLAUDE.md / docs / config layer
- [x] CI: ruff + pytest smoke tests
- [ ] Fix startup reliability on all target machines (camera timeout, clear UI errors)

## Phase 2 — Reliability

- Structured logging with request IDs
- `/api/health` deep checks (camera available, models cached)
- Integration test script (`scripts/smoke.sh`)
- Graceful shutdown and restart without process kill
- Config flag: `yolov8n` vs `yolov8s` for speed/quality tradeoff

## Phase 3 — Code structure

- Split `vision.py` into `backend/vision/` package (see ARCHITECTURE.md)
- Pydantic response models for all API endpoints
- OpenAPI tags and versioned API (`/api/v1/`)

## Phase 4 — Product features

- Line/zone counting (supervision `LineZone`, `PolygonZone`)
- Clip recording (annotated MP4 export)
- Class-based alerts (webhook when `person` detected)
- Snapshot + JSON export

## Phase 5 — Production ops

- Docker + docker-compose
- Prometheus metrics (fps, inference ms, frame drops)
- Optional WebRTC low-latency stream
- Auth (API key or OAuth) if exposed beyond localhost
- Frontend build step (optional Vite) only if complexity warrants it

## Non-goals (unless requested)

- Multi-tenant SaaS
- Custom model training UI
- Mobile native apps
