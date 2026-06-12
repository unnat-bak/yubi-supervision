# Backend — YUBI Supervision

Parent context: `@../AGENTS.md`

## Module roles

- `main.py` — HTTP routes only; single shared `VisionEngine` instance.
- `config.py` — `Settings` from environment; import `get_settings()` not raw constants.
- `schemas.py` — Pydantic models for API payloads.
- `vision.py` — Threading, MediaPipe, YOLO, supervision annotators. **Do not import FastAPI here.**

## Threading rules

- `_bootstrap` thread: one-time camera + model init.
- `_loop` thread: per-frame capture and inference.
- Never call `VideoCapture.read()` or YOLO inference on the asyncio event loop.
- Use `self._lock` when reading/writing `_stats`, `_latest_jpeg`, `_running`, `_starting`.

## Adding a detector

1. Extend `VisionConfig` / `Settings` for toggles and thresholds.
2. Init model in `_bootstrap`; release in `stop()` / `_release_models`.
3. Annotate in `_loop` after objects, before or after keypoints as appropriate.
4. Expose counts via `VisionStats` and `/api/status`.
