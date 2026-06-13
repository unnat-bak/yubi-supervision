# Backend — YUBI Supervision

Parent context: `@../AGENTS.md` · deep design: `@../docs/ARCHITECTURE.md`

## Module map

| Module | Responsibility |
|--------|----------------|
| `main.py` | HTTP routes only; shared `VisionEngine`; MJPEG + session-report stream |
| `config.py` | `Settings` from `.env`; `get_settings()` cached |
| `schemas.py` | Pydantic API models |
| `vision.py` | `VisionEngine`: threads, capture, YOLO, MediaPipe, annotate, alerts, recording, session stats |
| `gemini_vision.py` | **YUBI v3.0**: `GeminiEnricher`, reconciliation, `TrackLabelCache`, semantic boxes |
| `expression_vision.py` | Micro-expression tracker, mesh overlay, `ExpressionEnricher` (face-structure guidance) |
| `session_report.py` | Multi-pass markdown enrichment (YUBI v3.0); NDJSON progress events |

Do **not** import FastAPI in vision modules.

## Threading rules

| Thread | Owner |
|--------|--------|
| `_bootstrap` | One-time camera + model init |
| `_loop` | Per-frame capture, inference, JPEG encode |
| `GeminiEnricher` | Periodic scene GenAI calls |
| `ExpressionEnricher` | Periodic face-crop GenAI calls |
| Session report worker | `enrich_session_report_events` (invoked from `main.py` stream) |

- Never call `VideoCapture.read()`, YOLO, or GenAI on the asyncio event loop.
- Use `self._lock` for `_stats`, `_latest_jpeg`, `_running`, `_starting`, `_config`, record requests.
- Stream endpoints: worker thread + `asyncio.Queue` or `asyncio.to_thread` for blocking enrichment.

## Vision pipeline order (annotation)

1. Objects (reconciled labels from v3.0 cache)
2. Pose edges
3. **Expressions** (`draw_expression_overlay`) **or** face edges/vertices
4. Hand edges/vertices
5. YUBI v3.0 semantic boxes (if enabled)

`show_expressions` disables the standard face overlay path.

## Adding a detector or overlay

1. Extend `VisionConfig` in `vision.py` + `schemas.ConfigUpdate` + frontend layer chip.
2. Init model in `_bootstrap`; release in `stop()` / `_release_models`.
3. Run inference in `_loop`; expose counts via `VisionStats` and `/api/status`.
4. Add to `degraded` list on init failure.

## Adding YUBI v3.0 behavior

- Scene / boxes: `gemini_vision.py` (`GeminiEnricher`, prompts, `reconcile_tracked_objects`).
- Label stickiness: `TrackLabelCache` + `GEMINI_LABEL_CACHE_SEC`.
- Low-confidence verify: `GEMINI_VERIFY_BELOW`, `collect_uncertain_hints`.
- Post-session report: `session_report.py` (`PASS_PROMPTS`, `SESSION_REPORT_PASSES`).
- User-facing strings: **YUBI v3.0** only — never Gemini/Google in exports or UI copy.

## Adding expression behavior

- Blendshape deltas: `MicroExpressionTracker` in `expression_vision.py`.
- Drawing: `draw_expression_overlay` (mesh at low alpha, dots at full opacity).
- Structure guidance: `ExpressionEnricher` + `refine_landmarks_with_guidance`.
- Status payload: `VisionEngine.get_expression_status()`.

## Tests

| File | Focus |
|------|--------|
| `tests/test_api.py` | HTTP lifecycle |
| `tests/test_gemini_reconcile.py` | Reconciliation + cache |
| `tests/test_session_report.py` | Report offline/stream (patch `settings` if env has API key) |
