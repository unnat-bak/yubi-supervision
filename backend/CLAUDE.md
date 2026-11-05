# Backend — YUBI Supervision

Parent context: `@../AGENTS.md` · deep design: `@../docs/ARCHITECTURE.md`

## Module map

| Module | Responsibility |
|--------|----------------|
| `main.py` | HTTP routes only; shared `VisionEngine`; MJPEG + session-report stream |
| `config.py` | `Settings` from `.env`; `get_settings()` cached |
| `schemas.py` | Pydantic API models |
| `vision.py` | `VisionEngine`: threads, capture, YOLO, MediaPipe, annotate, alerts, recording, session stats |
| `profiles.py` | `AnalysisProfile` presets (`general` / `interview` / `security`) |
| `ai_budget.py` | `AIBudget`: pricing table, per-feature ledger, USD + rate caps, billing circuit breaker |
| `ai_client.py` | `AIClient.generate`: the only way to call the model (admission, JSON, thinking, media resolution) |
| `gemini_vision.py` | **YUBI v3.0**: `GeminiEnricher` + `SceneChangeGate`, reconciliation, `TrackLabelCache`, semantic boxes |
| `identity_vision.py` | Per-track identity enricher (skips when hints are fresh or budget blocks) |
| `facs.py` | Pure FACS mapping: action units, emotion prototypes, cue library |
| `signal_filters.py` / `pulse.py` | One Euro filter / experimental rPPG pulse |
| `micro_expressions.py` | `ExpressionAnalyzer`: subjects, baselines, episodes, speech gating, vitals, analysis |
| `insight_engine.py` | `InsightEngine`: event-driven interpretation of notable episodes |
| `expression_vision.py` | `draw_expression_overlay` only (no analysis, no API calls) |
| `session_report.py` | Analysis markdown + analyze / compose / audit passes; NDJSON progress events |

Do **not** import FastAPI in vision modules.

## Threading rules

| Thread | Owner |
|--------|--------|
| `_bootstrap` | One-time camera + model init |
| `_loop` | Per-frame capture, inference, JPEG encode |
| `GeminiEnricher` / `IdentityEnricher` | Gated GenAI calls, one in flight each |
| `InsightEngine` worker | Wakes on notable episodes; min interval + digest cadence |
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
- Post-session report: `session_report.py` (`ANALYZE_PROMPT` / `COMPOSE_PROMPT` / `AUDIT_PROMPT`, `pass_plan`, `SESSION_REPORT_PASSES`).
- Every call: `self._ai.generate(feature=..., model=self._ai.perception_model | insight_model, ...)`; catch `AIUnavailable` and surface its (vendor-neutral) message as a `paused` state. Never build a `genai.Client` elsewhere.
- Check `ai.budget.blocked_reason(feature, model)` before doing work to build a request.
- User-facing strings: **YUBI v3.0** only — never Gemini/Google in exports, UI copy, or `ai_usage`.

## Adding expression behavior

- New action unit / emotion / cue: `facs.py` (pure; add tests in `test_signal_processing.py`).
- Episode logic, speech gating, vitals: `micro_expressions.py` (`_Subject`); test with the synthetic face in `test_micro_expressions.py`.
- Drawing: `draw_expression_overlay` renders `SubjectRender`; OpenCV text must be ASCII.
- Status payload: `VisionEngine.get_expression_status()`; session data: `get_analysis()`.

## Tests

| File | Focus |
|------|--------|
| `tests/test_api.py` | HTTP lifecycle |
| `tests/test_gemini_reconcile.py` | Reconciliation + cache |
| `tests/test_session_report.py` | Report offline/stream (patch `settings` if env has API key) |
| `tests/test_ai_budget.py` | Pricing, caps, breaker, per-family settings |
| `tests/test_intelligence.py` | Scene gate, insight engine, report pipeline (`FakeAI`) |
| `tests/test_micro_expressions.py` | Synthetic-face episode engine |
| `tests/test_signal_processing.py` | Filters, pulse, FACS, cues |
