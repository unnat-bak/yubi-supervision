# Contributing

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
cp .env.example .env   # optional: camera, YOLO, GEMINI_API_KEY for YUBI v3.0
```

## Before opening a PR

```bash
ruff check backend tests
pytest
node --check frontend/app.js    # required after frontend JS changes
./scripts/dev.sh                # manual: Initialize, toggle layers, expressions, terminate, download report
```

## Context for agents & humans

| Doc | Contents |
|-----|----------|
| [AGENTS.md](AGENTS.md) | Canonical agent instructions, feature matrix, file map |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | API contracts, threading, v3.0, expressions, session report |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Production evolution |
| [backend/CLAUDE.md](backend/CLAUDE.md) | Backend module responsibilities |

## Scope

Prefer small, focused PRs. Do not add auth, databases, or frontend build tooling without an explicit issue.

When adding features:

- Update `AGENTS.md` and `docs/ARCHITECTURE.md` if behavior or API changes.
- Add env vars to `config.py`, `.env.example`, and the architecture config table.
- Add tests in `tests/` for non-trivial backend logic.
- User-visible strings: **YUBI v3.0** branding (not vendor model names).
