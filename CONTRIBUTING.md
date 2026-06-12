# Contributing

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
cp .env.example .env   # optional
```

## Before opening a PR

```bash
ruff check backend tests
pytest
./scripts/dev.sh       # manual smoke: Start Vision in browser
```

## Context for agents

- **Canonical instructions:** [AGENTS.md](AGENTS.md)
- **Architecture:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Roadmap:** [docs/ROADMAP.md](docs/ROADMAP.md)

## Scope

This project intentionally stays minimal until features are requested. Prefer small, focused PRs. Do not add auth, databases, or frontend build tooling without an explicit issue or discussion.
