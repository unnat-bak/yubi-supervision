from __future__ import annotations

import json
from unittest.mock import patch

from backend.config import Settings
from backend.main import app
from backend.session_report import enrich_session_report_events
from fastapi.testclient import TestClient

client = TestClient(app)

OFFLINE_SETTINGS = Settings(gemini_enabled=False, gemini_api_key="")


def test_enrich_offline_returns_draft() -> None:
    events = list(enrich_session_report_events("# Session draft", {}, OFFLINE_SETTINGS))
    assert events[0]["phase"] == "compile"
    done = events[-1]
    assert done["phase"] == "done"
    assert done["markdown"] == "# Session draft"
    assert done["enriched"] is False
    assert done["passes_completed"] == 0


def test_session_report_api_offline() -> None:
    with patch("backend.main.settings", OFFLINE_SETTINGS):
        response = client.post(
            "/api/session-report",
            json={"draft_markdown": "# Test log", "session": {"id": "sess-1"}},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["markdown"] == "# Test log"
    assert payload["enriched"] is False


def test_session_report_stream_offline() -> None:
    with patch("backend.main.settings", OFFLINE_SETTINGS):
        response = client.post(
            "/api/session-report/stream",
            json={"draft_markdown": "# Stream draft", "session": {}},
        )
    assert response.status_code == 200
    lines = [line for line in response.text.strip().split("\n") if line]
    events = [json.loads(line) for line in lines]
    assert events[0]["phase"] == "compile"
    assert events[-1]["phase"] == "done"
    assert events[-1]["markdown"] == "# Stream draft"
