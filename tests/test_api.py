from __future__ import annotations

from backend.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_index_returns_html() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_health() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["app"] == "YUBI Supervision"
    assert payload["vision_state"] == "idle"
    assert "youtube_cache_dir" in payload
    assert payload["youtube_cache_dir"]
    assert {p["key"] for p in payload["profiles"]} == {"general", "interview", "security"}
    assert payload["report_steps"]


def test_config_round_trip() -> None:
    response = client.get("/api/config")
    assert response.status_code == 200
    assert "show_objects" in response.json()

    response = client.post("/api/config", json={"confidence": 0.5})
    assert response.status_code == 200
    assert response.json()["confidence"] == 0.5


def test_snapshot_requires_live() -> None:
    response = client.get("/api/snapshot")
    assert response.status_code == 409

    response = client.get("/api/snapshot/json")
    assert response.status_code == 409


def test_record_requires_live() -> None:
    response = client.post("/api/record/start")
    assert response.status_code == 409

    # stop is idempotent and safe when idle
    response = client.post("/api/record/stop")
    assert response.status_code == 200
    assert response.json() == {"recording": False}


def test_status_exposes_intelligence_without_model_names() -> None:
    response = client.get("/api/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["expressions"]["state"] in {"off", "idle", "no_face"}
    assert "state" in payload["insights"]
    assert payload["ai_usage"]["calls"] == 0
    assert "gemini" not in str(payload["ai_usage"]).lower()
    assert "gemini" not in str(payload["insights"]).lower()


def test_analysis_endpoints_when_idle() -> None:
    response = client.get("/api/analysis")
    assert response.status_code == 200
    payload = response.json()
    assert payload["expressions"]["episodes"] == []
    assert "ai_usage" in payload

    response = client.get("/api/analysis/export")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    assert response.json()["session"] is not None

    response = client.get("/api/analysis/report.md")
    assert response.status_code == 200
    assert response.text == ""


def test_start_stop_lifecycle() -> None:
    response = client.get("/api/status")
    assert response.status_code == 200
    assert response.json()["state"] == "idle"

    response = client.post("/api/start")
    assert response.status_code == 200
    assert response.json()["state"] in {"starting", "live"}

    response = client.post("/api/stop")
    assert response.status_code == 200
    assert response.json()["state"] == "idle"
