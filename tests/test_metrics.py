import threading

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.metrics import ServiceMetrics, metrics

client = TestClient(app)

FIRE_PAYLOAD = {"node_id": "T-1", "temperature": 48, "humidity": 10, "smoke": 320, "gas": 150, "flame": True}
CALM_PAYLOAD = {"node_id": "T-2", "temperature": 22, "humidity": 70, "smoke": 10, "gas": 0, "flame": False}


@pytest.fixture(autouse=True)
def clean_metrics():
    metrics.reset()
    yield
    metrics.reset()


def test_empty_snapshot_has_zeroes():
    snapshot = ServiceMetrics().snapshot()
    assert snapshot["predictions_total"] == 0
    assert snapshot["predictions_by_risk_level"] == {"NORMAL": 0, "WARNING": 0, "FIRE": 0}
    assert snapshot["latency_ms"] == {"count": 0, "average": 0.0, "p95": 0.0, "max": 0.0}


def test_snapshot_aggregates_latency():
    collector = ServiceMetrics()
    for milliseconds in range(1, 21):
        collector.record_prediction("FIRE", milliseconds / 1000)
    latency = collector.snapshot()["latency_ms"]
    assert latency == {"count": 20, "average": 10.5, "p95": 19.0, "max": 20.0}


def test_p95_only_uses_the_recent_window():
    collector = ServiceMetrics(window=3)
    for seconds in (5.0, 0.001, 0.002, 0.003):
        collector.record_prediction("NORMAL", seconds)
    snapshot = collector.snapshot()["latency_ms"]
    assert snapshot["p95"] == 3.0
    assert snapshot["max"] == 5000.0
    assert snapshot["count"] == 4


def test_counters_are_thread_safe():
    collector = ServiceMetrics()

    def work():
        for _ in range(500):
            collector.record_prediction("WARNING", 0.001)

    threads = [threading.Thread(target=work) for _ in range(8)]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]
    assert collector.snapshot()["predictions_by_risk_level"]["WARNING"] == 4000


def test_metrics_endpoint_counts_predictions_by_risk_level():
    assert client.post("/predict", json=FIRE_PAYLOAD).json()["risk_level"] == "FIRE"
    client.post("/predict", json=CALM_PAYLOAD)
    client.post("/predict", json=CALM_PAYLOAD)
    body = client.get("/metrics").json()
    assert body["predictions_total"] == 3
    assert body["predictions_by_risk_level"] == {"NORMAL": 2, "WARNING": 0, "FIRE": 1}
    assert body["latency_ms"]["count"] == 3
    assert body["latency_ms"]["max"] >= body["latency_ms"]["average"] > 0


def test_rejected_requests_are_not_counted():
    client.post("/predict", json={"node_id": "T-3"})
    assert client.get("/metrics").json()["predictions_total"] == 0


def test_metrics_requires_the_api_key_when_configured(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", "clave-de-prueba")
    denied = client.get("/metrics")
    assert denied.status_code == 401
    assert denied.json()["code"] == "UNAUTHORIZED"
    assert client.get("/metrics", headers={"X-AI-API-Key": "otra"}).status_code == 401
    allowed = client.get("/metrics", headers={"X-AI-API-Key": "clave-de-prueba"})
    assert allowed.status_code == 200
