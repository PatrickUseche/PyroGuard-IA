import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def normal_payload():
    return {
        "node_id": "TEST-001",
        "temperature": 22.0,
        "humidity": 70.0,
        "smoke": 12.0,
        "gas": 5.0,
        "flame": False,
        "timestamp": "2026-10-13T12:00:00-05:00",
    }


def test_health_has_model_loaded():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True
    assert body["status"] == "ok"


def test_prediction_contract_and_probabilities():
    response = client.post("/predict", json=normal_payload())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert body["node_id"] == "TEST-001"
    assert body["risk_level"] in {"NORMAL", "WARNING", "FIRE"}
    assert 0 <= body["risk_probability"] <= 1
    assert set(body["probabilities"]) == {"NORMAL", "WARNING", "FIRE"}
    assert body["model_version"]
    assert body["contract_version"] == "1.1"


def test_gas_is_optional():
    payload = normal_payload()
    payload.pop("gas")
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text


def test_gas_null_is_treated_as_zero():
    payload = normal_payload()
    payload["gas"] = None
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("field, value", [
    ("temperature", -50),
    ("temperature", 150),
    ("smoke", 20000),
    ("gas", 20000),
])
def test_accepts_values_valid_in_backend(field, value):
    payload = normal_payload()
    payload[field] = value
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("flame, expected", [(True, True), (False, False), (1, True), (0, False)])
def test_flame_accepts_bool_or_binary(flame, expected):
    payload = normal_payload()
    payload["flame"] = flame
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    assert ("sensor de llama activo" in response.json()["safety_rules"]["warning_signals"]) is expected


@pytest.mark.parametrize("flame", ["yes", "true", 2, 1.0, None])
def test_flame_rejects_non_boolean_values(flame):
    payload = normal_payload()
    payload["flame"] = flame
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize("field, value", [
    ("humidity", 150),
    ("temperature", 150.1),
    ("temperature", -50.1),
    ("smoke", -1),
    ("gas", -1),
])
def test_rejects_out_of_range_values(field, value):
    payload = normal_payload()
    payload[field] = value
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_model_is_loaded_when_the_service_starts(monkeypatch):
    from app import main

    main.get_predictor.cache_clear()
    with TestClient(main.app):
        assert main.get_predictor.cache_info().currsize == 1


def test_service_starts_even_if_the_model_cannot_be_loaded(monkeypatch):
    from app import main

    def failing_loader():
        raise FileNotFoundError("sin modelo")

    monkeypatch.setattr(main, "get_predictor", failing_loader)
    with TestClient(main.app) as started:
        assert started.get("/health").json()["status"] == "degraded"
