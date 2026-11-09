import json

import pytest
from fastapi.testclient import TestClient
from app import main
from app.main import app

client = TestClient(app)


def payload():
    return {
        "node_id": "TEST-001",
        "temperature": 22.0,
        "humidity": 70.0,
        "smoke": 12.0,
        "gas": 5.0,
        "flame": False,
        "timestamp": "2026-10-13T12:00:00-05:00",
    }


def assert_error(response, status_code, code):
    assert response.status_code == status_code, response.text
    body = response.json()
    assert body["status"] == "error"
    assert body["code"] == code
    assert isinstance(body["errors"], dict)
    return body["errors"]


def test_validation_error_is_grouped_by_field():
    data = payload()
    data["temperature"] = 500
    data["flame"] = "yes"
    errors = assert_error(client.post("/predict", json=data), 422, "VALIDATION_ERROR")
    assert set(errors) == {"temperature", "flame"}
    assert all(isinstance(messages, list) and messages for messages in errors.values())


def test_missing_field_is_reported():
    data = payload()
    data.pop("smoke")
    errors = assert_error(client.post("/predict", json=data), 422, "VALIDATION_ERROR")
    assert "smoke" in errors


def test_nan_is_rejected_without_server_error():
    body = json.dumps(payload()).replace('"smoke": 12.0', '"smoke": NaN')
    response = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    errors = assert_error(response, 422, "VALIDATION_ERROR")
    assert "smoke" in errors


def test_malformed_json_is_a_non_field_error():
    response = client.post("/predict", content="{node_id:", headers={"Content-Type": "application/json"})
    errors = assert_error(response, 422, "VALIDATION_ERROR")
    assert list(errors) == ["non_field_errors"]


def test_missing_api_key_is_unauthorized(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", "clave-de-prueba")
    errors = assert_error(client.post("/predict", json=payload()), 401, "UNAUTHORIZED")
    assert errors["detail"]


def test_wrong_api_key_is_unauthorized(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", "clave-de-prueba")
    response = client.post("/predict", json=payload(), headers={"X-AI-API-Key": "otra"})
    assert_error(response, 401, "UNAUTHORIZED")


def test_valid_api_key_is_accepted(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", "clave-de-prueba")
    response = client.post("/predict", json=payload(), headers={"X-AI-API-Key": "clave-de-prueba"})
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("exc", [FileNotFoundError("modelo"), ValueError("columnas")])
def test_model_load_failure_is_model_unavailable(monkeypatch, exc):
    def failing_loader():
        raise exc

    monkeypatch.setattr(main, "get_predictor", failing_loader)
    assert_error(client.post("/predict", json=payload()), 503, "MODEL_UNAVAILABLE")


def test_inference_failure_does_not_leak_details(monkeypatch):
    class BrokenPredictor:
        def predict(self, _reading):
            raise ValueError("detalle interno sensible")

    monkeypatch.setattr(main, "get_predictor", BrokenPredictor)
    response = client.post("/predict", json=payload())
    errors = assert_error(response, 500, "PREDICTION_FAILED")
    assert "detalle interno sensible" not in json.dumps(errors)


def test_unexpected_exception_is_internal_error(monkeypatch):
    class CrashingPredictor:
        def predict(self, _reading):
            raise RuntimeError("fallo no previsto")

    monkeypatch.setattr(main, "get_predictor", CrashingPredictor)
    safe_client = TestClient(app, raise_server_exceptions=False)
    errors = assert_error(safe_client.post("/predict", json=payload()), 500, "INTERNAL_ERROR")
    assert "fallo no previsto" not in json.dumps(errors)


def test_unknown_route_is_not_found():
    assert_error(client.get("/no-existe"), 404, "NOT_FOUND")


def test_wrong_method_is_not_allowed():
    assert_error(client.get("/predict"), 405, "METHOD_NOT_ALLOWED")


def test_production_refuses_to_start_without_api_key(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("AI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="AI_API_KEY"):
        with TestClient(app):
            pass


def test_production_refuses_a_blank_api_key(monkeypatch):
    monkeypatch.setenv("APP_ENV", "Production")
    monkeypatch.setenv("AI_API_KEY", "   ")
    with pytest.raises(RuntimeError):
        with TestClient(app):
            pass


def test_production_starts_with_api_key(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AI_API_KEY", "clave-de-prueba")
    with TestClient(app) as started:
        assert started.get("/health").status_code == 200


@pytest.mark.parametrize("env", [None, "development", "test"])
def test_other_environments_start_without_api_key(monkeypatch, env):
    monkeypatch.delenv("AI_API_KEY", raising=False)
    if env is None:
        monkeypatch.delenv("APP_ENV", raising=False)
    else:
        monkeypatch.setenv("APP_ENV", env)
    with TestClient(app) as started:
        assert started.get("/health").status_code == 200
