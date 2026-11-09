"""Verifica docs/contract/examples contra la API real (contrato IA-Backend v1.1)."""
import json
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from app import main
from app.main import app
from app.schemas import BatchResponse, MetricsResponse, PredictionResponse

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "docs" / "contract" / "examples"
RISK_LEVELS = {"NORMAL", "WARNING", "FIRE"}

client = TestClient(app)


def load_example(name):
    return json.loads((EXAMPLES_DIR / name).read_text(encoding="utf-8"))


def is_probability(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def assert_backend_can_consume(body):
    # Mismas comprobaciones que api/services/ai_client.py del backend.
    assert body["contract_version"].split(".")[0] == "1"
    assert body["risk_level"] in RISK_LEVELS
    assert is_probability(body["risk_probability"])
    assert set(body["probabilities"]) <= RISK_LEVELS
    assert all(is_probability(value) for value in body["probabilities"].values())
    assert isinstance(body["model_version"], str) and body["model_version"]
    assert isinstance(body["safety_rules"]["explanation"], str)


@pytest.mark.parametrize("name", sorted(path.name for path in EXAMPLES_DIR.glob("request_*.json")))
def test_request_examples_are_accepted(name):
    response = client.post("/predict", json=load_example(name))

    assert response.status_code == 200, response.text
    assert_backend_can_consume(response.json())


def test_response_example_matches_schema_and_backend_checks():
    example = load_example("response_ok.json")

    PredictionResponse.model_validate(example)
    assert_backend_can_consume(example)


def test_response_example_is_current():
    # Si falla tras reentrenar o cambiar el esquema, regenera response_ok.json desde la API.
    response = client.post("/predict", json=load_example("request_fire.json"))

    assert response.json() == load_example("response_ok.json")


def test_validation_error_example_is_current():
    payload = load_example("request_fire.json")
    payload["temperature"] = 151

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert response.json() == load_example("error_validation.json")


def test_unauthorized_error_example_is_current(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", "clave-de-prueba")

    response = client.post("/predict", json=load_example("request_fire.json"))

    assert response.status_code == 401
    assert response.json() == load_example("error_unauthorized.json")


def test_model_unavailable_error_example_is_current(monkeypatch):
    def missing_model():
        raise FileNotFoundError("modelo")

    monkeypatch.setattr(main, "get_predictor", missing_model)

    response = client.post("/predict", json=load_example("request_fire.json"))

    assert response.status_code == 503
    assert response.json() == load_example("error_model_unavailable.json")


def test_batch_request_example_is_accepted_and_response_is_current():
    response = client.post("/predict/batch", json=load_example("batch_request.json"))

    assert response.status_code == 200, response.text
    assert response.json() == load_example("batch_response.json")


def test_batch_response_example_matches_schema_and_mixes_results():
    example = load_example("batch_response.json")

    BatchResponse.model_validate(example)
    assert example["contract_version"].split(".")[0] == "1"
    assert [item["status"] for item in example["results"]] == ["ok", "error", "ok"]
    assert example["results"][1]["index"] == 1
    assert_backend_can_consume(example["results"][0])


def test_metrics_response_example_matches_schema():
    example = load_example("metrics_response.json")

    MetricsResponse.model_validate(example)
    assert sum(example["predictions_by_risk_level"].values()) == example["predictions_total"]
