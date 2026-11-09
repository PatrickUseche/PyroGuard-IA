import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.metrics import metrics
from app.schemas import MAX_BATCH_SIZE

client = TestClient(app)

CALM = {"node_id": "B-1", "temperature": 22, "humidity": 70, "smoke": 10, "gas": 0, "flame": False}
FIRE = {"node_id": "B-2", "temperature": 48, "humidity": 10, "smoke": 320, "gas": 150, "flame": True}
INVALID = {"node_id": "B-3", "temperature": 999, "humidity": 10, "smoke": 3, "flame": "yes"}


@pytest.fixture(autouse=True)
def clean_metrics():
    metrics.reset()
    yield
    metrics.reset()


def post(readings):
    return client.post("/predict/batch", json={"readings": readings})


def comparable(result):
    return {key: value for key, value in result.items() if key != "timestamp"}


def test_batch_returns_one_result_per_reading_in_order():
    response = post([CALM, FIRE])
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["status"], body["count"], body["succeeded"], body["failed"]) == ("ok", 2, 2, 0)
    assert [item["node_id"] for item in body["results"]] == ["B-1", "B-2"]
    assert [item["risk_level"] for item in body["results"]] == ["NORMAL", "FIRE"]


def test_batch_results_are_equivalent_to_predict():
    batch = post([CALM, FIRE]).json()["results"]
    single = [client.post("/predict", json=reading).json() for reading in (CALM, FIRE)]
    assert [comparable(item) for item in batch] == [comparable(item) for item in single]


def test_invalid_reading_is_reported_without_rejecting_the_batch():
    body = post([CALM, INVALID, FIRE]).json()
    assert (body["count"], body["succeeded"], body["failed"]) == (3, 2, 1)
    assert [item["status"] for item in body["results"]] == ["ok", "error", "ok"]
    error = body["results"][1]
    assert error["index"] == 1
    assert set(error["errors"]) == {"temperature", "flame"}
    assert "999" not in str(error)


def test_batch_where_every_reading_is_invalid_still_answers_200():
    body = post([INVALID, INVALID]).json()
    assert (body["succeeded"], body["failed"]) == (0, 2)
    assert [item["index"] for item in body["results"]] == [0, 1]


def test_gas_null_is_accepted_in_batch():
    assert post([{**CALM, "gas": None}]).json()["succeeded"] == 1


def test_empty_batch_is_rejected():
    response = post([])
    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"
    assert "readings" in response.json()["errors"]


def test_batch_limit_is_inclusive():
    assert post([CALM] * MAX_BATCH_SIZE).json()["count"] == MAX_BATCH_SIZE
    assert post([CALM] * (MAX_BATCH_SIZE + 1)).status_code == 422


@pytest.mark.parametrize("body", [{}, {"readings": "texto"}, {"readings": [1, 2]}, []])
def test_malformed_batch_body_is_validation_error(body):
    response = client.post("/predict/batch", json=body)
    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"


def test_batch_requires_the_api_key_when_configured(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", "clave-de-prueba")
    assert post([CALM]).status_code == 401
    allowed = client.post("/predict/batch", json={"readings": [CALM]}, headers={"X-AI-API-Key": "clave-de-prueba"})
    assert allowed.status_code == 200


def test_batch_counts_only_successful_predictions_in_metrics():
    post([CALM, INVALID, FIRE])
    body = client.get("/metrics").json()
    assert body["predictions_by_risk_level"] == {"NORMAL": 1, "WARNING": 0, "FIRE": 1}


def test_batch_model_unavailable_is_503(monkeypatch):
    from app import main

    def failing_loader():
        raise FileNotFoundError("sin modelo")

    monkeypatch.setattr(main, "get_predictor", failing_loader)
    response = post([CALM])
    assert response.status_code == 503
    assert response.json()["code"] == "MODEL_UNAVAILABLE"


def test_batch_with_only_invalid_readings_does_not_need_the_model(monkeypatch):
    from app import main

    def failing_loader():
        raise FileNotFoundError("sin modelo")

    monkeypatch.setattr(main, "get_predictor", failing_loader)
    assert post([INVALID]).status_code == 200
