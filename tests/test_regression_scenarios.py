"""Escenarios normal / warning / fire que detectan cambios de comportamiento al tocar el modelo o las reglas."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

SCENARIOS = json.loads((Path(__file__).parent / "regression" / "scenarios.json").read_text(encoding="utf-8"))["scenarios"]
IDS = [scenario["name"] for scenario in SCENARIOS]

client = TestClient(app)


def with_node(scenario):
    return {"node_id": f"REG-{scenario['name']}"[:50], **scenario["reading"]}


def test_scenarios_cover_every_risk_level():
    assert {scenario["expected"]["risk_level"] for scenario in SCENARIOS} == {"NORMAL", "WARNING", "FIRE"}
    assert len(set(IDS)) == len(IDS)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=IDS)
def test_predict_keeps_the_expected_behavior(scenario):
    response = client.post("/predict", json=with_node(scenario))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["risk_level"] == scenario["expected"]["risk_level"]
    assert body["safety_rules"]["rule"] == scenario["expected"]["rule"]
    assert body["model_prediction"] == scenario["expected"]["model_prediction"]


def test_the_model_never_lowers_what_the_rules_raise():
    # Contrato: la salida final puede elevar la predicción del modelo, nunca bajarla por debajo de FIRE si hay confirmación.
    order = {"NORMAL": 0, "WARNING": 1, "FIRE": 2}
    for scenario in SCENARIOS:
        body = client.post("/predict", json=with_node(scenario)).json()
        if body["safety_rules"]["fire_confirmations"]:
            assert order[body["risk_level"]] == 2


def test_batch_gives_the_same_levels_as_single_predictions():
    response = client.post("/predict/batch", json={"readings": [with_node(scenario) for scenario in SCENARIOS]})
    levels = [item["risk_level"] for item in response.json()["results"]]
    assert levels == [scenario["expected"]["risk_level"] for scenario in SCENARIOS]


def test_probabilities_of_the_predicted_class_are_dominant_in_clear_scenarios():
    clear = {"normal_forest_morning": "NORMAL", "warning_smoke_rising": "WARNING", "fire_all_sensors": "FIRE"}
    for scenario in SCENARIOS:
        if scenario["name"] in clear:
            body = client.post("/predict", json=with_node(scenario)).json()
            assert body["probabilities"][clear[scenario["name"]]] >= 0.9
