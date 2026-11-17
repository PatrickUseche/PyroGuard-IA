"""Propiedades del predictor sobre lecturas aleatorias dentro de los rangos del contrato v1.1."""
import random

import pytest

from app.predictor import RiskPredictor

LEVELS = {"NORMAL", "WARNING", "FIRE"}


def random_readings(count, seed=20261117):
    rng = random.Random(seed)
    return [
        {
            "node_id": f"PROP-{index:03d}",
            "temperature": rng.uniform(-50, 150),
            "humidity": rng.uniform(0, 100),
            "smoke": rng.uniform(0, 2000),
            "gas": rng.choice([None, rng.uniform(0, 1000)]),
            "flame": rng.random() < 0.3,
        }
        for index in range(count)
    ]


@pytest.fixture(scope="module")
def results():
    return RiskPredictor().predict_many(random_readings(300))


def test_probabilities_cover_the_three_classes_and_sum_to_one(results):
    for result in results:
        assert set(result["probabilities"]) == LEVELS
        assert all(0.0 <= value <= 1.0 for value in result["probabilities"].values())
        assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)


def test_risk_probability_is_warning_plus_fire(results):
    for result in results:
        expected = result["probabilities"]["WARNING"] + result["probabilities"]["FIRE"]
        assert result["risk_probability"] == pytest.approx(min(1.0, expected), abs=1e-5)
        assert result["risk_probability_percent"] == pytest.approx(result["risk_probability"] * 100, abs=0.01)


def test_levels_are_valid_and_follow_the_input_order(results):
    assert [result["node_id"] for result in results] == [f"PROP-{index:03d}" for index in range(300)]
    for result in results:
        assert result["risk_level"] in LEVELS
        assert result["model_prediction"] in LEVELS
