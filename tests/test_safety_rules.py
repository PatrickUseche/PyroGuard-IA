import pytest

from app.safety_rules import evaluate_safety_rules


def test_flame_alone_does_not_escalate_directly_to_fire():
    reading = {"temperature": 24, "humidity": 70, "smoke": 5, "gas": 0, "flame": True}
    probabilities = {"NORMAL": 0.15, "WARNING": 0.20, "FIRE": 0.65}
    result = evaluate_safety_rules(reading, probabilities, "FIRE")
    assert result["risk_level"] == "WARNING"


def test_fire_requires_corroborating_measurements():
    reading = {"temperature": 48, "humidity": 10, "smoke": 320, "gas": 150, "flame": True}
    probabilities = {"NORMAL": 0.01, "WARNING": 0.04, "FIRE": 0.95}
    result = evaluate_safety_rules(reading, probabilities, "FIRE")
    assert result["risk_level"] == "FIRE"
    assert result["rule"] == "FIRE_CORROBORATED"


CALM = {"temperature": 22, "humidity": 70, "smoke": 10, "gas": 0, "flame": False}
NO_RISK = {"NORMAL": 1.0, "WARNING": 0.0, "FIRE": 0.0}
FIRE_LIKELY = {"NORMAL": 0.2, "WARNING": 0.2, "FIRE": 0.6}


def evaluate(overrides, probabilities=NO_RISK, prediction="NORMAL"):
    return evaluate_safety_rules({**CALM, **overrides}, probabilities, prediction)


def test_calm_reading_is_normal():
    result = evaluate({})
    assert result["risk_level"] == "NORMAL"
    assert result["rule"] == "NORMAL_BASELINE"
    assert result["warning_signals"] == []


@pytest.mark.parametrize("overrides, expected", [
    ({"temperature": 29.9}, "NORMAL"),
    ({"temperature": 30}, "NORMAL"),
    ({"humidity": 35.1}, "NORMAL"),
    ({"humidity": 35}, "NORMAL"),
    ({"smoke": 49.9}, "NORMAL"),
    ({"smoke": 50}, "NORMAL"),
    ({"gas": 29.9}, "NORMAL"),
    ({"gas": 30}, "NORMAL"),
])
def test_a_single_warning_signal_is_not_enough(overrides, expected):
    assert evaluate(overrides)["risk_level"] == expected


@pytest.mark.parametrize("signal, below, at", [
    ("temperature", 29.9, 30),
    ("humidity", 35.1, 35),
    ("smoke", 49.9, 50),
    ("gas", 29.9, 30),
])
def test_warning_signal_thresholds_are_inclusive(signal, below, at):
    assert evaluate({signal: below})["warning_signals"] == []
    assert len(evaluate({signal: at})["warning_signals"]) == 1


def test_two_warning_signals_raise_warning():
    result = evaluate({"temperature": 31, "smoke": 60})
    assert result["risk_level"] == "WARNING"
    assert result["rule"] == "WARNING_REVIEW"
    assert len(result["warning_signals"]) == 2


def test_model_risk_at_threshold_raises_warning():
    assert evaluate({}, {"NORMAL": 0.45, "WARNING": 0.55, "FIRE": 0.0})["risk_level"] == "WARNING"
    assert evaluate({}, {"NORMAL": 0.46, "WARNING": 0.54, "FIRE": 0.0})["risk_level"] == "NORMAL"


def test_model_prediction_warning_raises_warning():
    assert evaluate({}, NO_RISK, "WARNING")["risk_level"] == "WARNING"


@pytest.mark.parametrize("overrides, confirmed", [
    ({"flame": True, "smoke": 80}, True),
    ({"flame": True, "smoke": 79.9}, False),
    ({"flame": True, "temperature": 35}, True),
    ({"flame": True, "temperature": 34.9}, False),
    ({"flame": True, "gas": 40}, True),
    ({"flame": True, "gas": 39.9}, False),
])
def test_flame_with_a_second_sensor_confirms_fire(overrides, confirmed):
    result = evaluate(overrides)
    assert (result["risk_level"] == "FIRE") is confirmed
    assert bool(result["fire_confirmations"]) is confirmed


@pytest.mark.parametrize("overrides", [
    {"temperature": 42, "smoke": 180},
    {"temperature": 42, "gas": 100},
    {"smoke": 180, "humidity": 15},
])
def test_two_high_signals_need_a_likely_fire_probability(overrides):
    assert evaluate(overrides, FIRE_LIKELY)["risk_level"] == "FIRE"
    assert evaluate(overrides, {"NORMAL": 0.5, "WARNING": 0.0, "FIRE": 0.5})["risk_level"] != "FIRE"


def test_one_high_signal_never_reaches_fire():
    assert evaluate({"temperature": 60}, FIRE_LIKELY)["risk_level"] != "FIRE"


def test_three_high_signals_with_dense_smoke_reach_fire_without_the_model():
    overrides = {"temperature": 42, "gas": 100, "smoke": 180}
    result = evaluate(overrides, NO_RISK)
    assert result["risk_level"] == "FIRE"
    assert result["rule"] == "FIRE_CORROBORATED"


def test_three_high_signals_below_dense_smoke_need_the_model():
    overrides = {"temperature": 42, "gas": 100, "humidity": 15, "smoke": 119}
    assert evaluate(overrides, NO_RISK)["risk_level"] != "FIRE"
    assert evaluate({**overrides, "smoke": 120}, NO_RISK)["risk_level"] == "FIRE"


def test_without_flame_confirmations_stay_empty():
    result = evaluate({"smoke": 500, "temperature": 50, "gas": 200, "humidity": 10})
    assert result["risk_level"] == "FIRE"
    assert "sensor de llama activo" not in " ".join(result["fire_confirmations"])
