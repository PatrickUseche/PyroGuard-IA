import pytest

from app.predictor import RiskPredictor

READINGS = [
    {"node_id": "N-1", "temperature": 22, "humidity": 70, "smoke": 10, "gas": 0, "flame": False},
    {"node_id": "N-2", "temperature": 33, "humidity": 28, "smoke": 90, "gas": 40, "flame": False},
    {"node_id": "N-3", "temperature": 48, "humidity": 10, "smoke": 320, "gas": 150, "flame": True},
    {"node_id": "N-4", "temperature": 24, "humidity": 70, "smoke": 5, "gas": None, "flame": True},
]


@pytest.fixture(scope="module")
def predictor():
    return RiskPredictor()


def strip_time(result):
    return {key: value for key, value in result.items() if key != "timestamp"}


def test_predict_many_matches_predict_one_by_one(predictor):
    batch = predictor.predict_many(READINGS)
    single = [predictor.predict(reading) for reading in READINGS]
    assert [strip_time(item) for item in batch] == [strip_time(item) for item in single]


def test_predict_many_keeps_input_order(predictor):
    assert [item["node_id"] for item in predictor.predict_many(READINGS)] == ["N-1", "N-2", "N-3", "N-4"]


def test_predict_many_with_no_readings_returns_empty_list(predictor):
    assert predictor.predict_many([]) == []


def test_predict_many_calls_the_model_once(predictor, monkeypatch):
    calls = {"predict": 0, "predict_proba": 0}
    real_predict, real_proba = predictor.model.predict, predictor.model.predict_proba

    def counting_predict(frame):
        calls["predict"] += 1
        return real_predict(frame)

    def counting_proba(frame):
        calls["predict_proba"] += 1
        return real_proba(frame)

    monkeypatch.setattr(predictor.model, "predict", counting_predict, raising=False)
    monkeypatch.setattr(predictor.model, "predict_proba", counting_proba, raising=False)
    predictor.predict_many(READINGS)
    assert calls == {"predict": 0, "predict_proba": 1}


def test_model_prediction_equals_the_models_own_predict(predictor):
    import pandas as pd

    frame = pd.DataFrame(
        [[r["temperature"], r["humidity"], r["smoke"], r["gas"] or 0.0, int(r["flame"])] for r in READINGS],
        columns=predictor.feature_columns,
    )
    expected = list(predictor.model.predict(frame))
    assert [item["model_prediction"] for item in predictor.predict_many(READINGS)] == expected


def build_artifact(tmp_path, **overrides):
    import joblib

    reference = joblib.load(RiskPredictor().model_path)
    reference.update(overrides)
    path = tmp_path / "modelo.joblib"
    joblib.dump(reference, path)
    return path


def test_warns_when_scikit_learn_version_differs(tmp_path, caplog):
    path = build_artifact(tmp_path, scikit_learn_version="0.0.1")
    with caplog.at_level("WARNING", logger="pyroguard.api"):
        RiskPredictor(path)
    assert "0.0.1" in caplog.text


def test_does_not_warn_when_scikit_learn_version_matches(caplog):
    with caplog.at_level("WARNING", logger="pyroguard.api"):
        RiskPredictor()
    assert "scikit-learn" not in caplog.text


def test_artifact_without_dataset_hash_loads_with_none(tmp_path):
    path = build_artifact(tmp_path, dataset_sha256=None)
    assert RiskPredictor(path).dataset_sha256 is None
