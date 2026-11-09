import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from app.calibration import (
    CALIBRATION_METHOD,
    calibrate,
    expected_calibration_error,
    multiclass_brier,
    probability_quality,
)
from app.evaluation_set import TRAINING_SET_PATH
from app.labeled_data import FEATURE_COLUMNS
from app.model_comparison import production_forest

CLASSES = ["FIRE", "NORMAL", "WARNING"]


@pytest.fixture(scope="module")
def split():
    data = pd.read_csv(TRAINING_SET_PATH)
    features, target = data[FEATURE_COLUMNS].astype(float), data["label"]
    # Misma partición que scripts/2_entrenar_modelo.py.
    return train_test_split(features, target, test_size=0.20, random_state=42, stratify=target)


@pytest.fixture(scope="module")
def models(split):
    x_train, _, y_train, _ = split
    return clone(production_forest()).fit(x_train, y_train), calibrate(production_forest(), x_train, y_train)


def test_calibrated_probabilities_are_valid_distributions(models, split):
    _, calibrated = models
    probabilities = calibrated.predict_proba(split[1])
    assert probabilities.shape == (len(split[1]), 3)
    assert probabilities.min() >= 0 and probabilities.max() <= 1
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_calibrated_model_keeps_the_classes_and_the_predictor_contract(models):
    base, calibrated = models
    assert list(calibrated.classes_) == list(base.classes_) == CLASSES
    assert CALIBRATION_METHOD == "isotonic"


def test_calibration_does_not_hurt_accuracy_much(models, split):
    base, calibrated = models
    _, x_test, _, y_test = split
    assert abs((base.predict(x_test) == y_test).mean() - (calibrated.predict(x_test) == y_test).mean()) <= 0.01


def test_calibration_improves_probability_quality_on_held_out_data(models, split):
    base, calibrated = models
    _, x_test, _, y_test = split
    before, after = probability_quality(base, x_test, y_test), probability_quality(calibrated, x_test, y_test)
    assert after["expected_calibration_error"] < before["expected_calibration_error"]
    assert after["brier"] < before["brier"]
    assert after["log_loss"] <= before["log_loss"] * 1.05


def test_perfect_confident_predictions_have_zero_error():
    truth = pd.Series(["NORMAL", "FIRE", "WARNING"])
    probabilities = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert multiclass_brier(probabilities, truth, CLASSES) == 0.0
    assert expected_calibration_error(probabilities, truth, CLASSES) == 0.0


def test_overconfident_wrong_predictions_are_penalized():
    truth = pd.Series(["NORMAL", "NORMAL"])
    probabilities = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # Siempre FIRE con confianza total.
    assert multiclass_brier(probabilities, truth, CLASSES) == 2.0
    assert expected_calibration_error(probabilities, truth, CLASSES) == 1.0


def test_half_confident_half_right_is_calibrated():
    truth = pd.Series(["FIRE", "NORMAL"])
    probabilities = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
    assert expected_calibration_error(probabilities, truth, CLASSES) == pytest.approx(0.0)


def test_probability_quality_reports_the_three_measures(models, split):
    quality = probability_quality(models[1], split[1], split[3])
    assert set(quality) == {"log_loss", "brier", "expected_calibration_error"}
    assert all(value >= 0 for value in quality.values())
