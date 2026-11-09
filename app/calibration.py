"""Calibración de las probabilidades del Random Forest.

Un bosque suele dar probabilidades poco confiables (por ejemplo, 0,9 de «FIRE» que acierta menos del 90 % de las veces).
Las reglas de seguridad y el backend usan esas probabilidades como umbrales, así que se calibran con regresión isotónica
y validación cruzada interna (`CalibratedClassifierCV`), que mantiene `classes_` y `predict_proba`.

Se usa `ensemble=False`: el calibrador se ajusta con predicciones fuera de pliegue pero el modelo final es un solo bosque.
Con el valor por defecto se guardan 5 bosques (~10 MB) y cada predicción tarda ~4 veces más, lo que el backend no tolera.
"""
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

CALIBRATION_METHOD = "isotonic"
CALIBRATION_FOLDS = 5
ECE_BINS = 10


def calibrate(base_model: Any, features: pd.DataFrame, target: pd.Series, method: str = CALIBRATION_METHOD,
              folds: int = CALIBRATION_FOLDS) -> CalibratedClassifierCV:
    """Entrena `base_model` (una copia por pliegue) y ajusta el calibrador sobre las predicciones fuera de pliegue."""
    return CalibratedClassifierCV(base_model, method=method, cv=folds, ensemble=False).fit(features, target)


def multiclass_brier(probabilities: np.ndarray, truth: pd.Series, classes: list[str]) -> float:
    """Error cuadrático medio entre las probabilidades y la clase real en one-hot (0 = perfecto)."""
    one_hot = np.array([[1.0 if actual == label else 0.0 for label in classes] for actual in truth])
    return float(((probabilities - one_hot) ** 2).sum(axis=1).mean())


def expected_calibration_error(probabilities: np.ndarray, truth: pd.Series, classes: list[str], bins: int = ECE_BINS) -> float:
    """Diferencia media entre la confianza de la clase predicha y su acierto real, ponderada por lecturas por tramo."""
    confidence = probabilities.max(axis=1)
    predicted = np.array(classes)[probabilities.argmax(axis=1)]
    correct = predicted == np.asarray(truth)
    error = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            error += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)


def probability_quality(model: Any, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    classes = [str(label) for label in model.classes_]
    probabilities = model.predict_proba(features)
    return {
        "log_loss": round(float(log_loss(target, probabilities, labels=classes)), 6),
        "brier": round(multiclass_brier(probabilities, target, classes), 6),
        "expected_calibration_error": round(expected_calibration_error(probabilities, target, classes), 6),
    }
