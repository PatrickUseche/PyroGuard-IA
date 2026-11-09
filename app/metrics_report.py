"""Informe de métricas por clase (precisión, recall y F1) sobre un conjunto etiquetado."""
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from app.labeled_data import FEATURE_COLUMNS, LABELS
from app.predictor import RiskPredictor

# Con menos lecturas de una clase, sus métricas son anecdóticas y el informe lo avisa.
MIN_RELIABLE_SUPPORT = 30


def classification_metrics(y_true: list[str], y_pred: list[str], labels: tuple[str, ...] = LABELS) -> dict[str, Any]:
    labels_list = list(labels)
    if not y_true:
        # scikit-learn falla con entradas vacías; un conjunto sin lecturas devuelve todo en cero.
        zero = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
        return {
            "accuracy": 0.0,
            "per_class": {label: dict(zero) for label in labels_list},
            "macro_avg": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "confusion_matrix": {
                "labels_order": labels_list,
                "rows_are_real_columns_are_predicted": [[0] * len(labels_list) for _ in labels_list],
            },
        }
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_list, zero_division=0
    )
    per_class = {
        label: {
            "precision": round(float(precision[index]), 6),
            "recall": round(float(recall[index]), 6),
            "f1": round(float(f1[index]), 6),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels_list)
    }
    present = [label for label in labels_list if per_class[label]["support"] > 0]
    macro = {
        key: round(sum(per_class[label][key] for label in present) / len(present), 6) if present else 0.0
        for key in ("precision", "recall", "f1")
    }
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "per_class": per_class,
        "macro_avg": macro,
        "confusion_matrix": {
            "labels_order": labels_list,
            "rows_are_real_columns_are_predicted": confusion_matrix(y_true, y_pred, labels=labels_list).tolist(),
        },
    }


def reliability_warnings(per_class: dict[str, dict[str, Any]]) -> list[str]:
    warnings = []
    for label, values in per_class.items():
        if values["support"] == 0:
            warnings.append(f"No hay lecturas de la clase {label}: sus métricas no se pueden calcular.")
        elif values["support"] < MIN_RELIABLE_SUPPORT:
            warnings.append(
                f"Solo hay {values['support']} lecturas de {label} (< {MIN_RELIABLE_SUPPORT}): sus métricas son poco fiables."
            )
    return warnings


def build_report(evaluation: pd.DataFrame, predictor: RiskPredictor, dataset_type: str = "simulated") -> dict[str, Any]:
    """Evalúa el modelo solo y el resultado final (modelo + reglas de seguridad) sobre el conjunto etiquetado."""
    readings = evaluation[["node_id", *FEATURE_COLUMNS]].to_dict("records")
    for reading in readings:
        reading["flame"] = bool(reading["flame"])
    predictions = predictor.predict_many(readings)
    truth = [str(label) for label in evaluation["label"]]
    model_only = classification_metrics(truth, [item["model_prediction"] for item in predictions])
    final = classification_metrics(truth, [item["risk_level"] for item in predictions])
    return {
        "model_version": predictor.model_version,
        "training_dataset_sha256": predictor.dataset_sha256,
        "evaluation_dataset_type": dataset_type,
        "evaluation_rows": int(len(evaluation)),
        "class_counts": {label: int(truth.count(label)) for label in LABELS},
        "model_only": model_only,
        "final_with_safety_rules": final,
        "warnings": reliability_warnings(final["per_class"]),
        "limitations": (
            "Lecturas del simulador etiquetadas por una persona: no son datos de campo, así que estas métricas "
            "no estiman el desempeño ante incendios reales."
        ),
    }
