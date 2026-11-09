import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import joblib
import pandas as pd
import sklearn

from app.safety_rules import evaluate_safety_rules

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "modelo_riesgo.joblib"
logger = logging.getLogger("pyroguard.api")
FEATURE_COLUMNS = ["temperature", "humidity", "smoke", "gas", "flame"]
DISCLAIMER = (
    "Prototipo académico entrenado con datos sintéticos. Las salidas no son una detección certificada "
    "de incendios reales y requieren validación con datos de sensores reales y pruebas supervisadas."
)


class RiskPredictor:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        artifact = joblib.load(self.model_path)
        if not isinstance(artifact, dict) or "model" not in artifact:
            raise ValueError("El archivo de modelo no contiene los metadatos esperados.")
        self.model = artifact["model"]
        self.model_version = str(artifact.get("model_version", "unknown"))
        self.feature_columns = list(artifact.get("feature_columns", FEATURE_COLUMNS))
        self.training_data_type = str(artifact.get("training_data_type", "unknown"))
        # Trazabilidad: con qué datos y con qué versión de scikit-learn se entrenó este artefacto.
        self.dataset_sha256 = artifact.get("dataset_sha256")
        self.trained_with_sklearn = artifact.get("scikit_learn_version")
        if self.trained_with_sklearn and self.trained_with_sklearn != sklearn.__version__:
            logger.warning(
                "El modelo se entrenó con scikit-learn %s y el servicio usa %s; reentrena si hay resultados raros.",
                self.trained_with_sklearn,
                sklearn.__version__,
            )
        if self.feature_columns != FEATURE_COLUMNS:
            raise ValueError(f"Columnas incompatibles en el modelo: {self.feature_columns}")
        if set(self.model.classes_) != {"NORMAL", "WARNING", "FIRE"}:
            raise ValueError(f"Clases inesperadas en el modelo: {self.model.classes_}")

    @property
    def classes(self) -> list[str]:
        return [str(item) for item in self.model.classes_]

    @staticmethod
    def _feature_row(reading: dict[str, Any]) -> dict[str, float]:
        return {
            "temperature": float(reading["temperature"]),
            "humidity": float(reading["humidity"]),
            "smoke": float(reading["smoke"]),
            "gas": float(reading.get("gas", 0.0) or 0.0),
            "flame": int(bool(reading["flame"])),
        }

    def predict(self, reading: dict[str, Any]) -> dict[str, Any]:
        return self.predict_many([reading])[0]

    def predict_many(self, readings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Clasifica varias lecturas con una sola llamada al modelo; mantiene el orden de entrada."""
        if not readings:
            return []
        rows = [self._feature_row(reading) for reading in readings]
        frame = pd.DataFrame(rows, columns=self.feature_columns)
        # La clase es el argmax de las probabilidades (lo que hace RandomForest.predict): una sola pasada por el bosque.
        raw_probabilities = self.model.predict_proba(frame)
        model_predictions = self.model.classes_[raw_probabilities.argmax(axis=1)]
        return [
            self._build_result(reading, row, str(prediction), probabilities)
            for reading, row, prediction, probabilities in zip(readings, rows, model_predictions, raw_probabilities)
        ]

    def _build_result(
        self, reading: dict[str, Any], row: dict[str, float], model_prediction: str, raw_probabilities: Any
    ) -> dict[str, Any]:
        probabilities = {str(label): round(float(prob), 6) for label, prob in zip(self.model.classes_, raw_probabilities)}
        for label in ("NORMAL", "WARNING", "FIRE"):
            probabilities.setdefault(label, 0.0)
        risk_probability = min(1.0, max(0.0, probabilities["WARNING"] + probabilities["FIRE"]))
        safety = evaluate_safety_rules(row, probabilities, model_prediction)
        measured_at = reading.get("timestamp") or datetime.now(timezone.utc)
        if measured_at.tzinfo is None:
            measured_at = measured_at.replace(tzinfo=timezone.utc)

        return {
            "status": "ok",
            "node_id": str(reading["node_id"]),
            "timestamp": measured_at,
            "model_version": self.model_version,
            "model_prediction": model_prediction,
            "risk_level": safety["risk_level"],
            "risk_probability": round(risk_probability, 6),
            "risk_probability_percent": round(risk_probability * 100, 2),
            "probabilities": probabilities,
            "safety_rules": {
                "rule": safety["rule"],
                "explanation": safety["explanation"],
                "fire_confirmations": safety["fire_confirmations"],
                "warning_signals": safety["warning_signals"],
            },
            "synthetic_model": self.training_data_type == "synthetic",
            "disclaimer": DISCLAIMER,
        }
