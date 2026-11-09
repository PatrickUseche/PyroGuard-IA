"""Entrena y evalúa el Random Forest. Genera el artefacto utilizado por la API."""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import joblib
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.calibration import CALIBRATION_FOLDS, CALIBRATION_METHOD, calibrate, probability_quality  # noqa: E402
from app.model_meta import DEFAULT_META_PATH, dataset_sha256  # noqa: E402

DATA_PATH = ROOT / "data" / "dataset_sintetico.csv"
MODEL_PATH = ROOT / "models" / "modelo_riesgo.joblib"
REPORT_PATH = ROOT / "reports" / "evaluation.json"
CONFUSION_PATH = ROOT / "reports" / "confusion_matrix.csv"
FEATURES = ["temperature", "humidity", "smoke", "gas", "flame"]
LABELS = ["NORMAL", "WARNING", "FIRE"]
META_PATH = DEFAULT_META_PATH
MODEL_VERSION = "pyroguard-rf-1.1.0"


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"No se encontró {DATA_PATH}. Ejecuta primero scripts/1_generar_dataset.py")
    data = pd.read_csv(DATA_PATH)
    missing = set(FEATURES + ["label"]) - set(data.columns)
    if missing:
        raise SystemExit(f"Faltan columnas necesarias: {sorted(missing)}")
    if data[FEATURES + ["label"]].isna().any().any():
        raise SystemExit("El dataset contiene valores vacíos en variables de entrenamiento.")
    X = data[FEATURES].copy()
    X["flame"] = X["flame"].astype(int)
    y = data["label"].astype(str)
    if set(y.unique()) != set(LABELS):
        raise SystemExit(f"Se esperaban las tres clases {LABELS}, pero se encontraron {sorted(y.unique())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    forest = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    # El bosque sin calibrar solo sirve de referencia para medir cuánto mejora la calibración.
    uncalibrated = clone(forest).fit(X_train, y_train)
    model = calibrate(forest, X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "model_version": MODEL_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_type": "synthetic",
        "dataset_path": DATA_PATH.relative_to(ROOT).as_posix(),
        "dataset_sha256": dataset_sha256(DATA_PATH),
        "dataset_rows": int(len(data)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": FEATURES,
        "labels": LABELS,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
        "calibration": {
            "method": CALIBRATION_METHOD,
            "folds": CALIBRATION_FOLDS,
            "uncalibrated": probability_quality(uncalibrated, X_test, y_test),
            "calibrated": probability_quality(model, X_test, y_test),
        },
        "classification_report": classification_report(y_test, predictions, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix_labels_order": LABELS,
        "scikit_learn_version": sklearn.__version__,
        "limitations": "Métricas sobre datos sintéticos; no estiman desempeño en incendios reales.",
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "model_version": MODEL_VERSION,
        "feature_columns": FEATURES,
        "training_data_type": "synthetic",
        "calibration": CALIBRATION_METHOD,
        "dataset_sha256": metrics["dataset_sha256"],
        "dataset_rows": metrics["dataset_rows"],
        "trained_at_utc": metrics["trained_at_utc"],
        "scikit_learn_version": sklearn.__version__,
    }, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    # El meta.json sí se versiona junto al .joblib: dice con qué datos y con qué resultado se entrenó ese artefacto.
    META_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cm = confusion_matrix(y_test, predictions, labels=LABELS)
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(CONFUSION_PATH, index_label="real\\predicho")

    print("\n=== ENTRENAMIENTO PYROGUARD IA ===")
    print(f"Versión: {MODEL_VERSION}")
    print(f"Filas de entrenamiento: {len(X_train)} | Filas de prueba: {len(X_test)}")
    print(f"Accuracy sintético: {metrics['accuracy']:.4f}")
    print(f"Calibración ({CALIBRATION_METHOD}): sin calibrar {metrics['calibration']['uncalibrated']} -> calibrado {metrics['calibration']['calibrated']}")
    report = classification_report(y_test, predictions, labels=LABELS, zero_division=0)
    print(report)
    print(f"Modelo guardado: {MODEL_PATH}")
    print(f"Métricas guardadas: {REPORT_PATH}")
    print(f"Metadatos del modelo: {META_PATH}")
    print(f"Matriz de confusión: {CONFUSION_PATH}")
    print("ADVERTENCIA: el dataset es sintético y no valida el funcionamiento ante incendios reales.")


if __name__ == "__main__":
    main()
