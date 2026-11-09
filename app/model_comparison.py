"""Compara el Random Forest de producción con otros clasificadores usando validación cruzada estratificada."""
import time
from typing import Any

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.labeled_data import FEATURE_COLUMNS, LABELS

SEED = 42
PRODUCTION_MODEL = "RandomForest"


def production_forest() -> RandomForestClassifier:
    # Mismos hiperparámetros que scripts/2_entrenar_modelo.py.
    return RandomForestClassifier(
        n_estimators=300, max_features="sqrt", min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1
    )


def candidate_models() -> dict[str, Any]:
    return {
        PRODUCTION_MODEL: production_forest(),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
        "DecisionTree": DecisionTreeClassifier(min_samples_leaf=5, class_weight="balanced", random_state=SEED),
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
    }


def compare_models(data: pd.DataFrame, folds: int = 5, models: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Devuelve una fila por modelo con exactitud y F1 macro (media y desviación entre pliegues) y tiempos."""
    features = data[FEATURE_COLUMNS].astype(float)
    target = data["label"].astype(str)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    scoring = {"accuracy": "accuracy", "f1_macro": make_scorer(f1_score, average="macro", labels=list(LABELS), zero_division=0)}
    rows = []
    for name, model in (models or candidate_models()).items():
        started = time.perf_counter()
        scores = cross_validate(model, features, target, cv=splitter, scoring=scoring, n_jobs=1)
        rows.append({
            "model": name,
            "accuracy_mean": round(float(scores["test_accuracy"].mean()), 4),
            "accuracy_std": round(float(scores["test_accuracy"].std()), 4),
            "f1_macro_mean": round(float(scores["test_f1_macro"].mean()), 4),
            "f1_macro_std": round(float(scores["test_f1_macro"].std()), 4),
            "fit_seconds": round(float(scores["fit_time"].mean()), 3),
            "predict_seconds": round(float(scores["score_time"].mean()), 3),
            "total_seconds": round(time.perf_counter() - started, 2),
        })
    return sorted(rows, key=lambda row: (-row["f1_macro_mean"], row["fit_seconds"]))


def render_markdown(rows: list[dict[str, Any]], dataset_rows: int, dataset_sha256: str, folds: int, date: str) -> str:
    best = rows[0]
    gap = max(row["f1_macro_mean"] for row in rows) - min(row["f1_macro_mean"] for row in rows)
    table = "\n".join(
        f"| {row['model']}{' (producción)' if row['model'] == PRODUCTION_MODEL else ''} | "
        f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f} | {row['f1_macro_mean']:.4f} ± {row['f1_macro_std']:.4f} | "
        f"{row['fit_seconds']:.3f} | {row['predict_seconds']:.3f} |"
        for row in rows
    )
    return f"""# Comparación de modelos

Generado con `python scripts/6_comparar_modelos.py` el {date}. Validación cruzada estratificada de {folds} pliegues
sobre `data/dataset_sintetico.csv` ({dataset_rows} filas, SHA-256 `{dataset_sha256[:12]}…`), semilla {SEED}.

| Modelo | Exactitud | F1 macro | Entrenar (s) | Predecir (s) |
|---|---|---|---|---|
{table}

Tiempos medios por pliegue en la máquina donde se generó la tabla; sirven para comparar entre filas, no como valor absoluto.

## Cómo leerlo

- **Estas cifras son sobre datos sintéticos** y el generador (`scripts/1_generar_dataset.py`) define las clases con rangos
  parecidos a los umbrales de las reglas de seguridad. La diferencia entre el mejor y el peor modelo es de
  {gap:.4f} de F1 macro: un modelo aprende ese reparto casi igual de bien que otro, así que **no demuestra que uno
  funcione mejor ante un incendio real**.
- El mejor por F1 macro en esta tabla es **{best['model']}**. Con diferencias dentro de la desviación entre pliegues no hay
  motivo para cambiar de modelo.
- Se mantiene el Random Forest porque entrega probabilidades razonables por clase (las usan las reglas de seguridad y el
  backend), no necesita escalar las variables y ya está desplegado y probado.
- Para decidir de verdad hace falta repetir esta comparación sobre `data/lecturas_etiquetadas.csv` (ver
  `docs/ETIQUETADO.md`) o, mejor, sobre datos de nodos reales.
"""
