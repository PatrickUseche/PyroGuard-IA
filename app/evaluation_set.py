"""Conjunto de prueba separado del entrenamiento.

`data/lecturas_etiquetadas.csv` (lecturas del simulador etiquetadas a mano) nunca se usa para entrenar: solo para medir.
Antes de usarlo se comprueba que no esté contaminado con filas del dataset de entrenamiento ni con datos sintéticos.
"""
from pathlib import Path

import pandas as pd

from app.labeled_data import FEATURE_COLUMNS, LabeledDataError, load_labeled_csv
from app.model_meta import ROOT_DIR

EVALUATION_SET_PATH = ROOT_DIR / "data" / "lecturas_etiquetadas.csv"
TRAINING_SET_PATH = ROOT_DIR / "data" / "dataset_sintetico.csv"
SYNTHETIC_NODE_PREFIX = "SYNTH-"
# Lo que mide este conjunto: lecturas del simulador del backend etiquetadas por una persona, no datos de campo.
EVALUATION_DATA_TYPE = "simulated"


class EvaluationSetError(ValueError):
    """El conjunto de prueba no sirve para evaluar (vacío, sintético o contaminado)."""


def _feature_keys(frame: pd.DataFrame) -> set[tuple]:
    rounded = frame[FEATURE_COLUMNS].astype(float).round(2)
    return set(map(tuple, rounded.itertuples(index=False, name=None)))


def find_overlap(evaluation: pd.DataFrame, training: pd.DataFrame) -> int:
    """Cuenta filas de evaluación cuyas medidas (a 2 decimales) también están en el entrenamiento."""
    training_keys = _feature_keys(training)
    rounded = evaluation[FEATURE_COLUMNS].astype(float).round(2)
    return sum(1 for key in map(tuple, rounded.itertuples(index=False, name=None)) if key in training_keys)


def ensure_valid_evaluation_set(evaluation: pd.DataFrame, training: pd.DataFrame | None = None) -> pd.DataFrame:
    synthetic = evaluation["node_id"].astype(str).str.startswith(SYNTHETIC_NODE_PREFIX)
    if synthetic.any():
        raise EvaluationSetError(
            f"{int(synthetic.sum())} fila(s) son sintéticas (node_id «{SYNTHETIC_NODE_PREFIX}…»): no sirven como conjunto de prueba."
        )
    if training is not None:
        overlap = find_overlap(evaluation, training)
        if overlap:
            raise EvaluationSetError(
                f"{overlap} fila(s) de prueba tienen las mismas medidas que filas de entrenamiento: hay fuga de datos."
            )
    return evaluation


def load_evaluation_set(
    path: str | Path = EVALUATION_SET_PATH, training_path: str | Path | None = TRAINING_SET_PATH
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise EvaluationSetError(
            f"No existe {path}. Importa lecturas con scripts/4_importar_lecturas.py, etiquétalas y guárdalas ahí."
        )
    try:
        evaluation = load_labeled_csv(path)
    except LabeledDataError as error:
        raise EvaluationSetError(str(error)) from error
    training = pd.read_csv(training_path) if training_path and Path(training_path).exists() else None
    return ensure_valid_evaluation_set(evaluation, training)
