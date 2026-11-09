"""Esquema y validación de lecturas etiquetadas por una persona.

Un CSV etiquetado tiene una fila por lectura y una columna `label` con `NORMAL`, `WARNING` o `FIRE`. Los valores de los
sensores se validan con el mismo esquema que `POST /predict` (`SensorReadingInput`), de modo que un dato que la API
rechazaría tampoco entra al entrenamiento ni a la evaluación.
"""
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from app.errors import group_validation_errors
from app.schemas import SensorReadingInput

LABELS = ("NORMAL", "WARNING", "FIRE")
REQUIRED_COLUMNS = ["node_id", "temperature", "humidity", "smoke", "gas", "flame", "timestamp", "label"]
# Columnas extra permitidas: ayudan a quien etiqueta, pero no se usan para entrenar.
OPTIONAL_COLUMNS = ["backend_risk_level", "notes"]
FEATURE_COLUMNS = ["temperature", "humidity", "smoke", "gas", "flame"]
MAX_REPORTED_PROBLEMS = 20

_TRUE = {"1", "true", "t", "yes", "si", "sí"}
_FALSE = {"0", "false", "f", "no"}


class LabeledDataError(ValueError):
    """El CSV no cumple el esquema; `problems` lista cada fallo con su número de fila (1 = primera fila de datos)."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        shown = problems[:MAX_REPORTED_PROBLEMS]
        extra = len(problems) - len(shown)
        suffix = f"\n... y {extra} problema(s) más" if extra > 0 else ""
        super().__init__("Datos etiquetados inválidos:\n- " + "\n- ".join(shown) + suffix)


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == ""


def _parse_flame(value: Any) -> Any:
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError("flame debe ser 0/1 o true/false")


def _row_problems(row: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """Devuelve (problemas, registro limpio); el registro queda vacío si hay algún problema."""
    problems: list[str] = []
    label = "" if _blank(row.get("label")) else str(row["label"]).strip().upper()
    if not label:
        problems.append("label vacío: falta etiquetar la lectura")
    elif label not in LABELS:
        problems.append(f"label «{row['label']}» no es NORMAL, WARNING ni FIRE")

    reading: dict[str, Any] = {"node_id": "" if _blank(row.get("node_id")) else str(row["node_id"]).strip()}
    for name in ("temperature", "humidity", "smoke"):
        if _blank(row.get(name)):
            problems.append(f"{name} vacío")
            reading[name] = 0  # Evita un segundo error de Pydantic sobre el mismo campo.
        else:
            reading[name] = row[name]
    reading["gas"] = None if _blank(row.get("gas")) else row["gas"]  # El gas es opcional: vacío equivale a 0.
    try:
        reading["flame"] = _parse_flame(row.get("flame"))
    except ValueError as error:
        problems.append(str(error))
        reading["flame"] = False
    if _blank(row.get("timestamp")):
        problems.append("timestamp vacío")
        reading["timestamp"] = None
    else:
        reading["timestamp"] = str(row["timestamp"]).strip()

    try:
        validated = SensorReadingInput.model_validate(reading)
    except ValidationError as error:
        for field, messages in group_validation_errors(error.errors()).items():
            problems.append(f"{field}: {'; '.join(messages)}")
        return problems, {}
    if problems:
        return problems, {}
    record = validated.model_dump()
    record["timestamp"] = record["timestamp"].isoformat()
    return [], {**record, "label": label}


def validate_labeled_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Devuelve el DataFrame limpio (tipos normalizados) o lanza `LabeledDataError` con todos los problemas."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise LabeledDataError([f"faltan las columnas {missing}"])
    if frame.empty:
        raise LabeledDataError(["el archivo no tiene filas"])

    problems: list[str] = []
    clean: list[dict[str, Any]] = []
    for position, row in enumerate(frame.to_dict("records"), start=1):
        row_problems, record = _row_problems(row)
        problems.extend(f"fila {position}: {text}" for text in row_problems)
        if record:
            clean.append(record)
    if problems:
        raise LabeledDataError(problems)

    result = pd.DataFrame(clean, columns=REQUIRED_COLUMNS)
    result["gas"] = result["gas"].astype(float)
    result["flame"] = result["flame"].astype(int)
    duplicated = result.duplicated(subset=["node_id", "timestamp"], keep=False)
    if duplicated.any():
        rows = [str(index + 1) for index in result.index[duplicated]]
        raise LabeledDataError([f"filas {', '.join(rows[:10])}: mismo node_id y timestamp repetidos"])
    return result


def load_labeled_csv(path: str | Path) -> pd.DataFrame:
    # Todo como texto: así se distingue «vacío» de 0 y se informa de cada celda mala en lugar de fallar al leer.
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    return validate_labeled_frame(frame)
