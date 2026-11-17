"""Pruebas de la validación del dataset en scripts/2_entrenar_modelo.py (sin entrenar el bosque)."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def trainer():
    spec = importlib.util.spec_from_file_location("entrenar_modelo", ROOT / "scripts" / "2_entrenar_modelo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


ROW = {"temperature": 22.0, "humidity": 60.0, "smoke": 10.0, "gas": 0.0, "flame": 0}


def three_classes():
    return [dict(ROW, label=label) for label in ("NORMAL", "WARNING", "FIRE")]


def test_versioned_dataset_is_valid(trainer):
    data, X, y = trainer.load_training_data(trainer.DATA_PATH)
    assert list(X.columns) == trainer.FEATURES
    assert len(X) == len(y) == len(data)
    assert X["flame"].dtype.kind == "i"


def test_missing_file_stops_with_a_hint(trainer, tmp_path):
    with pytest.raises(SystemExit, match="1_generar_dataset"):
        trainer.load_training_data(tmp_path / "no_existe.csv")


def test_missing_columns_are_reported(trainer, tmp_path):
    rows = [{key: value for key, value in row.items() if key != "gas"} for row in three_classes()]
    with pytest.raises(SystemExit, match="gas"):
        trainer.load_training_data(write_csv(tmp_path / "sin_gas.csv", rows))


def test_empty_values_are_rejected(trainer, tmp_path):
    rows = three_classes()
    rows[0]["smoke"] = None
    with pytest.raises(SystemExit, match="vacíos"):
        trainer.load_training_data(write_csv(tmp_path / "vacios.csv", rows))


def test_all_three_classes_are_required(trainer, tmp_path):
    rows = [dict(ROW, label="NORMAL"), dict(ROW, label="FIRE")]
    with pytest.raises(SystemExit, match="tres clases"):
        trainer.load_training_data(write_csv(tmp_path / "dos_clases.csv", rows))
