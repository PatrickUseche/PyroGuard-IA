"""Pruebas de scripts/1_generar_dataset.py: el dataset versionado debe poder regenerarse igual."""
import importlib.util
from pathlib import Path

import pytest

from app.model_meta import dataset_sha256, load_meta

ROOT = Path(__file__).resolve().parents[1]


def load_generator():
    spec = importlib.util.spec_from_file_location("generar_dataset", ROOT / "scripts" / "1_generar_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    generator = load_generator()
    generator.OUT = tmp_path_factory.mktemp("dataset") / "dataset_sintetico.csv"
    generator.main()
    return generator.OUT


def test_generator_reproduces_the_versioned_dataset(generated):
    # Mismo SHA-256 que el CSV del repo y que el meta.json del modelo: el modelo se puede reentrenar con los mismos datos.
    assert dataset_sha256(generated) == dataset_sha256(ROOT / "data" / "dataset_sintetico.csv")
    assert dataset_sha256(generated) == load_meta()["dataset_sha256"]


def test_generated_dataset_is_balanced_and_inside_contract_ranges(generated):
    import pandas as pd

    generator = load_generator()
    frame = pd.read_csv(generated)
    assert len(frame) == 3 * generator.N_PER_CLASS
    assert frame["label"].value_counts().to_dict() == {
        label: generator.N_PER_CLASS for label in ("NORMAL", "WARNING", "FIRE")
    }
    # Rangos de SensorReadingInput (contrato v1.1): todo lo generado debe aceptarse en /predict.
    assert frame["temperature"].between(-50, 150).all()
    assert frame["humidity"].between(0, 100).all()
    assert (frame["smoke"] >= 0).all() and (frame["gas"] >= 0).all()
    assert set(frame["flame"].unique()) <= {0, 1}
    assert frame["node_id"].str.len().between(1, 50).all()
    assert not frame.isna().any().any()
