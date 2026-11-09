import importlib.util
from pathlib import Path

import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from app.evaluation_set import TRAINING_SET_PATH
from app.model_comparison import PRODUCTION_MODEL, candidate_models, compare_models, production_forest, render_markdown


@pytest.fixture(scope="module")
def sample():
    data = pd.read_csv(TRAINING_SET_PATH)
    # 100 filas por clase: suficiente para 3 pliegues y rápido.
    return pd.concat([group.head(100) for _, group in data.groupby("label")])


def test_candidates_include_the_production_model_with_its_hyperparameters():
    models = candidate_models()
    assert PRODUCTION_MODEL in models and len(models) >= 4
    forest = production_forest()
    assert (forest.n_estimators, forest.min_samples_leaf, forest.class_weight) == (300, 2, "balanced")


def test_compare_models_scores_every_candidate_sorted_by_f1(sample):
    models = {"Arbol": DecisionTreeClassifier(random_state=1), PRODUCTION_MODEL: production_forest()}
    rows = compare_models(sample, folds=3, models=models)
    assert {row["model"] for row in rows} == {"Arbol", PRODUCTION_MODEL}
    assert rows == sorted(rows, key=lambda row: (-row["f1_macro_mean"], row["fit_seconds"]))
    for row in rows:
        assert 0 <= row["f1_macro_mean"] <= 1 and 0 <= row["accuracy_mean"] <= 1
        assert row["f1_macro_std"] >= 0 and row["fit_seconds"] >= 0


def test_comparison_is_reproducible(sample):
    models = {"Arbol": DecisionTreeClassifier(random_state=1)}
    first = compare_models(sample, folds=3, models=models)[0]
    second = compare_models(sample, folds=3, models={"Arbol": DecisionTreeClassifier(random_state=1)})[0]
    assert (first["f1_macro_mean"], first["accuracy_mean"]) == (second["f1_macro_mean"], second["accuracy_mean"])


def test_markdown_states_the_synthetic_limitation_and_marks_production():
    rows = [
        {"model": "RandomForest", "accuracy_mean": 0.98, "accuracy_std": 0.004, "f1_macro_mean": 0.98, "f1_macro_std": 0.004,
         "fit_seconds": 0.5, "predict_seconds": 0.05, "total_seconds": 3.0},
        {"model": "KNN", "accuracy_mean": 0.96, "accuracy_std": 0.006, "f1_macro_mean": 0.96, "f1_macro_std": 0.006,
         "fit_seconds": 0.01, "predict_seconds": 0.1, "total_seconds": 1.0},
    ]
    text = render_markdown(rows, 2400, "a" * 64, 5, "2026-11-09")
    assert "RandomForest (producción)" in text
    assert "sintéticos" in text and "no demuestra" in text
    assert "0.0200" in text  # diferencia entre mejor y peor
    assert "aaaaaaaaaaaa…" in text


def test_script_runs_and_writes_the_document(tmp_path, monkeypatch, sample):
    path = Path(__file__).resolve().parents[1] / "scripts" / "6_comparar_modelos.py"
    spec = importlib.util.spec_from_file_location("comparar_modelos", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "compare_models", lambda data, folds: compare_models(
        sample, folds=3, models={"Arbol": DecisionTreeClassifier(random_state=1)}))
    out = tmp_path / "comparacion.md"
    assert script.main(["--out", str(out)]) == 0
    assert "# Comparación de modelos" in out.read_text(encoding="utf-8")
