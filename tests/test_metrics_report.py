import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from app.labeled_data import LABELS
from app.metrics_report import MIN_RELIABLE_SUPPORT, build_report, classification_metrics, reliability_warnings
from app.predictor import RiskPredictor


def test_perfect_predictions_give_ones():
    truth = ["NORMAL", "WARNING", "FIRE", "FIRE"]
    metrics = classification_metrics(truth, truth)
    assert metrics["accuracy"] == 1.0
    assert all(values["f1"] == 1.0 for values in metrics["per_class"].values())
    assert metrics["confusion_matrix"]["rows_are_real_columns_are_predicted"] == [[1, 0, 0], [0, 1, 0], [0, 0, 2]]


def test_precision_recall_and_f1_are_computed_per_class():
    truth = ["FIRE", "FIRE", "FIRE", "FIRE", "NORMAL", "NORMAL"]
    predicted = ["FIRE", "FIRE", "NORMAL", "NORMAL", "FIRE", "NORMAL"]
    fire = classification_metrics(truth, predicted)["per_class"]["FIRE"]
    # 2 aciertos de 3 predichos FIRE; 2 de 4 FIRE reales detectados.
    assert fire["precision"] == pytest.approx(2 / 3, abs=1e-6)
    assert fire["recall"] == 0.5
    assert fire["f1"] == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5), abs=1e-6)
    assert fire["support"] == 4


def test_missing_class_is_zero_and_excluded_from_macro_average():
    truth = ["NORMAL", "NORMAL", "FIRE"]
    metrics = classification_metrics(truth, truth)
    assert metrics["per_class"]["WARNING"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
    assert metrics["macro_avg"]["f1"] == 1.0


def test_empty_input_does_not_crash():
    metrics = classification_metrics([], [])
    assert metrics["accuracy"] == 0.0 and metrics["macro_avg"]["f1"] == 0.0


def test_warnings_flag_missing_and_small_classes():
    metrics = classification_metrics(["NORMAL"] * 40 + ["FIRE"] * 5, ["NORMAL"] * 45)
    warnings = reliability_warnings(metrics["per_class"])
    assert any("WARNING" in text and "No hay" in text for text in warnings)
    assert any("FIRE" in text and f"< {MIN_RELIABLE_SUPPORT}" in text for text in warnings)
    assert not any("NORMAL" in text for text in warnings)


@pytest.fixture(scope="module")
def evaluation():
    rows = [
        ["SIM-001", 22, 70, 10, 0, 0, "2026-11-01T10:00:00+00:00", "NORMAL"],
        ["SIM-001", 36.4, 25.3, 71.2, 20.1, 0, "2026-11-01T10:01:00+00:00", "WARNING"],
        ["SIM-001", 24.2, 70.1, 5.3, 0.4, 1, "2026-11-01T10:02:00+00:00", "WARNING"],
        ["SIM-001", 48.2, 10.4, 320.1, 150.2, 1, "2026-11-01T10:03:00+00:00", "FIRE"],
    ]
    return pd.DataFrame(rows, columns=["node_id", "temperature", "humidity", "smoke", "gas", "flame", "timestamp", "label"])


def test_report_evaluates_model_alone_and_final_result(evaluation):
    report = build_report(evaluation, RiskPredictor())
    assert report["evaluation_rows"] == 4
    assert report["evaluation_dataset_type"] == "simulated"
    assert report["class_counts"] == {"NORMAL": 1, "WARNING": 2, "FIRE": 1}
    assert report["training_dataset_sha256"] and len(report["training_dataset_sha256"]) == 64
    # La llama aislada: el modelo dice NORMAL pero las reglas la elevan a WARNING.
    assert report["final_with_safety_rules"]["accuracy"] == 1.0
    assert report["model_only"]["accuracy"] < report["final_with_safety_rules"]["accuracy"]
    assert "no son datos de campo" in report["limitations"]
    assert set(report["final_with_safety_rules"]["per_class"]) == set(LABELS)
    assert report["warnings"]


def test_report_is_json_serializable(evaluation):
    assert json.loads(json.dumps(build_report(evaluation, RiskPredictor())))["model_version"]


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "5_informe_metricas.py"
    spec = importlib.util.spec_from_file_location("informe_metricas", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_writes_the_report_and_prints_tables(tmp_path, capsys):
    data = tmp_path / "etiquetadas.csv"
    data.write_text(
        "node_id,temperature,humidity,smoke,gas,flame,timestamp,label\n"
        "SIM-001,22.37,70.11,10.4,0,0,2026-11-01T10:00:00Z,NORMAL\n"
        "SIM-001,48.21,10.37,320.1,150.2,1,2026-11-01T10:03:00Z,FIRE\n",
        encoding="utf-8",
    )
    out = tmp_path / "informe.json"
    assert load_script().main(["--data", str(data), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "Modelo solo" in printed and "Resultado final" in printed and "AVISO" in printed
    assert json.loads(out.read_text(encoding="utf-8"))["evaluation_rows"] == 2


def test_script_fails_cleanly_without_data(tmp_path, capsys):
    assert load_script().main(["--data", str(tmp_path / "nada.csv"), "--out", str(tmp_path / "x.json")]) == 1
    assert "Error:" in capsys.readouterr().err
    assert not (tmp_path / "x.json").exists()
