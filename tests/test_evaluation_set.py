import pandas as pd
import pytest

from app.evaluation_set import (
    EVALUATION_DATA_TYPE,
    EVALUATION_SET_PATH,
    TRAINING_SET_PATH,
    EvaluationSetError,
    ensure_valid_evaluation_set,
    find_overlap,
    load_evaluation_set,
)
from app.labeled_data import REQUIRED_COLUMNS


def labeled(rows):
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def reading(node="SIM-001", minute=0, temperature=22.0, humidity=70.0, smoke=10.0, gas=0.0, flame=0, label="NORMAL"):
    return [node, temperature, humidity, smoke, gas, flame, f"2026-11-01T10:{minute:02d}:00+00:00", label]


@pytest.fixture(scope="module")
def training():
    return pd.read_csv(TRAINING_SET_PATH)


def test_paths_live_in_the_data_folder_and_are_different():
    assert EVALUATION_SET_PATH != TRAINING_SET_PATH
    assert EVALUATION_SET_PATH.parent == TRAINING_SET_PATH.parent
    assert EVALUATION_DATA_TYPE == "simulated"


def test_clean_evaluation_set_passes(training):
    frame = labeled([reading(minute=0), reading(minute=1, temperature=47.3, smoke=311.7, flame=1, label="FIRE")])
    assert ensure_valid_evaluation_set(frame, training) is frame


def test_synthetic_rows_are_rejected(training):
    frame = labeled([reading(node="SYNTH-003")])
    with pytest.raises(EvaluationSetError, match="sintéticas"):
        ensure_valid_evaluation_set(frame, training)


def test_rows_copied_from_training_are_leakage(training):
    copied = training.iloc[0]
    frame = labeled([
        reading(minute=0),
        ["SIM-009", copied["temperature"], copied["humidity"], copied["smoke"], copied["gas"], int(copied["flame"]),
         "2026-11-01T10:05:00+00:00", "WARNING"],
    ])
    assert find_overlap(frame, training) == 1
    with pytest.raises(EvaluationSetError, match="fuga de datos"):
        ensure_valid_evaluation_set(frame, training)


def test_overlap_compares_at_two_decimals():
    train = pd.DataFrame([{"temperature": 22.001, "humidity": 70.0, "smoke": 10.0, "gas": 0.0, "flame": 0}])
    near = labeled([reading(temperature=22.004)])
    far = labeled([reading(temperature=22.2)])
    assert find_overlap(near, train) == 1
    assert find_overlap(far, train) == 0


def test_validation_without_training_data_only_checks_synthetic_rows():
    assert len(ensure_valid_evaluation_set(labeled([reading()]), None)) == 1


def test_missing_file_explains_how_to_create_it(tmp_path):
    with pytest.raises(EvaluationSetError, match="4_importar_lecturas"):
        load_evaluation_set(tmp_path / "no_existe.csv")


def test_invalid_labeled_file_is_reported_as_evaluation_error(tmp_path):
    path = tmp_path / "malo.csv"
    path.write_text("node_id,temperature,humidity,smoke,gas,flame,timestamp,label\nA,22,70,5,0,0,2026-11-01T10:00:00Z,\n", encoding="utf-8")
    with pytest.raises(EvaluationSetError, match="falta etiquetar"):
        load_evaluation_set(path)


def test_load_evaluation_set_reads_and_checks_a_file(tmp_path):
    path = tmp_path / "bien.csv"
    path.write_text(
        "node_id,temperature,humidity,smoke,gas,flame,timestamp,label\n"
        "SIM-001,22.37,70.11,10.4,0,0,2026-11-01T10:00:00Z,NORMAL\n"
        "SIM-001,47.31,9.87,311.6,149.2,1,2026-11-01T10:01:00Z,FIRE\n",
        encoding="utf-8",
    )
    assert list(load_evaluation_set(path)["label"]) == ["NORMAL", "FIRE"]


def test_the_repository_does_not_ship_the_dataset_as_its_own_test_set():
    # Si existe el conjunto de prueba real del equipo, debe pasar las mismas comprobaciones.
    if EVALUATION_SET_PATH.exists():
        assert len(load_evaluation_set()) > 0
