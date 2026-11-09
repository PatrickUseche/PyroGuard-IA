import pandas as pd
import pytest

from app.labeled_data import REQUIRED_COLUMNS, LabeledDataError, load_labeled_csv, validate_labeled_frame


def row(**overrides):
    base = {
        "node_id": "NODE-001", "temperature": "22.5", "humidity": "70", "smoke": "12", "gas": "3",
        "flame": "0", "timestamp": "2026-11-01T10:00:00Z", "label": "NORMAL",
    }
    return {**base, **overrides}


def frame(*rows):
    return pd.DataFrame(list(rows))


def problems_of(*rows):
    with pytest.raises(LabeledDataError) as caught:
        validate_labeled_frame(frame(*rows))
    return caught.value.problems


def test_valid_rows_are_normalized():
    clean = validate_labeled_frame(frame(row(label=" fire ", flame="True", gas="")))
    record = clean.iloc[0]
    assert list(clean.columns) == REQUIRED_COLUMNS
    assert record["label"] == "FIRE"
    assert record["flame"] == 1
    assert record["gas"] == 0.0
    assert record["timestamp"] == "2026-11-01T10:00:00+00:00"


@pytest.mark.parametrize("flame, expected", [("0", 0), ("1", 1), ("false", 0), ("TRUE", 1), ("no", 0), ("sí", 1)])
def test_flame_accepts_binary_and_boolean_words(flame, expected):
    assert validate_labeled_frame(frame(row(flame=flame))).iloc[0]["flame"] == expected


def test_extra_columns_are_ignored():
    clean = validate_labeled_frame(frame({**row(), "backend_risk_level": "WARNING", "notes": "humo de cocina"}))
    assert "notes" not in clean.columns


def test_missing_columns_are_reported():
    with pytest.raises(LabeledDataError, match="faltan las columnas"):
        validate_labeled_frame(pd.DataFrame([{"node_id": "A", "label": "NORMAL"}]))


def test_empty_frame_is_rejected():
    with pytest.raises(LabeledDataError, match="no tiene filas"):
        validate_labeled_frame(pd.DataFrame(columns=REQUIRED_COLUMNS))


@pytest.mark.parametrize("label", ["", "  ", "MAYBE", "fuego", "1"])
def test_unlabeled_or_unknown_labels_are_rejected(label):
    assert any("label" in text for text in problems_of(row(label=label)))


@pytest.mark.parametrize("field, value, fragment", [
    ("temperature", "151", "temperature"),
    ("temperature", "-51", "temperature"),
    ("humidity", "101", "humidity"),
    ("smoke", "-1", "smoke"),
    ("gas", "-1", "gas"),
    ("gas", "abc", "gas"),
    ("temperature", "", "temperature vacío"),
    ("smoke", "", "smoke vacío"),
    ("flame", "2", "flame"),
    ("flame", "", "flame"),
    ("timestamp", "", "timestamp vacío"),
    ("timestamp", "ayer", "timestamp"),
    ("node_id", "", "node_id"),
    ("node_id", "x" * 51, "node_id"),
    ("temperature", "nan", "temperature"),
])
def test_invalid_values_are_reported_with_the_row_number(field, value, fragment):
    problems = problems_of(row(), row(**{field: value}))
    assert problems and all(text.startswith("fila 2:") for text in problems)
    assert any(fragment in text for text in problems)


def test_every_problem_of_every_row_is_listed():
    problems = problems_of(row(label="X"), row(smoke="", flame="9"))
    assert any(text.startswith("fila 1:") for text in problems)
    assert sum(text.startswith("fila 2:") for text in problems) >= 2


def test_repeated_node_and_timestamp_are_rejected():
    problems = problems_of(row(), row(label="WARNING"))
    assert "repetidos" in problems[0]


def test_same_timestamp_on_different_nodes_is_fine():
    assert len(validate_labeled_frame(frame(row(), row(node_id="NODE-002")))) == 2


def test_error_message_truncates_long_lists():
    rows = [row(node_id=f"N-{index}", label="X") for index in range(30)]
    with pytest.raises(LabeledDataError) as caught:
        validate_labeled_frame(frame(*rows))
    assert len(caught.value.problems) == 30
    assert "10 problema(s) más" in str(caught.value)


def test_load_labeled_csv_reads_a_file_with_bom_and_blank_gas(tmp_path):
    path = tmp_path / "datos.csv"
    path.write_text(
        "node_id,temperature,humidity,smoke,gas,flame,timestamp,label\n"
        "A,22,70,5,,0,2026-11-01T10:00:00Z,NORMAL\n"
        "A,40,20,300,150,1,2026-11-01T10:01:00Z,FIRE\n",
        encoding="utf-8-sig",
    )
    clean = load_labeled_csv(path)
    assert list(clean["label"]) == ["NORMAL", "FIRE"]
    assert list(clean["flame"]) == [0, 1]
