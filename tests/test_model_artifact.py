"""El modelo versionado, su meta.json y el dataset deben contar la misma historia."""
import joblib
import pytest

from app.model_meta import DEFAULT_META_PATH, ROOT_DIR, dataset_sha256, load_meta
from app.predictor import DEFAULT_MODEL_PATH, FEATURE_COLUMNS

DATASET_PATH = ROOT_DIR / "data" / "dataset_sintetico.csv"


@pytest.fixture(scope="module")
def meta():
    return load_meta()


@pytest.fixture(scope="module")
def artifact():
    return joblib.load(DEFAULT_MODEL_PATH)


def test_meta_file_is_versioned_next_to_the_model():
    assert DEFAULT_META_PATH.parent == DEFAULT_MODEL_PATH.parent
    assert DEFAULT_META_PATH.exists()


def test_meta_hash_matches_the_dataset_in_the_repo(meta):
    # Si falla, el CSV cambió sin reentrenar: ejecuta scripts/2_entrenar_modelo.py y sube el meta.json.
    assert meta["dataset_sha256"] == dataset_sha256(DATASET_PATH)


def test_meta_and_artifact_describe_the_same_training(meta, artifact):
    assert artifact["model_version"] == meta["model_version"]
    assert artifact["dataset_sha256"] == meta["dataset_sha256"]
    assert artifact["dataset_rows"] == meta["dataset_rows"]
    assert artifact["trained_at_utc"] == meta["trained_at_utc"]
    assert artifact["scikit_learn_version"] == meta["scikit_learn_version"]
    assert artifact["training_data_type"] == meta["dataset_type"]


def test_meta_matches_the_features_and_rows(meta):
    assert meta["features"] == FEATURE_COLUMNS
    assert meta["train_rows"] + meta["test_rows"] == meta["dataset_rows"]
    with DATASET_PATH.open(encoding="utf-8") as handle:
        assert sum(1 for _ in handle) - 1 == meta["dataset_rows"]


def test_dataset_hash_ignores_line_endings(tmp_path):
    unix = tmp_path / "unix.csv"
    windows = tmp_path / "windows.csv"
    unix.write_bytes(b"a,b\n1,2\n")
    windows.write_bytes(b"a,b\r\n1,2\r\n")
    assert dataset_sha256(unix) == dataset_sha256(windows)


def test_dataset_hash_changes_with_the_content(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text("a,b\n1,2\n", encoding="utf-8")
    second.write_text("a,b\n1,3\n", encoding="utf-8")
    assert dataset_sha256(first) != dataset_sha256(second)
