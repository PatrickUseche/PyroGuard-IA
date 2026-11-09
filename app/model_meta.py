"""Metadatos del modelo entrenado: qué dataset lo generó y cómo se evaluó."""
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_META_PATH = ROOT_DIR / "models" / "modelo_riesgo.meta.json"


def dataset_sha256(path: str | Path) -> str:
    """SHA-256 del CSV con saltos de línea normalizados, para que no cambie entre Windows y Linux."""
    content = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def load_meta(path: str | Path = DEFAULT_META_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
