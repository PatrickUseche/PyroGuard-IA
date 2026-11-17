"""La configuración de despliegue no debe dejar fuera el modelo ni la clave."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_vercel_bundles_the_model_and_skips_what_is_not_needed():
    config = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
    function = config["functions"]["app/main.py"]
    # Sin el .joblib la función arranca en «degraded» y /predict responde 503.
    assert "models/**" in function["includeFiles"]
    excluded = function["excludeFiles"]
    for folder in ("data", "docs", "tests", "scripts"):
        assert f"{folder}/**" in excluded
    assert "models" not in excluded
    assert (ROOT / "app" / "main.py").exists()
