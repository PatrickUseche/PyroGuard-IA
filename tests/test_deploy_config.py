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


def render_env_vars():
    """Lee los pares key/value de `envVars` de render.yaml sin depender de PyYAML."""
    lines = (ROOT / "render.yaml").read_text(encoding="utf-8").splitlines()
    env, current = {}, None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- key:"):
            current = stripped.split(":", 1)[1].strip()
            env[current] = {}
        elif current and ":" in stripped and not stripped.startswith("#"):
            name, value = (part.strip() for part in stripped.split(":", 1))
            env[current][name] = value
    return env


def test_render_forces_production_and_keeps_the_key_secret():
    env = render_env_vars()
    assert env["APP_ENV"]["value"] == "production"
    # sync: false -> Render pide el valor en el panel; la clave nunca se versiona.
    assert env["AI_API_KEY"] == {"sync": "false"}


def test_render_uses_the_health_endpoint_and_uvicorn():
    text = (ROOT / "render.yaml").read_text(encoding="utf-8")
    assert "healthCheckPath: /health" in text
    assert "uvicorn app.main:app --host 0.0.0.0 --port $PORT" in text
