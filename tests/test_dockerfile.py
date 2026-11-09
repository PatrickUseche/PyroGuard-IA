"""Comprobaciones estáticas del Dockerfile (en CI no se construye la imagen)."""
from pathlib import Path

DOCKERFILE = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(encoding="utf-8")
LINES = [line.strip() for line in DOCKERFILE.splitlines()]


def test_container_runs_as_an_unprivileged_user():
    assert any(line.startswith("USER ") and line != "USER root" for line in LINES)
    assert "useradd" in DOCKERFILE
    # USER debe ir después de instalar dependencias y copiar los archivos.
    assert max(i for i, line in enumerate(LINES) if line.startswith("COPY")) < LINES.index("USER appuser")


def test_container_has_a_healthcheck_on_the_public_health_endpoint():
    assert any(line.startswith("HEALTHCHECK") for line in LINES)
    assert "/health" in DOCKERFILE


def test_container_uses_the_deployment_python_and_the_port_variable():
    assert LINES[0] == "FROM python:3.12-slim"
    assert "${PORT:-8001}" in DOCKERFILE


def test_runtime_files_are_copied():
    for folder in ("app", "models", "data"):
        assert f"COPY {folder} ./{folder}" in LINES
