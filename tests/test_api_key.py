"""Protección con X-AI-API-Key: qué rutas la exigen y cómo se compara la clave."""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
KEY = "clave-de-prueba-larga"
READING = {"node_id": "K-1", "temperature": 22, "humidity": 70, "smoke": 10, "gas": 0, "flame": False}


@pytest.fixture
def configured_key(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", KEY)
    return KEY


@pytest.mark.parametrize("path", ["/", "/health"])
def test_public_routes_stay_open_and_do_not_leak_the_key(configured_key, path):
    response = client.get(path)
    assert response.status_code == 200
    assert configured_key not in response.text


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [("get", "/metrics", None), ("post", "/predict/batch", {"readings": [READING]}), ("post", "/predict", READING)],
)
def test_protected_routes_reject_a_wrong_key(configured_key, method, path, body):
    kwargs = {"headers": {"X-AI-API-Key": "otra-clave"}}
    if body is not None:
        kwargs["json"] = body
    response = getattr(client, method)(path, **kwargs)
    assert response.status_code == 401
    assert response.json()["code"] == "UNAUTHORIZED"
    assert configured_key not in response.text


def test_configured_key_is_compared_without_surrounding_spaces(monkeypatch):
    # Un valor pegado en el panel de Vercel o Render con espacios no debe dejar fuera al backend.
    monkeypatch.setenv("AI_API_KEY", f"  {KEY}\n")
    assert client.post("/predict", json=READING, headers={"X-AI-API-Key": KEY}).status_code == 200
