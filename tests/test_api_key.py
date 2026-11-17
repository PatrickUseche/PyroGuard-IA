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
