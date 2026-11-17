import requests

from app.deploy_check import SAMPLE_READING, check_health, check_predict, verify_deployment

BASE = "https://ia.test"
HEALTHY = {"status": "ok", "model_loaded": True, "model_version": "pyroguard-rf-1.1.0"}


class FakeResponse:
    def __init__(self, status_code, body=None):
        self.status_code, self._body = status_code, body or {}

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, get=None, post=None, error=None):
        self._get, self._post, self._error = get, post, error
        self.calls = []

    def get(self, url, timeout):
        self.calls.append(("GET", url, None))
        if self._error:
            raise self._error
        return self._get

    def post(self, url, json, headers, timeout):
        self.calls.append(("POST", url, headers))
        if self._error:
            raise self._error
        return self._post


def test_healthy_service_passes():
    result = check_health(BASE, FakeSession(get=FakeResponse(200, HEALTHY)))
    assert result.ok and "pyroguard-rf-1.1.0" in result.detail


def test_degraded_service_fails():
    result = check_health(BASE, FakeSession(get=FakeResponse(200, {"status": "degraded", "model_loaded": False})))
    assert not result.ok


def test_unreachable_service_fails_without_raising():
    session = FakeSession(error=requests.ConnectionError("sin red"))
    assert [item.ok for item in verify_deployment(BASE, None, session)] == [False, False]


def test_without_key_predict_must_be_protected():
    assert check_predict(BASE, None, FakeSession(post=FakeResponse(401))).ok
    assert not check_predict(BASE, None, FakeSession(post=FakeResponse(200))).ok


def test_with_key_predict_must_answer_with_the_current_contract():
    session = FakeSession(post=FakeResponse(200, {"contract_version": "1.1", "risk_level": "NORMAL"}))
    result = check_predict(BASE, "clave", session)
    assert result.ok and result.detail == "risk_level=NORMAL"
    method, url, headers = session.calls[0]
    assert (method, url, headers) == ("POST", f"{BASE}/predict", {"X-AI-API-Key": "clave"})


def test_with_key_an_old_contract_fails():
    session = FakeSession(post=FakeResponse(200, {"contract_version": "1.0"}))
    assert not check_predict(BASE, "clave", session).ok


def test_with_a_wrong_key_fails():
    assert not check_predict(BASE, "mala", FakeSession(post=FakeResponse(401))).ok


def test_trailing_slash_is_removed():
    session = FakeSession(get=FakeResponse(200, HEALTHY), post=FakeResponse(401))
    verify_deployment(BASE + "/", None, session)
    assert [url for _, url, _ in session.calls] == [f"{BASE}/health", f"{BASE}/predict"]


def test_sample_reading_is_accepted_by_the_api():
    from fastapi.testclient import TestClient

    from app.main import app

    assert TestClient(app).post("/predict", json=SAMPLE_READING).status_code == 200
