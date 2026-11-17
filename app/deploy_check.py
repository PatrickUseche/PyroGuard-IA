"""Comprobaciones de un despliegue de PyroGuard-IA (lo usa scripts/7_verificar_despliegue.py)."""
from dataclasses import dataclass

import requests

from app.schemas import CONTRACT_VERSION

TIMEOUT_SECONDS = 30.0  # holgado: la primera petición a Vercel o Render puede pagar un arranque en frío
SAMPLE_READING = {"node_id": "CHECK-001", "temperature": 22.0, "humidity": 70.0, "smoke": 10.0, "gas": 0.0, "flame": False}


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def check_health(base_url: str, session=requests) -> CheckResult:
    try:
        response = session.get(f"{base_url}/health", timeout=TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        return CheckResult("health", False, f"sin respuesta ({type(exc).__name__})")
    if response.status_code != 200:
        return CheckResult("health", False, f"HTTP {response.status_code}")
    body = response.json()
    if body.get("status") != "ok" or not body.get("model_loaded"):
        return CheckResult("health", False, f"status={body.get('status')} model_loaded={body.get('model_loaded')}")
    return CheckResult("health", True, f"modelo {body.get('model_version')}")


def check_predict(base_url: str, api_key: str | None, session=requests) -> CheckResult:
    headers = {"X-AI-API-Key": api_key} if api_key else {}
    try:
        response = session.post(f"{base_url}/predict", json=SAMPLE_READING, headers=headers, timeout=TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        return CheckResult("predict", False, f"sin respuesta ({type(exc).__name__})")
    if not api_key:
        # Sin clave solo se puede comprobar que /predict no está abierto en producción.
        if response.status_code == 401:
            return CheckResult("predict", True, "protegido con X-AI-API-Key (401 sin clave)")
        return CheckResult("predict", False, f"sin clave se esperaba 401 y respondió HTTP {response.status_code}")
    if response.status_code != 200:
        return CheckResult("predict", False, f"HTTP {response.status_code}")
    body = response.json()
    if body.get("contract_version") != CONTRACT_VERSION:
        return CheckResult("predict", False, f"contrato {body.get('contract_version')}, se esperaba {CONTRACT_VERSION}")
    return CheckResult("predict", True, f"risk_level={body.get('risk_level')}")


def verify_deployment(base_url: str, api_key: str | None = None, session=requests) -> list[CheckResult]:
    base_url = base_url.rstrip("/")
    return [check_health(base_url, session), check_predict(base_url, api_key, session)]
