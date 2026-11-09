"""Importa lecturas del backend a un CSV para que una persona las etiquete.

Usa `GET /api/nodes/` y `GET /api/nodes/<id>/readings/` (paginado, máximo 500 por página). Esos endpoints piden sesión salvo
que el backend tenga `PUBLIC_READ_ACCESS=True`; un script no puede iniciar sesión con Google, así que se admite un token
de DRF en la cabecera `Authorization: Token <token>`.
"""
from typing import Any, Callable, Iterator

import pandas as pd

from app.labeled_data import OPTIONAL_COLUMNS, REQUIRED_COLUMNS

PAGE_SIZE = 500
TOTAL_COUNT_HEADER = "X-Total-Count"
IMPORT_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


class BackendImportError(RuntimeError):
    """El backend respondió algo que impide importar (sesión, nodo inexistente, formato inesperado)."""


def _headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Token {token}"} if token else {}


def _get(get: Callable[..., Any], url: str, token: str | None, params: dict[str, Any] | None = None):
    try:
        response = get(url, params=params, headers=_headers(token), timeout=(3.0, 20.0))
    except Exception as error:  # requests.RequestException y los dobles de prueba
        raise BackendImportError(f"No se pudo conectar con {url}: {type(error).__name__}") from error
    if response.status_code in (401, 403):
        raise BackendImportError(
            "El backend pide sesión para leer. Usa --token con un token de DRF o activa PUBLIC_READ_ACCESS=True en el backend."
        )
    if response.status_code == 404:
        raise BackendImportError(f"No existe {url}")
    if response.status_code != 200:
        raise BackendImportError(f"El backend respondió {response.status_code} en {url}")
    return response


def fetch_node_ids(base_url: str, get: Callable[..., Any], token: str | None = None) -> list[str]:
    body = _get(get, f"{base_url.rstrip('/')}/api/nodes/", token).json()
    nodes = body.get("results", []) if isinstance(body, dict) else body
    if not isinstance(nodes, list):
        raise BackendImportError("La lista de nodos no tiene el formato esperado.")
    return [str(node["node_id"]) for node in nodes if isinstance(node, dict) and node.get("node_id")]


def fetch_readings(
    base_url: str,
    node_id: str,
    get: Callable[..., Any],
    token: str | None = None,
    start: str | None = None,
    end: str | None = None,
    max_rows: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Recorre todas las páginas de un nodo, de la lectura más antigua a la más nueva."""
    url = f"{base_url.rstrip('/')}/api/nodes/{node_id}/readings/"
    offset = 0
    returned = 0
    while True:
        params: dict[str, Any] = {"limit": PAGE_SIZE, "offset": offset, "order": "asc"}
        if start:
            params["from"] = start
        if end:
            params["to"] = end
        response = _get(get, url, token, params)
        page = response.json()
        if not isinstance(page, list):
            raise BackendImportError(f"Las lecturas de {node_id} no tienen el formato esperado.")
        for reading in page:
            if max_rows is not None and returned >= max_rows:
                return
            returned += 1
            yield reading
        offset += len(page)
        total = response.headers.get(TOTAL_COUNT_HEADER)
        if not page or len(page) < PAGE_SIZE or (total is not None and offset >= int(total)):
            return


def to_labeling_frame(readings: list[dict[str, Any]]) -> pd.DataFrame:
    """Filas listas para etiquetar: `label` vacío y, como pista, el nivel que calculó el backend."""
    rows = [
        {
            "node_id": reading.get("node_id"),
            "temperature": reading.get("temperature"),
            "humidity": reading.get("humidity"),
            "smoke": reading.get("smoke"),
            "gas": reading.get("gas"),
            "flame": int(bool(reading.get("flame"))),
            "timestamp": reading.get("timestamp"),
            "label": "",
            "backend_risk_level": reading.get("risk_level") or "",
            "notes": "",
        }
        for reading in readings
    ]
    return pd.DataFrame(rows, columns=IMPORT_COLUMNS)
