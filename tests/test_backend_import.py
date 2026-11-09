import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from app import backend_import
from app.backend_import import PAGE_SIZE, BackendImportError, fetch_node_ids, fetch_readings, to_labeling_frame
from app.labeled_data import REQUIRED_COLUMNS, load_labeled_csv

BASE = "http://backend.test"


class FakeResponse:
    def __init__(self, body, status_code=200, headers=None):
        self._body, self.status_code, self.headers = body, status_code, headers or {}

    def json(self):
        return self._body


class FakeBackend:
    """Imita /api/nodes/ y /api/nodes/<id>/readings/ con paginación y X-Total-Count."""

    def __init__(self, readings_by_node, status=200, node_body=None):
        self.readings_by_node = readings_by_node
        self.status = status
        self.node_body = node_body
        self.calls = []

    def __call__(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params or {}, "headers": headers or {}})
        if self.status != 200:
            return FakeResponse({}, self.status)
        if url.endswith("/api/nodes/"):
            nodes = self.node_body if self.node_body is not None else [{"node_id": node} for node in self.readings_by_node]
            return FakeResponse(nodes)
        node_id = url.split("/api/nodes/")[1].split("/")[0]
        if node_id not in self.readings_by_node:
            return FakeResponse({}, 404)
        rows = self.readings_by_node[node_id]
        offset, limit = params["offset"], params["limit"]
        return FakeResponse(rows[offset:offset + limit], headers={"X-Total-Count": str(len(rows))})


def make_readings(node_id, count):
    return [
        {"node_id": node_id, "temperature": 20.0 + index % 5, "humidity": 60.0, "smoke": float(index), "gas": None,
         "flame": index % 2 == 0, "timestamp": f"2026-11-01T10:{index // 60:02d}:{index % 60:02d}Z",
         "risk_level": "NORMAL", "ai": None}
        for index in range(count)
    ]


def test_fetch_readings_walks_every_page_in_ascending_order():
    backend = FakeBackend({"N-1": make_readings("N-1", PAGE_SIZE * 2 + 37)})
    readings = list(fetch_readings(BASE, "N-1", backend))
    assert len(readings) == PAGE_SIZE * 2 + 37
    assert [call["params"]["offset"] for call in backend.calls] == [0, PAGE_SIZE, PAGE_SIZE * 2]
    assert all(call["params"]["order"] == "asc" for call in backend.calls)


def test_fetch_readings_stops_when_the_total_is_reached_on_a_full_last_page():
    backend = FakeBackend({"N-1": make_readings("N-1", PAGE_SIZE)})
    assert len(list(fetch_readings(BASE, "N-1", backend))) == PAGE_SIZE
    assert len(backend.calls) == 1


def test_fetch_readings_respects_max_rows_and_date_filters():
    backend = FakeBackend({"N-1": make_readings("N-1", 50)})
    readings = list(fetch_readings(BASE, "N-1", backend, start="2026-11-01T00:00:00Z", end="2026-11-02T00:00:00Z", max_rows=10))
    assert len(readings) == 10
    assert backend.calls[0]["params"]["from"] == "2026-11-01T00:00:00Z"
    assert backend.calls[0]["params"]["to"] == "2026-11-02T00:00:00Z"


def test_token_is_sent_as_drf_authorization_header():
    backend = FakeBackend({"N-1": make_readings("N-1", 1)})
    list(fetch_readings(BASE, "N-1", backend, token="abc123"))
    assert backend.calls[0]["headers"] == {"Authorization": "Token abc123"}


def test_no_token_means_no_authorization_header():
    backend = FakeBackend({"N-1": make_readings("N-1", 1)})
    list(fetch_readings(BASE, "N-1", backend))
    assert backend.calls[0]["headers"] == {}


@pytest.mark.parametrize("status", [401, 403])
def test_session_required_gives_an_actionable_message(status):
    with pytest.raises(BackendImportError, match="PUBLIC_READ_ACCESS"):
        list(fetch_readings(BASE, "N-1", FakeBackend({}, status=status)))


def test_unknown_node_and_server_errors_are_reported():
    with pytest.raises(BackendImportError, match="No existe"):
        list(fetch_readings(BASE, "N-X", FakeBackend({"N-1": []})))
    with pytest.raises(BackendImportError, match="500"):
        list(fetch_readings(BASE, "N-1", FakeBackend({}, status=500)))


def test_connection_failures_are_wrapped():
    def broken(*_args, **_kwargs):
        raise TimeoutError("lento")

    with pytest.raises(BackendImportError, match="No se pudo conectar"):
        fetch_node_ids(BASE, broken)


@pytest.mark.parametrize("body", [[{"node_id": "A"}, {"node_id": "B"}], {"results": [{"node_id": "A"}, {"node_id": "B"}]}])
def test_node_list_accepts_plain_and_paginated_formats(body):
    assert fetch_node_ids(BASE, FakeBackend({}, node_body=body)) == ["A", "B"]


def test_node_list_with_unexpected_format_is_rejected():
    with pytest.raises(BackendImportError, match="formato"):
        fetch_node_ids(BASE, FakeBackend({}, node_body={"results": "no"}))


def test_labeling_frame_has_empty_labels_and_a_hint_column():
    frame = to_labeling_frame(make_readings("N-1", 3))
    assert list(frame.columns)[:8] == REQUIRED_COLUMNS
    assert (frame["label"] == "").all()
    assert list(frame["backend_risk_level"]) == ["NORMAL"] * 3
    assert list(frame["flame"]) == [1, 0, 1]


def test_labeling_frame_of_nothing_keeps_the_columns():
    assert list(to_labeling_frame([]).columns)[:8] == REQUIRED_COLUMNS


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "4_importar_lecturas.py"
    spec = importlib.util.spec_from_file_location("importar_lecturas", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_writes_a_csv_that_validates_once_labeled(tmp_path, monkeypatch, capsys):
    script = load_script()
    backend = FakeBackend({"N-1": make_readings("N-1", 3), "N-2": make_readings("N-2", 2)})
    monkeypatch.setattr(script.requests, "get", backend)
    out = tmp_path / "por_etiquetar.csv"
    assert script.main(["--base-url", BASE, "--all-nodes", "--out", str(out)]) == 0
    assert "5 lecturas guardadas" in capsys.readouterr().out

    frame = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert len(frame) == 5 and (frame["label"] == "").all()
    frame["label"] = "NORMAL"
    frame.to_csv(out, index=False)
    assert len(load_labeled_csv(out)) == 5


def test_cli_returns_error_code_when_the_backend_asks_for_a_session(tmp_path, monkeypatch, capsys):
    script = load_script()
    monkeypatch.setattr(script.requests, "get", FakeBackend({}, status=401))
    assert script.main(["--base-url", BASE, "--node", "N-1", "--out", str(tmp_path / "x.csv")]) == 1
    assert "PUBLIC_READ_ACCESS" in capsys.readouterr().err
    assert not (tmp_path / "x.csv").exists()


def test_cli_requires_a_node_or_all_nodes():
    with pytest.raises(SystemExit):
        load_script().parse_args(["--base-url", BASE])


def test_module_reuses_the_labeled_data_columns():
    assert backend_import.IMPORT_COLUMNS[:8] == REQUIRED_COLUMNS
