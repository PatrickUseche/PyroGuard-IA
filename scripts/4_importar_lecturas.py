"""Importa lecturas del backend a un CSV por etiquetar (columna `label` vacía).

Ejemplo:
    python scripts/4_importar_lecturas.py --base-url http://127.0.0.1:8000 --all-nodes --out data/por_etiquetar.csv

Después se rellena `label` a mano con NORMAL, WARNING o FIRE y se valida al generar el informe (scripts/5_informe_metricas.py).
Las lecturas del simulador NO son datos de campo: el informe lo declara como `simulated`.
"""
import argparse
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.backend_import import BackendImportError, fetch_node_ids, fetch_readings, to_labeling_frame  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "por_etiquetar.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.getenv("PYROGUARD_BACKEND_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--token", default=os.getenv("PYROGUARD_API_TOKEN"), help="Token de DRF (o variable PYROGUARD_API_TOKEN)")
    parser.add_argument("--node", action="append", default=[], help="Nodo a importar; se puede repetir")
    parser.add_argument("--all-nodes", action="store_true", help="Importa todos los nodos que liste el backend")
    parser.add_argument("--from", dest="start", help="Fecha inicial ISO 8601")
    parser.add_argument("--to", dest="end", help="Fecha final ISO 8601")
    parser.add_argument("--max-rows", type=int, help="Máximo de lecturas por nodo")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if not args.node and not args.all_nodes:
        parser.error("indica --node ID (repetible) o --all-nodes")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        nodes = args.node or fetch_node_ids(args.base_url, requests.get, args.token)
        readings = []
        for node_id in nodes:
            before = len(readings)
            readings.extend(
                fetch_readings(args.base_url, node_id, requests.get, args.token, args.start, args.end, args.max_rows)
            )
            print(f"{node_id}: {len(readings) - before} lecturas")
    except BackendImportError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    frame = to_labeling_frame(readings)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False, encoding="utf-8")
    print(f"{len(frame)} lecturas guardadas en {args.out}. Rellena la columna label (NORMAL, WARNING o FIRE).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
