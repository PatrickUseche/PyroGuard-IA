"""Verifica un despliegue de PyroGuard-IA: /health con el modelo cargado y /predict protegido o respondiendo.

Ejemplos:
    python scripts/7_verificar_despliegue.py --url https://pyro-guard-ia.vercel.app
    python scripts/7_verificar_despliegue.py --url https://pyro-guard-ia.vercel.app --key "$AI_API_KEY"

Sin clave comprueba que /predict responde 401; con clave, que clasifica una lectura de ejemplo.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.deploy_check import verify_deployment  # noqa: E402

DEFAULT_URL = "https://pyro-guard-ia.vercel.app"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.getenv("AI_SERVICE_URL", DEFAULT_URL), help="URL base del servicio")
    parser.add_argument("--key", default=os.getenv("AI_API_KEY") or None, help="Clave X-AI-API-Key (o variable AI_API_KEY)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results = verify_deployment(args.url, args.key)
    for result in results:
        print(f"[{'OK' if result.ok else 'FALLA'}] {result.name}: {result.detail}")
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
