"""Compara clasificadores con validación cruzada y escribe docs/COMPARACION_MODELOS.md.

    python scripts/6_comparar_modelos.py [--folds 5]
"""
import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.model_comparison import compare_models, render_markdown  # noqa: E402
from app.model_meta import dataset_sha256  # noqa: E402

DATA_PATH = ROOT / "data" / "dataset_sintetico.csv"
OUT_PATH = ROOT / "docs" / "COMPARACION_MODELOS.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)
    data = pd.read_csv(DATA_PATH)
    rows = compare_models(data, folds=args.folds)
    for row in rows:
        print(f"{row['model']:22} F1 macro {row['f1_macro_mean']:.4f} ± {row['f1_macro_std']:.4f} | exactitud {row['accuracy_mean']:.4f}")
    args.out.write_text(
        render_markdown(rows, len(data), dataset_sha256(DATA_PATH), args.folds, date.today().isoformat()), encoding="utf-8"
    )
    print(f"Documento escrito en {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
