"""Informe de precisión, recall y F1 por clase sobre el conjunto de prueba etiquetado.

    python scripts/5_informe_metricas.py [--data data/lecturas_etiquetadas.csv] [--out reports/evaluation_labeled.json]

Evalúa dos cosas: el modelo solo y el resultado final del servicio (modelo + reglas de seguridad).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation_set import EVALUATION_DATA_TYPE, EVALUATION_SET_PATH, EvaluationSetError, load_evaluation_set  # noqa: E402
from app.metrics_report import build_report  # noqa: E402
from app.predictor import RiskPredictor  # noqa: E402

DEFAULT_OUT = ROOT / "reports" / "evaluation_labeled.json"


def format_table(title: str, metrics: dict) -> str:
    lines = [title, f"{'clase':10} {'precisión':>10} {'recall':>8} {'F1':>8} {'n':>6}"]
    for label, values in metrics["per_class"].items():
        lines.append(f"{label:10} {values['precision']:>10.3f} {values['recall']:>8.3f} {values['f1']:>8.3f} {values['support']:>6}")
    macro = metrics["macro_avg"]
    lines.append(f"{'macro':10} {macro['precision']:>10.3f} {macro['recall']:>8.3f} {macro['f1']:>8.3f}")
    lines.append(f"exactitud global: {metrics['accuracy']:.3f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=EVALUATION_SET_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    try:
        evaluation = load_evaluation_set(args.data)
    except EvaluationSetError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    report = build_report(evaluation, RiskPredictor(), EVALUATION_DATA_TYPE)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Lecturas evaluadas: {report['evaluation_rows']} ({EVALUATION_DATA_TYPE}) | modelo {report['model_version']}")
    print(format_table("\nModelo solo", report["model_only"]))
    print(format_table("\nResultado final (modelo + reglas)", report["final_with_safety_rules"]))
    for warning in report["warnings"]:
        print(f"AVISO: {warning}")
    print(f"\n{report['limitations']}\nInforme guardado en {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
