"""Ejecuta tres casos de demostración contra la función de predicción local."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.predictor import RiskPredictor

CASES = [
    {"node_id": "DEMO-NORMAL", "temperature": 22, "humidity": 70, "smoke": 12, "gas": 5, "flame": False},
    {"node_id": "DEMO-WARNING", "temperature": 33, "humidity": 28, "smoke": 90, "gas": 40, "flame": False},
    {"node_id": "DEMO-FIRE", "temperature": 48, "humidity": 10, "smoke": 320, "gas": 150, "flame": True},
    {"node_id": "DEMO-FLAME-ONLY", "temperature": 24, "humidity": 70, "smoke": 5, "gas": 0, "flame": True},
]


def main() -> None:
    predictor = RiskPredictor()
    for case in CASES:
        result = predictor.predict(case)
        print("-" * 68)
        print(f"Nodo: {result['node_id']}")
        print(f"Modelo: {result['model_prediction']} | Riesgo final (IA + reglas): {result['risk_level']}")
        print(f"Score de riesgo no-normal: {result['risk_probability_percent']:.2f}%")
        print("Probabilidades del modelo:", result["probabilities"])
        print("Regla aplicada:", result["safety_rules"]["rule"])
        print("Explicación:", result["safety_rules"]["explanation"])
    print("\nTodas las entradas son escenarios sintéticos de demostración, no datos de campo.")


if __name__ == "__main__":
    main()
