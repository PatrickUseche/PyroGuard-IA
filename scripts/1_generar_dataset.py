"""Genera un dataset sintético balanceado para desarrollo; no representa datos de campo."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "dataset_sintetico.csv"
SEED = 20261013
N_PER_CLASS = 800


def sample_scenario(rng: np.random.Generator, label: str) -> dict:
    # Distribuciones parcialmente solapadas para evitar que cada clase dependa de un solo umbral.
    if label == "NORMAL":
        temperature = rng.uniform(12, 33)
        humidity = rng.uniform(38, 95)
        smoke = rng.uniform(0, 90)
        gas = rng.uniform(0, 38)
        flame = False
    elif label == "WARNING":
        temperature = rng.uniform(24, 43)
        humidity = rng.uniform(18, 62)
        smoke = rng.uniform(25, 260)
        gas = rng.uniform(0, 140)
        flame = bool(rng.random() < 0.025)
    else:
        temperature = rng.uniform(31, 58)
        humidity = rng.uniform(5, 42)
        smoke = rng.uniform(95, 900)
        gas = rng.uniform(15, 500)
        flame = bool(rng.random() < 0.68)

    # Variabilidad aleatoria pequeña que simula ruido de medición.
    temperature = float(np.clip(temperature + rng.normal(0, 1.2), -40, 100))
    humidity = float(np.clip(humidity + rng.normal(0, 2.0), 0, 100))
    smoke = float(max(0, smoke + rng.normal(0, max(2, smoke * 0.035))))
    gas = float(max(0, gas + rng.normal(0, max(1, gas * 0.04))))
    return {
        "node_id": f"SYNTH-{rng.integers(1, 9):03d}",
        "temperature": round(temperature, 2),
        "humidity": round(humidity, 2),
        "smoke": round(smoke, 2),
        "gas": round(gas, 2),
        "flame": int(flame),
        "timestamp": pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(seconds=int(rng.integers(0, 30 * 24 * 3600))),
        "label": label,
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    labels = [label for label in ("NORMAL", "WARNING", "FIRE") for _ in range(N_PER_CLASS)]
    rng.shuffle(labels)
    rows = [sample_scenario(rng, label) for label in labels]
    frame = pd.DataFrame(rows)
    frame["timestamp"] = frame["timestamp"].map(lambda value: value.isoformat())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    print(f"Dataset sintético generado: {OUT}")
    print(f"Registros: {len(frame)}")
    print("Distribución:")
    print(frame["label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
