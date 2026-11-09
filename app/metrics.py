"""Contadores en memoria del servicio: predicciones por nivel de riesgo y latencia.

El estado es por proceso (Render corre un solo worker) y se pierde al reiniciar; no sustituye a un sistema de
monitoreo externo. Todas las operaciones están protegidas con un candado porque FastAPI atiende `def` en hilos.
"""
import math
from collections import deque
from threading import Lock

RISK_LEVELS = ("NORMAL", "WARNING", "FIRE")
LATENCY_WINDOW = 500  # Últimas N latencias usadas para el percentil 95.


class ServiceMetrics:
    def __init__(self, window: int = LATENCY_WINDOW):
        self._lock = Lock()
        self._window = window
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._by_risk_level = {level: 0 for level in RISK_LEVELS}
            self._latencies_ms: deque[float] = deque(maxlen=self._window)
            self._latency_count = 0
            self._latency_sum_ms = 0.0
            self._latency_max_ms = 0.0

    def record_prediction(self, risk_level: str, seconds: float) -> None:
        elapsed_ms = max(0.0, seconds * 1000)
        with self._lock:
            self._by_risk_level[risk_level] = self._by_risk_level.get(risk_level, 0) + 1
            self._latencies_ms.append(elapsed_ms)
            self._latency_count += 1
            self._latency_sum_ms += elapsed_ms
            self._latency_max_ms = max(self._latency_max_ms, elapsed_ms)

    def snapshot(self) -> dict:
        with self._lock:
            ordered = sorted(self._latencies_ms)
            count = self._latency_count
            p95 = ordered[math.ceil(len(ordered) * 0.95) - 1] if ordered else 0.0
            return {
                "predictions_total": sum(self._by_risk_level.values()),
                "predictions_by_risk_level": dict(self._by_risk_level),
                "latency_ms": {
                    "count": count,
                    "average": round(self._latency_sum_ms / count, 3) if count else 0.0,
                    "p95": round(p95, 3),
                    "max": round(self._latency_max_ms, 3),
                },
            }


metrics = ServiceMetrics()
