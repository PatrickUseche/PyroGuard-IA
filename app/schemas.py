from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

RiskLevel = Literal["NORMAL", "WARNING", "FIRE"]
# Versión de docs/CONTRATO_IA_BACKEND.md; subirla al cambiar campos, rangos o errores.
CONTRACT_VERSION = "1.0"


class SensorReadingInput(BaseModel):
    """Entrada alineada con la validación de PyroGuard-Backend (contrato v1.0)."""
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)

    node_id: str = Field(..., min_length=1, max_length=50, description="Identificador del nodo ESP32")
    temperature: float = Field(..., ge=-50, le=150, description="Temperatura en grados Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Humedad relativa porcentual")
    smoke: float = Field(..., ge=0, description="Lectura del sensor de humo")
    gas: float = Field(default=0.0, ge=0, description="Lectura opcional del sensor de gases; null equivale a 0.0")
    flame: bool = Field(..., description="Detección del sensor de llama (booleano o 0/1)")
    timestamp: datetime | None = Field(default=None, description="Fecha/hora ISO 8601 de la medición")

    @field_validator("gas", mode="before")
    @classmethod
    def gas_null_as_zero(cls, value: Any) -> Any:
        # El backend guarda gas como nullable cuando el nodo no tiene ese sensor.
        return 0.0 if value is None else value

    @field_validator("flame", mode="before")
    @classmethod
    def flame_bool_or_binary(cls, value: Any) -> bool:
        # Igual que JsonBooleanField del backend: sin coerción de strings como "yes" o "true".
        if type(value) is bool:
            return value
        if type(value) is int and value in (0, 1):
            return bool(value)
        raise ValueError("flame debe ser un booleano o 0/1")


class SafetyDecision(BaseModel):
    rule: str
    explanation: str
    fire_confirmations: list[str]
    warning_signals: list[str]


class PredictionResponse(BaseModel):
    status: Literal["ok"] = "ok"
    contract_version: str = CONTRACT_VERSION
    node_id: str
    timestamp: datetime
    model_version: str
    model_prediction: RiskLevel
    risk_level: RiskLevel
    # Probabilidad estimada de pertenecer a WARNING o FIRE; rango 0.0 a 1.0.
    risk_probability: float = Field(..., ge=0, le=1)
    risk_probability_percent: float = Field(..., ge=0, le=100)
    probabilities: dict[str, float]
    safety_rules: SafetyDecision
    synthetic_model: bool = True
    disclaimer: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None
    classes: list[str] = []


class LatencyMetrics(BaseModel):
    count: int
    average: float
    p95: float
    max: float


class MetricsResponse(BaseModel):
    """Contadores en memoria del proceso; se reinician con cada arranque."""
    predictions_total: int
    predictions_by_risk_level: dict[str, int]
    latency_ms: LatencyMetrics
