import hmac
import os
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import ValidationError
from app.errors import group_validation_errors, logger, register_error_handlers
from app.metrics import metrics
from app.predictor import RiskPredictor
from app.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    MetricsResponse,
    PredictionResponse,
    SensorReadingInput,
)

def is_production() -> bool:
    return os.getenv("APP_ENV", "").strip().lower() == "production"


def ensure_api_key_in_production() -> None:
    """En producción /predict no puede quedar abierto: sin AI_API_KEY el servicio no arranca."""
    if is_production() and not os.getenv("AI_API_KEY", "").strip():
        raise RuntimeError("APP_ENV=production exige definir AI_API_KEY; el servicio no arranca sin ella.")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_api_key_in_production()
    # Carga el modelo al arrancar para que la primera petición no pague ese tiempo (el backend espera pocos segundos).
    # Si falla, el servicio arranca igual: /health queda en «degraded» y /predict responde 503.
    try:
        get_predictor()
    except Exception as exc:
        logger.error("No se pudo cargar el modelo al arrancar: %s", type(exc).__name__)
    yield


app = FastAPI(
    title="PyroGuard IA API",
    description=(
        "Servicio local de Machine Learning para clasificar el riesgo ambiental en NORMAL, WARNING o FIRE. "
        "El modelo es Random Forest y el artefacto actual se entrenó con datos sintéticos."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
register_error_handlers(app)


@lru_cache(maxsize=1)
def get_predictor() -> RiskPredictor:
    return RiskPredictor()


def verify_api_key(x_ai_api_key: str | None = Header(default=None, alias="X-AI-API-Key")) -> None:
    expected = os.getenv("AI_API_KEY", "").strip()
    # Sin clave configurada se permite el uso local; con APP_ENV=production el servicio ni siquiera arranca así.
    if not expected:
        return
    if not x_ai_api_key or not hmac.compare_digest(x_ai_api_key, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key ausente o inválida")


@app.get("/", tags=["health"])
def root() -> dict:
    return {"service": "PyroGuard IA API", "docs": "/docs", "health": "/health", "prediction": "POST /predict"}


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    try:
        predictor = get_predictor()
        return HealthResponse(status="ok", model_loaded=True, model_version=predictor.model_version, classes=predictor.classes)
    except Exception:
        return HealthResponse(status="degraded", model_loaded=False, model_version=None, classes=[])


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"], dependencies=[Depends(verify_api_key)])
def predict(reading: SensorReadingInput) -> dict:
    try:
        predictor = get_predictor()
    except (FileNotFoundError, ValueError) as exc:
        logger.error("No se pudo cargar el modelo: %s", exc)
        raise HTTPException(status_code=503, detail="Modelo no disponible. Ejecuta scripts/2_entrenar_modelo.py.") from exc
    started = time.perf_counter()
    try:
        result = predictor.predict(reading.model_dump())
        metrics.record_prediction(result["risk_level"], time.perf_counter() - started)
        return result
    except (ValueError, KeyError) as exc:
        logger.exception("Falló la predicción para el nodo %s", reading.node_id)
        raise HTTPException(status_code=500, detail="No se pudo calcular la predicción.") from exc


@app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"], dependencies=[Depends(verify_api_key)])
def predict_batch(batch: BatchRequest) -> dict:
    valid: list[tuple[int, dict]] = []
    results: list[dict | None] = [None] * len(batch.readings)
    for index, raw in enumerate(batch.readings):
        try:
            valid.append((index, SensorReadingInput.model_validate(raw).model_dump()))
        except ValidationError as exc:
            results[index] = {"status": "error", "index": index, "errors": group_validation_errors(exc.errors())}
    if valid:
        try:
            predictor = get_predictor()
        except (FileNotFoundError, ValueError) as exc:
            logger.error("No se pudo cargar el modelo: %s", exc)
            raise HTTPException(status_code=503, detail="Modelo no disponible. Ejecuta scripts/2_entrenar_modelo.py.") from exc
        started = time.perf_counter()
        try:
            predictions = predictor.predict_many([reading for _, reading in valid])
        except (ValueError, KeyError) as exc:
            logger.exception("Falló la predicción del lote de %d lecturas", len(valid))
            raise HTTPException(status_code=500, detail="No se pudo calcular la predicción.") from exc
        per_reading = (time.perf_counter() - started) / len(valid)
        for (index, _), prediction in zip(valid, predictions):
            results[index] = prediction
            metrics.record_prediction(prediction["risk_level"], per_reading)
    failed = len(batch.readings) - len(valid)
    return {"count": len(results), "succeeded": len(valid), "failed": failed, "results": results}


@app.get("/metrics", response_model=MetricsResponse, tags=["health"], dependencies=[Depends(verify_api_key)])
def service_metrics() -> dict:
    return metrics.snapshot()
