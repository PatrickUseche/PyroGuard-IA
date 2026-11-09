import hmac
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from fastapi import Depends, FastAPI, Header, HTTPException, status
from app.errors import logger, register_error_handlers
from app.predictor import RiskPredictor
from app.schemas import HealthResponse, PredictionResponse, SensorReadingInput

@asynccontextmanager
async def lifespan(_app: FastAPI):
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
    # Sin clave configurada, se permite uso local; en despliegue público se recomienda definirla.
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
    try:
        return predictor.predict(reading.model_dump())
    except (ValueError, KeyError) as exc:
        logger.exception("Falló la predicción para el nodo %s", reading.node_id)
        raise HTTPException(status_code=500, detail="No se pudo calcular la predicción.") from exc
