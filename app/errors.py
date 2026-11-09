"""Respuestas de error uniformes según la sección 4 de docs/CONTRATO_IA_BACKEND.md."""
import logging
from typing import Any
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("pyroguard.api")

ERROR_CODES = {
    401: "UNAUTHORIZED",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    422: "VALIDATION_ERROR",
    500: "PREDICTION_FAILED",
    503: "MODEL_UNAVAILABLE",
}
REQUEST_LOCATIONS = {"body", "query", "path", "header"}


def error_response(status_code: int, code: str, errors: dict[str, Any], headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"status": "error", "code": code, "errors": errors}, headers=headers)


def _field_name(error: dict[str, Any]) -> str:
    location = list(error.get("loc", ()))
    if location and location[0] in REQUEST_LOCATIONS:
        location = location[1:]
    # JSON mal formado reporta la posición del carácter, no un campo.
    if not location or error.get("type") == "json_invalid":
        return "non_field_errors"
    return ".".join(str(part) for part in location)


def group_validation_errors(raw_errors: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Agrupa los errores de Pydantic por campo: `{"campo": ["mensaje", ...]}`."""
    # No se devuelve el valor recibido: puede no ser serializable (NaN) y no aporta al cliente.
    errors: dict[str, list[str]] = {}
    for error in raw_errors:
        errors.setdefault(_field_name(error), []).append(str(error.get("msg", "Valor inválido")))
    return errors


async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    return error_response(422, ERROR_CODES[422], group_validation_errors(exc.errors()))


async def http_exception_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = ERROR_CODES.get(exc.status_code, "HTTP_ERROR")
    return error_response(exc.status_code, code, {"detail": str(exc.detail)}, headers=getattr(exc, "headers", None))


async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    logger.error("Excepción inesperada en la API. Tipo: %s", type(exc).__name__, exc_info=exc)
    return error_response(500, "INTERNAL_ERROR", {"detail": "Error interno del servidor."})


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
