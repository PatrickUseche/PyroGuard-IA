# Contrato de integración PyroGuard-IA ↔ PyroGuard-Backend (v1.0)

Este documento define el contrato HTTP entre el servicio de inferencia (`PyroGuard-IA`, FastAPI) y el
backend (`PyroGuard-Backend`, Django REST Framework). Es la referencia única para ambos repositorios:
cualquier cambio de campos, rangos o códigos de error debe reflejarse aquí y subir `contract_version`.

Los ejemplos ejecutables de `docs/contract/examples/` se verifican en las pruebas de ambos repositorios.

## 1. Flujo

```
ESP32 / simulador ──POST /api/readings/──▶ Backend
                                            ├─ 1. valida y guarda SensorReading (siempre)
                                            ├─ 2. POST {AI_SERVICE_URL}/predict  (timeouts de conexión y lectura)
                                            ├─ 3. evaluate_risk(reglas + resultado IA) ─▶ RiskAssessment
                                            └─ 4. maybe_create_alert(...)          ─▶ Alert
```

Principios:

1. **La lectura nunca se pierde por la IA.** Si la IA falla, el backend guarda la lectura y evalúa solo con reglas.
2. **El backend tiene la decisión final.** La IA puede *elevar* el nivel de riesgo, nunca reducirlo
   (comportamiento ya documentado en `api/services/risk_rules.py`).
3. **La validación del backend es la puerta de entrada.** Toda lectura aceptada por el backend debe ser
   aceptada por la IA; la IA no puede ser más estricta que el backend.

## 2. Petición: `POST /predict`

Cabeceras:

| Cabecera | Obligatoria | Valor |
|---|---|---|
| `Content-Type` | sí | `application/json` |
| `X-AI-API-Key` | si la IA define `AI_API_KEY` | Mismo valor configurado en ambos servicios |

Cuerpo:

| Campo | Tipo | Rango v1.0 | Obligatorio | Notas |
|---|---|---|---|---|
| `node_id` | string | 1–50 caracteres | sí | Igual a `Node.node_id` |
| `temperature` | number | −50 a 150 °C | sí | Igual a la validación del backend |
| `humidity` | number | 0 a 100 % | sí | |
| `smoke` | number | ≥ 0 | sí | Sin tope superior (el backend no lo tiene) |
| `gas` | number \| null | ≥ 0 | no | `null` o ausente ⇒ la IA usa `0.0` |
| `flame` | boolean | `true`/`false` (se aceptan `0`/`1`) | sí | Strings como `"yes"` se rechazan |
| `timestamp` | string ISO 8601 | — | no | El backend siempre lo envía con zona horaria |

Ejemplo:

```json
{
  "node_id": "NODE-001",
  "temperature": 48.0,
  "humidity": 10.0,
  "smoke": 320.0,
  "gas": 150.0,
  "flame": true,
  "timestamp": "2026-10-13T22:00:00-05:00"
}
```

## 3. Respuesta exitosa: `200 OK`

```json
{
  "status": "ok",
  "contract_version": "1.0",
  "node_id": "NODE-001",
  "timestamp": "2026-10-13T22:00:00-05:00",
  "model_version": "pyroguard-rf-1.0.0",
  "model_prediction": "FIRE",
  "risk_level": "FIRE",
  "risk_probability": 1.0,
  "risk_probability_percent": 100.0,
  "probabilities": {"NORMAL": 0.0, "WARNING": 0.0, "FIRE": 1.0},
  "safety_rules": {
    "rule": "FIRE_CORROBORATED",
    "explanation": "Se detectaron varias señales concordantes de alto riesgo. ...",
    "fire_confirmations": ["sensor de llama activo junto con humo >= 80"],
    "warning_signals": ["temperatura >= 30 °C", "humedad <= 35 %"]
  },
  "synthetic_model": true,
  "disclaimer": "Prototipo académico entrenado con datos sintéticos. ..."
}
```

Semántica de los campos que consume el backend:

| Campo IA | Significado | Destino en backend |
|---|---|---|
| `risk_probability` | `P(WARNING) + P(FIRE)`, rango 0–1 | `RiskAssessment.ai_probability` (entrada de `evaluate_risk`) |
| `risk_level` | Nivel tras las reglas de seguridad de la IA | `RiskAssessment.ai_risk_level` (nuevo, trazabilidad) |
| `probabilities` | Probabilidad por clase | `RiskAssessment.ai_probabilities` (nuevo, JSON) |
| `model_version` | Versión del artefacto `.joblib` | `RiskAssessment.ai_model_version` (ya existe) |
| `safety_rules.explanation` | Razón legible | Se añade a `rule_reasons` con prefijo `IA:` |
| `contract_version` | Versión de este contrato | El cliente rechaza versiones mayores incompatibles (`2.x`) |

Los umbrales de `evaluate_risk` (0.5 ⇒ WARNING, 0.85 + señal de sensor ⇒ FIRE) se aplican sobre `risk_probability`.

## 4. Errores

Formato uniforme (alineado con `api/exceptions.py` del backend):

```json
{"status": "error", "code": "VALIDATION_ERROR", "errors": {"temperature": ["..."]}}
```

| HTTP | `code` | Causa | Acción del backend |
|---|---|---|---|
| 401 | `UNAUTHORIZED` | `X-AI-API-Key` ausente o inválida | Registrar `ERROR` (mala configuración); fallback a reglas |
| 422 | `VALIDATION_ERROR` | Payload fuera de contrato | Registrar `WARNING`; fallback a reglas; **sin reintento** |
| 503 | `MODEL_UNAVAILABLE` | Modelo no cargado | Fallback a reglas |
| 500 | `PREDICTION_FAILED` | Error interno de inferencia | Fallback a reglas |
| 500 | `INTERNAL_ERROR` | Excepción no prevista | Fallback a reglas |
| — | — | Timeout / conexión rechazada | Fallback a reglas |
| 200 con cuerpo inválido | — | Respuesta no cumple este contrato | Tratar como error de contrato; fallback a reglas |

En `VALIDATION_ERROR`, `errors` agrupa mensajes por campo (`{"campo": ["mensaje", ...]}`); el JSON mal
formado se reporta en `non_field_errors`. El resto de errores usa `{"detail": "mensaje"}`. Las respuestas
de error nunca incluyen el valor recibido ni detalles internos de la excepción.

`ai_status` registrado en `RiskAssessment`: `OK`, `DISABLED` (sin `AI_SERVICE_URL`), `UNAVAILABLE`
(timeout, conexión, 5xx), `REJECTED` (401/422) o `INVALID_RESPONSE`.

## 5. Configuración

| Variable | Servicio | Ejemplo | Descripción |
|---|---|---|---|
| `AI_API_KEY` | IA y backend | `cambia-esta-clave` | Vacía en IA ⇒ endpoint abierto (solo desarrollo local); con `APP_ENV=production` la IA no arranca sin ella |
| `APP_ENV` | IA | `production` | Con el valor `production` (sin distinguir mayúsculas) la clave pasa a ser obligatoria al arrancar |
| `AI_SERVICE_URL` | backend | `http://127.0.0.1:8001` | Vacía ⇒ IA deshabilitada (`ai_status=DISABLED`) |
| `AI_CONNECT_TIMEOUT` | backend | `3.0` | Segundos máximos para establecer la conexión; detecta rápido un servicio caído |
| `AI_READ_TIMEOUT` | backend | `10.0` | Segundos máximos de espera de la respuesta de inferencia una vez conectado |

El backend envía `timeout=(AI_CONNECT_TIMEOUT, AI_READ_TIMEOUT)`, de modo que cada límite se aplica a su
fase. Con los valores por defecto, una lectura puede esperar hasta ~13 s a la IA antes de responder al
nodo; el timeout HTTP del ESP32 debe ser mayor para que no reintente lecturas ya guardadas.
`AI_SERVICE_TIMEOUT` ya no se usa.

## 6. Salud

`GET /health` (sin autenticación) devuelve `{"status": "ok"|"degraded", "model_loaded": bool, "model_version": str|null, "classes": [...]}`.
El backend puede exponerlo de forma opcional para monitoreo; no se consulta en cada lectura.

## 7. Brechas detectadas en la revisión (2026-10-22) y resolución

Verificadas enviando a `/predict` lecturas que el backend acepta:

| # | Caso | Backend | IA actual | Resolución en v1.0 |
|---|---|---|---|---|
| 1 | `gas: null` | Válido (campo nullable) | 422 | IA acepta `null` ⇒ `0.0` |
| 2 | `temperature: 120` o `-45` | Válido (−50..150) | 422 (−40..100) | IA amplía a −50..150 |
| 3 | `smoke: 20000` | Válido (sin máximo) | 422 (≤ 10000) | IA elimina el tope superior |
| 4 | `flame: "yes"` | Rechazado | 200 (coerción laxa) | IA acepta solo bool o 0/1 |
| 5 | Errores | `{"status","errors"}` | `{"detail"}` de FastAPI | IA adopta formato uniforme con `code` |
| 6 | Versionado | — | Sin versión de contrato | IA añade `contract_version` |
| 7 | `backend_integration/` en IA | Ya tiene `RiskAssessment` y `Alert` | Borradores para un backend antiguo | Se elimina; este documento lo reemplaza |
| 8 | `ingestion.py` | `TODO(IA)` con `ai_probability=None` | — | Cliente HTTP + fallback (sección 4) |

Notas adicionales:

- Los valores fuera del rango de entrenamiento sintético (p. ej. 140 °C) se clasifican igualmente; el
  Random Forest no extrapola, por lo que las reglas del backend siguen siendo la salvaguarda principal.
- La llamada HTTP a la IA debe hacerse **fuera** de la transacción de base de datos que guarda la lectura,
  para no mantener bloqueos abiertos durante el timeout.
- `pandas==2.2.3` no tenía wheel para Python 3.14; se actualizó a `pandas==2.3.3`, compatible con
  Python 3.12 (Render) y 3.14.
