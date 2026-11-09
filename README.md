# PyroGuard-IA — Servicio de Machine Learning

Servicio de inferencia de IA para **PyroGuard / Forest Guardian**. Implementa un clasificador **Random Forest** entrenado localmente con un dataset sintético y lo expone mediante una API REST propia con FastAPI.

> **Alcance real:** es un prototipo académico reproducible. El dataset es sintético; no se debe presentar el modelo como detector certificado de incendios reales ni usarlo como único mecanismo de seguridad.

## Qué incluye

- Dataset simulado alineado con el contrato: `node_id`, `temperature`, `humidity`, `smoke`, `gas`, `flame`, `timestamp`, `label`.
- Entrenamiento Random Forest, partición estratificada 80/20 y métricas `accuracy`, `precision`, `recall`, `F1-score` y matriz de confusión.
- Artefacto versionado `models/modelo_riesgo.joblib` para que la API inicie sin entrenar en cada arranque.
- API REST con `GET /health`, `GET /` y `POST /predict`; documentación Swagger automática en `/docs`.
- Reglas de seguridad RF08 que combinan la salida del modelo con señales de sensores y no convierten una lectura aislada de llama en `FIRE`.
- Pruebas automatizadas, Dockerfile y configuración base para Render.
- Contrato de integración con PyroGuard-Backend en `docs/CONTRATO_IA_BACKEND.md`.

## Requisitos

Python 3.12 (versión de despliegue en Render) hasta 3.14. No requiere una API externa de OpenAI, Gemini u otro proveedor; Random Forest se ejecuta en el propio servicio.

## Inicio rápido en Windows

1. Descomprime este repositorio dentro de tu carpeta local de `PyroGuard-IA`.
2. Haz doble clic en `setup_windows.cmd`.
3. Ejecuta `start_windows.cmd`.
4. Abre `http://127.0.0.1:8001/docs`.

También puedes ejecutar manualmente desde esta raíz:

```bash
python -m venv .venv
# Windows CMD: .venv\Scripts\activate.bat
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

## API de predicción

`POST http://127.0.0.1:8001/predict`

Ejemplo de entrada compatible con el contrato del documento:

```json
{
  "node_id": "NODE-001",
  "temperature": 48,
  "humidity": 10,
  "smoke": 320,
  "gas": 150,
  "flame": true,
  "timestamp": "2026-10-13T22:00:00-05:00"
}
```

`gas` es opcional: si falta o llega como `null`, el servicio utiliza `0.0`. `node_id`, temperatura, humedad, humo y llama son obligatorios. Los rangos aceptados coinciden con la validación del backend (temperatura de −50 a 150 °C, humedad de 0 a 100 %, humo y gas ≥ 0, `flame` booleano o 0/1); el detalle está en el contrato.

La respuesta incluye:

- `model_prediction`: clase elegida directamente por Random Forest.
- `probabilities`: probabilidades estimadas por clase (no necesariamente calibradas).
- `risk_probability`: suma `P(WARNING) + P(FIRE)`, expresada entre `0.0` y `1.0`.
- `risk_probability_percent`: la misma puntuación en porcentaje.
- `risk_level`: clasificación final tras las reglas de corroboración.
- `safety_rules`: regla usada y señales que motivaron la salida.
- `model_version`, `timestamp` y advertencia del alcance sintético.
- `contract_version`: versión del contrato IA-Backend que cumple la respuesta.

Ejemplo de respuesta abreviada:

```json
{
  "status": "ok",
  "contract_version": "1.0",
  "node_id": "NODE-001",
  "model_version": "pyroguard-rf-1.0.0",
  "model_prediction": "FIRE",
  "risk_level": "FIRE",
  "risk_probability": 0.98,
  "risk_probability_percent": 98.0,
  "probabilities": {"NORMAL": 0.01, "WARNING": 0.01, "FIRE": 0.98},
  "safety_rules": {
    "rule": "FIRE_CORROBORATED",
    "explanation": "...",
    "fire_confirmations": ["..."],
    "warning_signals": ["..."]
  },
  "synthetic_model": true,
  "disclaimer": "..."
}
```

Las probabilidades son ilustrativas; la respuesta concreta se calcula con el modelo incluido.

Los errores usan un formato uniforme con un `code` estable (`VALIDATION_ERROR`, `UNAUTHORIZED`, `MODEL_UNAVAILABLE`, `PREDICTION_FAILED`, `INTERNAL_ERROR`, entre otros):

```json
{"status": "error", "code": "VALIDATION_ERROR", "errors": {"temperature": ["Input should be less than or equal to 150"]}}
```

## Reentrenar

```bash
python scripts/1_generar_dataset.py
python scripts/2_entrenar_modelo.py
python -m pytest -q
```

El dataset tiene 2.400 filas sintéticas, 800 por clase. El script usa un split estratificado 80/20 y escribe el modelo, el reporte de clasificación y la matriz de confusión.

Para probar casos prediseñados:

```bash
python scripts/3_probar_predicciones.py
```

## Etiquetado y métricas

Para medir el modelo con lecturas del backend etiquetadas a mano: `scripts/4_importar_lecturas.py` exporta un CSV por
etiquetar y `scripts/5_informe_metricas.py` calcula precisión, recall y F1 por clase. El flujo, el esquema del CSV y
los límites de esos datos (vienen del simulador) están en [docs/ETIQUETADO.md](docs/ETIQUETADO.md).

## Variables de entorno

Copia `.env.example` a `.env` si quieres documentar variables localmente (el código no carga `.env` automáticamente para la IA). Las variables deben definirse en el entorno del proceso o plataforma:

- `AI_API_KEY`: opcional en local. Si se configura, `POST /predict` requiere el header `X-AI-API-Key` con la misma clave.
- `APP_ENV`: con `production` el servicio **no arranca** si falta `AI_API_KEY`. El `render.yaml` ya la define así.

La versión que devuelve la API se lee de los metadatos guardados dentro del artefacto entrenado, para conservar la trazabilidad del modelo.

En producción, define `AI_API_KEY` en la configuración de secretos del host; nunca subas una clave real al repositorio.

## Integración con Django

Consulta `docs/CONTRATO_IA_BACKEND.md`. Django guarda la lectura, solicita la predicción a esta API y combina el resultado con sus propias reglas en `RiskAssessment`. El backend conserva la decisión final: la IA puede elevar el nivel de riesgo, pero nunca reducirlo. Si la IA no responde, la lectura se conserva y se evalúa solo con las reglas del backend.

Arquitectura:

```text
ESP32 / simulador
      ↓ JSON del contrato
PyroGuard-Backend (Django)
      ↓ POST /predict
PyroGuard-IA (FastAPI + Random Forest + reglas RF08)
      ↓ risk_level / risk_probability / model_version
Django combina IA + reglas y guarda RiskAssessment / Alert
      ↓ GET /api/readings/
Angular Dashboard
```

## Relación con el documento de requisitos

- **RF06:** modelo entrenado que calcula clase y puntuación de riesgo, con versión documentada.
- **RF07:** clasificación `NORMAL`, `WARNING` o `FIRE` y probabilidades estimadas.
- **RF08:** reglas de corroboración que combinan IA y sensores; una lectura atípica aislada no marca por sí sola `FIRE`.
- **RF15:** dataset y casos de prueba sin ESP32 real.
- **RNF07/RNF08/RNF14:** versión del modelo, estructura modular, documentación y pruebas automatizadas.

## Despliegue

### Render

1. Crea un Web Service apuntando al repositorio `PyroGuard-IA`.
2. Usa `render.yaml` (o define Build Command `pip install -r requirements.txt`, Start Command `uvicorn app.main:app --host 0.0.0.0 --port $PORT`, Health Check Path `/health`).
3. Define una variable secreta `AI_API_KEY` si el servicio queda público.
4. Verifica `https://TU-SERVICIO.onrender.com/health` y la documentación en `/docs`.
5. Configura en el backend `AI_SERVICE_URL=https://TU-SERVICIO.onrender.com` y `AI_API_KEY` con el mismo valor secreto.

No escribas el dominio de ejemplo como si fuera real: usa la URL que la plataforma genere al desplegar.

### Vercel (gratis, sin tarjeta)

`vercel.json` incluye el modelo (`models/**`) en la función y excluye lo que no hace falta en ejecución. Vercel detecta `app/main.py` como entrada de FastAPI.

1. En Vercel: **Add New → Project**, importa `PyroGuard-IA` (rama `main`). Framework: FastAPI u Other.
2. En **Settings → Environment Variables** crea `AI_API_KEY` con una clave larga.
3. Despliega y abre `https://<proyecto>.vercel.app/health`.
4. En el backend: `AI_SERVICE_URL=https://<proyecto>.vercel.app` (sin barra final) y la misma `AI_API_KEY`.

Si el build falla por tamaño, el límite es ~500 MB por función (`scikit-learn`, `scipy`, `pandas` y `numpy` rozan esa cifra). La primera petición tras un rato sin uso tarda más por el arranque en frío.

## Limitaciones y trabajo pendiente

- Sustituir progresivamente el dataset sintético por datos reales, etiquetados y recogidos bajo supervisión segura.
- Revisar umbrales de RF08 con el docente y el equipo de hardware.
- Medir recall de FIRE, falsos negativos y falsos positivos en un conjunto separado que represente el entorno objetivo.
- Verificar contrato exacto del ESP32: escala real del sensor de humo/gas y codificación de `flame`.
