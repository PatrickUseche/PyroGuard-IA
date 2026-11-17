# Cambios de PyroGuard-IA

Versiones del servicio (`SERVICE_VERSION`), del contrato con el backend (`contract_version`) y del modelo
(`model_version`). El detalle de cada cambio está en `TASKS.md` y en el historial de commits.

## Servicio 1.1.0 — contrato 1.1 — modelo `pyroguard-rf-1.1.0` (2026-11-17)

- Despliegue en producción en Vercel: `https://pyro-guard-ia.vercel.app`.
- `GET /` enumera todos los endpoints y la versión del servicio; `/docs` muestra la versión 1.1.0.
- Script `scripts/7_verificar_despliegue.py` para comprobar `/health` y `/predict` de un despliegue.
- Lint con ruff y cobertura con coverage.py en GitHub Actions.

## Servicio 1.1 — contrato 1.1 — modelo `pyroguard-rf-1.1.0` (2026-11-09)

- `POST /predict/batch` (hasta 100 lecturas, errores por lectura) y `GET /metrics`.
- `AI_API_KEY` obligatoria con `APP_ENV=production`; el modelo se carga al arrancar.
- Probabilidades calibradas y `models/modelo_riesgo.meta.json` con el SHA-256 del dataset.
- `/health` expone `dataset_sha256`.

## Servicio 1.0 — contrato 1.0 — modelo `pyroguard-rf-1.0.0` (2026-10-25)

- `POST /predict` y `GET /health` con errores uniformes.
- Random Forest entrenado con 2.400 registros sintéticos y reglas de seguridad RF08.
