# TASKS — PyroGuard IA (FastAPI + Random Forest)

Plan de mejoras del servicio de IA. Cada task es un commit; el mensaje del commit es el texto de la task con su sufijo `[x.y]`.
Al terminarla se marca `[x]` en ese mismo commit.

Este repo es parte de un sistema de tres: **PyroGuard-Backend** (consume esta IA), **PyroGuard-IA** (este) y
**PyroGuard-Frontend** (dashboard). Cada repo tiene su propio `TASKS.md`; las dependencias entre ellos se indican en la task.

Estado de partida (2026-10-25): API `POST /predict` con contrato v1.0, modelo Random Forest entrenado con 2.400 registros
sintéticos, reglas de seguridad RF08, CI y 44 pruebas en verde. Limitación declarada: el dataset es sintético.

Revisión del plan (2026-11-09): las fases están ordenadas por orden de ejecución. Cambios respecto a la versión anterior:

- Cada task de API actualiza en su mismo commit el contrato y los ejemplos (el contrato lo exige); la Fase 5 solo revisa.
- «Datos reales» son lecturas del simulador etiquetadas a mano; los informes lo declaran (`dataset_type: simulated`).
- Comparar modelos y calibrar (Fase 4) van después de tener un conjunto de prueba separado (Fase 3): el dataset
  sintético es casi separable y sus métricas no demuestran nada por sí solas.
- Despliegue en Render (3.4 original, ahora 5.4); la rama de Hugging Face Spaces se descarta.

## Fase 0 — Base

- [x] test(safety): pruebas parametrizadas de cada umbral de las reglas de seguridad [0.1]
- [x] feat(api): cargar el modelo al arrancar el servicio en vez de en la primera petición [0.2]
- [x] refactor(predictor): quitar el reparseo de `timestamp`, que Pydantic ya convierte [0.3]

## Fase 1 — Servicio

- [x] feat(security): con `APP_ENV=production` el servicio no arranca sin `AI_API_KEY` [1.1]
- [x] test(security): pruebas del arranque en producción con y sin clave [1.2]
- [x] chore(deploy): declarar `APP_ENV` y `AI_API_KEY` en `render.yaml` y reflejarlo en el contrato [1.3]
- [x] feat(api): contadores en memoria de latencia y de predicciones por nivel de riesgo [1.4]
- [x] feat(api): `GET /metrics`, protegido con la misma clave que `/predict` [1.5]
- [x] test(api): pruebas de `/metrics` y de que exige la clave [1.6]
- [x] feat(ml): `RiskPredictor.predict_many` vectorizado, con una sola llamada al modelo [1.7]
- [x] feat(api): `POST /predict/batch` con límite de tamaño y resultado por lectura [1.8]
- [x] test(api): pruebas de `/predict/batch` (errores parciales, límite y equivalencia con `/predict`) [1.9]
- [x] docs: contrato v1.1 con `/metrics`, `/predict/batch` y sus ejemplos ejecutables [1.10]

## Fase 2 — Versionado y regresión del modelo

- [x] feat(model): el entrenamiento escribe `models/modelo_riesgo.meta.json` con versión, SHA-256 del dataset, filas y métricas [2.1]
- [x] feat(api): `/health` expone `dataset_sha256` y el predictor avisa si cambia la versión de scikit-learn [2.2]
- [x] test(model): el artefacto, el `meta.json` y el CSV son coherentes entre sí [2.3]
- [x] test(api): pruebas de regresión con escenarios `normal`, `warning` y `fire` para detectar cambios de comportamiento [2.4]

## Fase 3 — Datos etiquetados (simulador)

- [x] feat(data): esquema y validador de lecturas etiquetadas por una persona (`NORMAL`, `WARNING`, `FIRE`) [3.1]
- [x] test(data): pruebas del validador con casos válidos e inválidos [3.2]
- [x] feat(scripts): script que importa lecturas del backend a un CSV listo para etiquetar [3.3]
- [x] test(scripts): pruebas del importador con una respuesta simulada del backend [3.4]
- [x] test(ml): conjunto de prueba separado del entrenamiento, con comprobación de que no se solapan [3.5]
- [x] feat(ml): informe de métricas por clase (precisión, recall y F1) sobre ese conjunto [3.6]
- [x] test(ml): pruebas del informe de métricas [3.7]
- [x] docs: flujo de etiquetado y límites de los datos del simulador [3.8]

## Fase 4 — Modelo sobre esos datos

- [x] test(ml): comparar Random Forest con otros modelos usando validación cruzada y documentar el resultado [4.1]
- [x] feat(ml): calibrar las probabilidades del modelo [4.2]
- [x] test(ml): pruebas de la calibración (rango y suma de probabilidades) [4.3]
- [x] chore(model): reentrenar y regenerar el `meta.json` [4.4]

## Fase 5 — Cierre

- [x] docs: revisar `docs/CONTRATO_IA_BACKEND.md` contra el código final [5.1]
- [x] docs: actualizar el README y `ENTREGA_AVANCE.md` [5.2]
- [x] chore(docker): usuario sin privilegios y `HEALTHCHECK` en el Dockerfile [5.3]
- [x] chore(deploy): desplegar y publicar la URL del servicio [5.4] (desplegado en Vercel: `https://pyro-guard-ia.vercel.app`; Render quedó como alternativa documentada)

## Fase 6 — Entrega

Revisión del 2026-11-17: la IA quedó desplegada en Vercel (`https://pyro-guard-ia.vercel.app`) en lugar de Render. Esta fase
deja el repo listo para entregar: URL publicada, lint y cobertura en CI, pruebas de los scripts que no tenían y un verificador
del despliegue.

- [x] docs(deploy): URL de producción de Vercel en el README y en el contrato [6.1]
- [x] docs(entrega): `ENTREGA_AVANCE.md` con el despliegue y cómo verificarlo [6.2]
- [x] fix(api): versión de la app FastAPI (OpenAPI) alineada con la del servicio `1.1.0` [6.3]
- [x] test(api): la versión de OpenAPI coincide con `SERVICE_VERSION` [6.4]
- [x] feat(api): `GET /` enumera también `/predict/batch` y `/metrics` [6.5]
- [x] test(api): prueba de `GET /` [6.6]
- [x] refactor(api): una sola función que carga el predictor o responde 503 [6.7]
- [x] test(api): `/predict/batch` responde 500 si el modelo falla al clasificar el lote [6.8]
- [x] test(predictor): artefacto sin metadatos, con columnas o con clases incompatibles [6.9]
- [x] test(predictor): aviso cuando cambia la versión de scikit-learn [6.10]
- [x] test(predictor): un `timestamp` sin zona horaria se interpreta como UTC [6.11]
- [x] test(data): el generador reproduce el CSV versionado (mismo SHA-256 que `meta.json`) [6.12]
- [x] test(data): el dataset generado está balanceado y dentro de los rangos del contrato [6.13]
- [x] refactor(scripts): extraer la carga y validación del dataset de `2_entrenar_modelo.py` a una función [6.14]
- [x] test(scripts): pruebas de esa validación (columnas, vacíos y clases) [6.15]
- [x] chore(lint): ruff en `requirements-dev.txt` y `ruff.toml` [6.16]
- [x] style: ordenar los imports (ruff I001) [6.17]
- [x] style: `datetime.UTC` e import obsoleto de `typing` (ruff UP017, UP035) [6.18]
- [ ] style: `zip(..., strict=True)` (ruff B905) [6.19]
- [ ] style: unir los `with` anidados (ruff SIM117) [6.20]
- [ ] style: líneas largas e import fuera de lugar (ruff E501, E402) [6.21]
- [ ] chore(ci): paso de ruff en GitHub Actions [6.22]
- [ ] chore(ci): medir la cobertura con coverage.py [6.23]
- [ ] chore(ci): resumen de cobertura en el job [6.24]
- [ ] chore(git): ignorar `.coverage` y `htmlcov/` [6.25]
- [ ] test(deploy): `vercel.json` incluye el modelo y excluye lo que no hace falta en ejecución [6.26]
- [ ] test(deploy): `render.yaml` fija `APP_ENV=production` y deja `AI_API_KEY` como secreto [6.27]
- [ ] feat(scripts): `7_verificar_despliegue.py` comprueba `/health` y `/predict` de una URL [6.28]
- [ ] test(scripts): pruebas del verificador con respuestas simuladas [6.29]
- [ ] docs: verificador de despliegue en el README [6.30]
- [ ] feat(api): `/health` incluye `contract_version` y el contrato lo documenta [6.31]
- [ ] test(api): `/health` devuelve `contract_version` también en `degraded` [6.32]
- [ ] test(security): la clave configurada se compara sin los espacios de la variable de entorno [6.33]
- [ ] test(schemas): `flame` rechaza textos y enteros distintos de 0/1, y `gas: null` vale 0 [6.34]
- [ ] docs(env): `.env.example` documenta `PORT` [6.35]
- [ ] docs: `CHANGELOG.md` con las versiones del modelo y del contrato [6.36]
- [ ] docs(readme): insignia del workflow de pruebas [6.37]
- [ ] docs(tasks): cierre de la Fase 6 en `ENTREGA_AVANCE.md` [6.38]
