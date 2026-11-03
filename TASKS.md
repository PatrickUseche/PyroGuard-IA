# TASKS — PyroGuard IA (FastAPI + Random Forest)

Plan de mejoras del servicio de IA. Cada task es un commit; el mensaje del commit es el texto de la task con su sufijo `[x.y]`.
Al terminarla se marca `[x]` en ese mismo commit.

Este repo es parte de un sistema de tres: **PyroGuard-Backend** (consume esta IA), **PyroGuard-IA** (este) y
**PyroGuard-Frontend** (dashboard). Cada repo tiene su propio `TASKS.md`; las dependencias entre ellos se indican en la task.

Estado de partida (2026-10-25): API `POST /predict` con contrato v1.0, modelo Random Forest entrenado con 2.400 registros
sintéticos, reglas de seguridad RF08, CI y 44 pruebas en verde. Limitación declarada: el dataset es sintético.

## Fase 1 — Datos reales

- [ ] feat(data): definir el esquema de lecturas etiquetadas por una persona (`NORMAL`, `WARNING`, `FIRE`) [1.1]
- [ ] feat(scripts): script que importa lecturas del backend a un CSV listo para etiquetar [1.2]
- [ ] test(ml): conjunto de prueba separado y compuesto solo por datos reales [1.3]
- [ ] feat(ml): informe de métricas por clase (precisión, recall y F1) sobre ese conjunto [1.4]

## Fase 2 — Modelo

- [ ] feat(ml): calibrar las probabilidades del modelo [2.1]
- [ ] test(ml): comparar Random Forest con otros modelos y documentar el resultado [2.2]
- [ ] test(api): pruebas de regresión con escenarios `normal`, `warning` y `fire` para detectar cambios de comportamiento [2.3]
- [ ] chore(model): versionado del modelo junto con el dataset que lo entrenó [2.4]

## Fase 3 — Servicio

- [ ] feat(security): exigir `AI_API_KEY` cuando el servicio corre en producción [3.1]
- [ ] feat(api): `POST /predict/batch` para varias lecturas en una sola petición [3.2]
- [ ] feat(api): métricas de latencia y conteo de predicciones por nivel de riesgo [3.3]
- [ ] chore(deploy): desplegar en Render y publicar la URL del servicio [3.4] (la necesita el backend [5.1])

## Fase 4 — Documentación

- [ ] docs: actualizar `docs/CONTRATO_IA_BACKEND.md` con los cambios de las fases anteriores [4.1]
