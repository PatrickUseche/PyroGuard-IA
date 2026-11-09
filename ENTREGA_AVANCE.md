# Evidencia de avance — Componente IA de PyroGuard

## Implementado

- [x] Generación reproducible de 2.400 registros sintéticos balanceados entre `NORMAL`, `WARNING` y `FIRE`.
- [x] Entrenamiento Random Forest con partición estratificada 80/20 y semilla fija.
- [x] Evaluación con accuracy, precision, recall, F1-score y matriz de confusión.
- [x] Serialización/versionado del modelo mediante `joblib`.
- [x] API REST `POST /predict` para consumir el modelo.
- [x] Validación de rangos de entrada y contrato compatible con el documento de requisitos.
- [x] Capa inicial de reglas de seguridad para RF08.
- [x] Pruebas automatizadas para API y reglas.
- [x] Contrato de integración v1.1 con el backend Django (`docs/CONTRATO_IA_BACKEND.md`) y ejemplos ejecutables.
- [x] Errores con formato uniforme y códigos estables para el backend.
- [x] `AI_API_KEY` obligatoria con `APP_ENV=production` (el servicio no arranca sin ella) y modelo cargado al arrancar.
- [x] `POST /predict/batch` (hasta 100 lecturas, errores por lectura) y `GET /metrics` (latencia y conteo por nivel de riesgo).
- [x] Probabilidades calibradas (modelo `pyroguard-rf-1.1.0`) y `models/modelo_riesgo.meta.json` con el SHA-256 del dataset de entrenamiento.
- [x] Comparación de seis modelos con validación cruzada (`docs/COMPARACION_MODELOS.md`).
- [x] Importador de lecturas del backend, validador de CSV etiquetado e informe de precisión, recall y F1 por clase (`docs/ETIQUETADO.md`).
- [x] Pruebas de regresión con escenarios `normal`, `warning` y `fire`.

## Requisitos cubiertos en este avance

- RF06 — Cálculo de riesgo mediante IA.
- RF07 — Clasificación `NORMAL/WARNING/FIRE` y probabilidad estimada.
- RF08 — Reglas de seguridad que combinan modelo y variables ambientales.
- RF15 — Simulación de datos de prueba sin depender del ESP32.
- RNF07 — Versión de modelo y datos de trazabilidad en la respuesta.
- RNF08 — Organización modular y documentación.
- RNF14 — Pruebas automáticas.

## Cómo demostrarlo

1. Ejecutar `python scripts/3_probar_predicciones.py` para visualizar escenarios de prueba.
2. Ejecutar `python -m pytest -q` para mostrar el resultado de las pruebas.
3. Iniciar `start_windows.cmd`, abrir `/docs` y probar `POST /predict` con el JSON de README.
4. Revisar `docs/CONTRATO_IA_BACKEND.md` para el flujo de integración con el backend.

## Limitación declarada

El dataset es sintético y sirve para validar el flujo de software, no el rendimiento real del detector. No afirmar que un resultado alto de métricas demuestra detección confiable de incendios reales.

Las lecturas «reales» que se pueden etiquetar hoy vienen del simulador del backend, así que tampoco son datos de campo. Aún no hay un conjunto etiquetado: las métricas por clase sobre datos etiquetados están pendientes de que alguien etiquete lecturas.
