# Etiquetado de lecturas y conjunto de prueba

Este documento explica cómo se construye el conjunto de prueba que mide el modelo y qué **no** puede decir de él.

## Qué datos son

Las lecturas salen del backend, que hoy solo recibe las del **simulador** (`simulador_iot.py`). Etiquetarlas a mano
las convierte en un conjunto *simulado y etiquetado por una persona*, no en datos de campo. Por eso los informes
llevan `evaluation_dataset_type: simulated` y la advertencia de que no estiman el desempeño ante incendios reales.
Cuando haya nodos ESP32 reales, el mismo flujo sirve cambiando de dónde se importa.

## Flujo

1. **Importar** (`scripts/4_importar_lecturas.py`). Lee `GET /api/nodes/` y `GET /api/nodes/<id>/readings/` del backend,
   de la más antigua a la más nueva, y escribe un CSV con la columna `label` vacía.

   ```
   python scripts/4_importar_lecturas.py --base-url http://127.0.0.1:8000 --all-nodes --out data/por_etiquetar.csv
   ```

   El backend pide sesión para leer. Un script no puede iniciar sesión con Google, así que hay dos opciones:
   `PUBLIC_READ_ACCESS=True` en el backend (solo en local) o un token de DRF con `--token` o la variable
   `PYROGUARD_API_TOKEN`. `data/por_etiquetar*.csv` está en `.gitignore`.
2. **Etiquetar** a mano la columna `label` con `NORMAL`, `WARNING` o `FIRE`. La columna `backend_risk_level` es solo una
   pista de lo que calculó el backend; copiarla hace que el informe mida al backend contra sí mismo. `notes` es libre.
3. **Guardar** el resultado como `data/lecturas_etiquetadas.csv` (este sí se versiona).
4. **Medir** con `python scripts/5_informe_metricas.py`. Valida el archivo, comprueba que no haya fuga de datos y
   escribe `reports/evaluation_labeled.json` con precisión, recall y F1 por clase para el modelo solo y para el
   resultado final (modelo + reglas de seguridad).

## Esquema del CSV etiquetado

| Columna | Obligatoria | Regla |
|---|---|---|
| `node_id` | sí | 1 a 50 caracteres |
| `temperature` | sí | −50 a 150 °C |
| `humidity` | sí | 0 a 100 % |
| `smoke` | sí | ≥ 0 |
| `gas` | no | ≥ 0; vacío equivale a 0 |
| `flame` | sí | `0`/`1` o `true`/`false` |
| `timestamp` | sí | ISO 8601 |
| `label` | sí | `NORMAL`, `WARNING` o `FIRE` (sin distinguir mayúsculas) |
| `backend_risk_level`, `notes` | no | Se ignoran al evaluar |

Los valores se validan con el mismo esquema que `POST /predict`. Un `node_id` con `timestamp` repetido es un error.
Cada problema se informa con su número de fila (la primera fila de datos es la 1).

## Cómo etiquetar sin sesgar la medición

- Etiqueta por lo que ocurre en el escenario (humo de cocina, calor del mediodía, llama real), no aplicando los umbrales
  de `app/safety_rules.py`: si se copian, el informe mide las reglas contra sí mismas.
- `WARNING` es «merece revisión»; `FIRE` es «hay fuego o está muy probablemente en curso».
- Una llama aislada sin otros indicios es `WARNING`, igual que decide el servicio.
- Lo ideal es que otra persona revise una muestra de las etiquetas.

## Garantías contra la contaminación

Antes de evaluar, `app/evaluation_set.py` rechaza el conjunto si:

- tiene filas con `node_id` que empiece por `SYNTH-` (sintéticas);
- alguna fila tiene las mismas medidas, a dos decimales, que una fila de `data/dataset_sintetico.csv`.

## Límites

- Con menos de 30 lecturas de una clase el informe avisa de que sus métricas son poco fiables.
- El simulador genera escenarios que alguien programó: aciertos altos aquí no prueban que el modelo funcione en el campo.
- El conjunto de entrenamiento sigue siendo sintético (`data/dataset_sintetico.csv`); este conjunto solo sirve para medir.
