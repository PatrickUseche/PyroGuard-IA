# Comparación de modelos

Generado con `python scripts/6_comparar_modelos.py` el 2026-11-09. Validación cruzada estratificada de 5 pliegues
sobre `data/dataset_sintetico.csv` (2400 filas, SHA-256 `ff5a6bba3784…`), semilla 42.

| Modelo | Exactitud | F1 macro | Entrenar (s) | Predecir (s) |
|---|---|---|---|---|
| RandomForest (producción) | 0.9829 ± 0.0042 | 0.9829 ± 0.0042 | 0.722 | 0.082 |
| HistGradientBoosting | 0.9812 ± 0.0037 | 0.9813 ± 0.0037 | 1.303 | 0.026 |
| ExtraTrees | 0.9808 ± 0.0048 | 0.9808 ± 0.0048 | 0.545 | 0.089 |
| DecisionTree | 0.9758 ± 0.0039 | 0.9758 ± 0.0039 | 0.014 | 0.006 |
| LogisticRegression | 0.9662 ± 0.0076 | 0.9663 ± 0.0075 | 0.020 | 0.005 |
| KNN | 0.9654 ± 0.0058 | 0.9653 ± 0.0059 | 0.007 | 0.017 |

Tiempos medios por pliegue en la máquina donde se generó la tabla; sirven para comparar entre filas, no como valor absoluto.

## Cómo leerlo

- **Estas cifras son sobre datos sintéticos** y el generador (`scripts/1_generar_dataset.py`) define las clases con rangos
  parecidos a los umbrales de las reglas de seguridad. La diferencia entre el mejor y el peor modelo es de
  0.0176 de F1 macro: un modelo aprende ese reparto casi igual de bien que otro, así que **no demuestra que uno
  funcione mejor ante un incendio real**.
- El mejor por F1 macro en esta tabla es **RandomForest**. Con diferencias dentro de la desviación entre pliegues no hay
  motivo para cambiar de modelo.
- Se mantiene el Random Forest porque entrega probabilidades razonables por clase (las usan las reglas de seguridad y el
  backend), no necesita escalar las variables y ya está desplegado y probado.
- Para decidir de verdad hace falta repetir esta comparación sobre `data/lecturas_etiquetadas.csv` (ver
  `docs/ETIQUETADO.md`) o, mejor, sobre datos de nodos reales.
