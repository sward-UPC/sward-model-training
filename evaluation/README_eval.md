# Evaluación comparativa de modelos de Knowledge Tracing (SWARD)

Este módulo produce la **evidencia cuantitativa de la tesis**: una comparación
reproducible entre el modelo elegido para SWARD (SAKT) y alternativas, sobre el
dataset **real de Moodle** exportado por `export_moodle_to_pykt.py`.

## Qué hace

`compare_models.py` entrena y evalúa varios modelos de Knowledge Tracing usando
**validación cruzada K-fold (k=5 por defecto)** sobre
`outputs/moodle_kt_dataset.json`, y reporta cuatro métricas por modelo,
promediadas sobre los folds como **media ± desviación estándar**:

| Métrica | Significado |
|---|---|
| **AUC**  | Área bajo la curva ROC. Mide la capacidad del modelo de **ordenar** aciertos por encima de fallos, independiente del umbral. Es la métrica estándar en la literatura de Knowledge Tracing. 0.5 = azar; 1.0 = perfecto. |
| **ACC**  | Exactitud (proporción de aciertos/fallos correctamente clasificados) con umbral 0.5. Sensible al desbalance de clases. |
| **F1**   | Media armónica de precisión y recall sobre la clase "acierto". Útil cuando las clases están desbalanceadas. |
| **RMSE** | Error cuadrático medio entre la probabilidad predicha y la respuesta real (0/1). Mide **calibración**: penaliza predicciones confiadas pero erróneas. Menor es mejor. |

### Modelos comparados

1. **SAKT (pyKT)** — *Self-Attentive Knowledge Tracing*. El modelo en producción
   en SWARD (`ms-recomendacion`). Atención auto-regresiva sobre la historia.
2. **DKT (pyKT)** — *Deep Knowledge Tracing*. LSTM clásico; línea de comparación
   neuronal de referencia en la literatura.
3. **Baseline global** — predice para todas las posiciones el **promedio global
   de aciertos** del conjunto de entrenamiento. Es el clasificador trivial; su
   AUC es exactamente 0.5 (no ordena nada). Sirve como **piso**: cualquier
   modelo útil debe superarlo.
4. **Baseline por-concepto** — predice la **tasa media de acierto de cada
   concepto** (la "dificultad" de cada sección de Moodle), estimada en train.
   Conceptos no vistos en train caen al promedio global. Mide cuánto se explica
   solo con la dificultad de cada sección, sin modelar el conocimiento del
   estudiante a lo largo del tiempo.

Los baselines **no requieren pyKT ni PyTorch**, por lo que aíslan la
contribución real del modelado neuronal: si SAKT/DKT no superan claramente al
baseline por-concepto, no están aprovechando la **secuencialidad** del aprendizaje.

## Rigor metodológico

- **Formato KT correcto:** en cada paso `t` el modelo conoce `(concepto_t,
  respuesta_t)` y debe predecir la respuesta del paso siguiente `t+1`. Las
  métricas se calculan sobre esos pares `t → t+1`.
- **Máscara/padding:** las secuencias se paddean a `SEQ_LEN`, pero **solo las
  posiciones válidas** entran en la pérdida y en las métricas (máscara binaria).
  Ninguna posición de relleno contamina los resultados.
- **Validación cruzada:** la partición en folds es **determinística** (barajado
  con semilla fija), de modo que train/test no se solapan y la corrida es
  **reproducible**. Cada secuencia (estudiante) aparece en test exactamente una vez.
- **Semilla fija:** `SEED=42` controla el barajado de folds y la inicialización
  de PyTorch.
- **AUC = n/a** si un fold no contiene ambas clases (AUC indefinido); se reporta
  honestamente en lugar de inventar un valor.
- **Métricas:** se usan las de `scikit-learn` si está instalado; si no, hay
  implementaciones puras equivalentes (AUC por Mann-Whitney U con corrección de
  empates, F1, ACC, RMSE) para que el script no dependa de sklearn.

## Salidas

- `outputs/model_comparison.csv` — datos crudos (media y std por métrica y modelo).
- `outputs/model_comparison.md` — **tabla en Markdown lista para pegar en la tesis**.
- Tabla impresa en consola con el detalle por fold.

## Cómo correrlo

### Local

```bash
cd sward-model-training
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pykt-toolkit scikit-learn numpy pandas

python evaluation/compare_models.py
```

El modelo es pequeño (23 conceptos, ~100 secuencias), así que entrena en CPU en
minutos. No requiere GPU.

#### Solo baselines (sin pyKT/torch)

```bash
python evaluation/compare_models.py --models baseline_global,baseline_concept
```

### CI (GitHub Actions)

Workflow `Evaluar modelos KT` (`.github/workflows/eval-models.yml`), disparable
manualmente (`workflow_dispatch`). Instala torch CPU + pykt-toolkit + scikit-learn,
corre la evaluación y sube `outputs/model_comparison.*` como **artifact**
descargable. Acepta `kfolds`, `epochs` y `models` como inputs.

## Parámetros (CLI o variables de entorno)

| CLI | Env | Default | Descripción |
|---|---|---|---|
| `--dataset` | `DATASET` | `outputs/moodle_kt_dataset.json` | JSON de entrada |
| `--kfolds` | `KFOLDS` | `5` | Folds de la validación cruzada |
| `--epochs` | `EPOCHS` | `60` | Épocas (modelos neuronales) |
| `--seq-len` | `SEQ_LEN` | `64` | Longitud máx. de secuencia / padding |
| `--emb-size` | `EMB_SIZE` | `64` | Dimensión de embedding |
| `--heads` | `HEADS` | `4` | Cabezas de atención (SAKT) |
| `--layers` | `LAYERS` | `2` | Capas del encoder (SAKT) |
| `--dropout` | `DROPOUT` | `0.2` | Dropout |
| `--lr` | `LR` | `1e-3` | Learning rate |
| `--batch` | `BATCH` | `16` | Tamaño de batch |
| `--seed` | `SEED` | `42` | Semilla global (reproducibilidad) |
| `--models` | `MODELS` | `sakt,dkt,baseline_global,baseline_concept` | Modelos a comparar |
| `--outdir` | `OUTDIR` | `outputs` | Carpeta de salida |

## Cómo interpretar la tabla

1. **Mirar el AUC primero.** Es la métrica de referencia en KT. El orden esperado
   es `SAKT ≈ DKT > baseline por-concepto > baseline global (0.5)`. Si SAKT no
   supera al baseline por-concepto, el modelo **no está explotando la
   secuencialidad** del aprendizaje y conviene revisar datos o hiperparámetros.
2. **Contrastar con los baselines.** La diferencia `AUC(modelo) − AUC(baseline
   por-concepto)` es la **ganancia real** atribuible al modelado del conocimiento
   del estudiante, descontando la dificultad de cada sección.
3. **ACC y F1** complementan el AUC ante desbalance (en Moodle ~55% de aciertos,
   bastante balanceado). F1 alto del baseline global suele deberse a predecir
   siempre la clase mayoritaria: no implica buen modelo (su AUC sigue siendo 0.5).
4. **RMSE** evalúa **calibración**: un modelo con buen AUC pero RMSE alto ordena
   bien pero asigna probabilidades mal calibradas.
5. **La desviación estándar** entre folds indica **estabilidad**. Con un dataset
   pequeño es esperable un std relativamente alto; reportarlo es parte del rigor.

> **Limitación honesta para la tesis:** el dataset de Moodle es pequeño
> (~100 secuencias, longitud media ~6, 23 conceptos). Los resultados son una
> **prueba de concepto** del pipeline de evaluación; los valores absolutos de AUC
> tienen alta varianza y no deben compararse 1:1 con benchmarks de la literatura
> (p. ej. ASSISTments 2015, ~683K interacciones). A medida que se acumulen
> interacciones reales en Moodle, re-ejecutar este script da una medición cada
> vez más confiable sin cambiar el código.
