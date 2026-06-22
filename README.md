# sward-model-training

Entrenamiento **offline** del modelo SAKT (*Self-Attentive Knowledge Tracing*) y
**suite de evaluación de tesis** del proyecto **SWARD**.

El checkpoint `.pth` resultante se sube a S3 y lo consume el microservicio
[`sward-ms-recomendacion`](#conexión-con-ms-recomendacion) para inferir el dominio
de conocimiento del estudiante en tiempo real. La carpeta [`evaluation/`](#suite-de-evaluación-evaluation)
contiene los experimentos reproducibles que sustentan la tesis (comparación de
modelos, fidelidad de la IA explicable y análisis pedagógico).

---

## Qué hace

1. **Entrena un SAKT** sobre dos dominios de datos:
   - **ASSISTments 2015** — benchmark público de knowledge tracing (~683K
     interacciones, 19K estudiantes, 100 skills). Sirve para validar la
     implementación contra la literatura.
   - **Moodle (SWARD)** — datos reales del LMS, donde el *concepto* es la **sección
     del curso** y el *acierto* es `nota / nota_máxima ≥ 0.5`.
2. **Embebe el `concept_index`** (mapa `concepto → entero`) dentro del checkpoint.
   Es el contrato con `ms-recomendacion`: le permite traducir un concepto Moodle al
   índice que entiende el modelo. Para ASSISTments es la identidad; para Moodle es
   no trivial.
3. **Sube el modelo a S3** para que el microservicio de recomendación lo cargue.
4. **Evalúa** modelos y explicaciones de forma reproducible para la tesis.

### Decisiones de diseño

- El SAKT está implementado **desde cero** (no depende del API interno de pyKT para
  el modelo), para tener control total del checkpoint y de la inferencia en
  `ms-recomendacion`. **pyKT** se usa para descarga y preprocesamiento de datos y
  como referencia de baselines en la evaluación.
- **MPS** (Apple Silicon) se habilita automáticamente; si no, usa CUDA o CPU.

---

## Stack

- Python 3.11
- [pyKT](https://github.com/pykt-team/pykt-toolkit) (`pykt-toolkit`)
- PyTorch ≥ 2.3 (MPS en Apple Silicon)
- numpy, pandas, scikit-learn, boto3, tqdm, gdown, requests

---

## Estructura

```
train.py                    ← entrena SAKT (ASSISTments o Moodle, según KT_DATASET)
prepare_data.py             ← descarga y preprocesa ASSISTments 2015 a formato pyKT
export_moodle_to_pykt.py    ← exporta interacciones reales de Moodle (API REST) a KT
generate_synthetic_kt.py    ← genera data KT sintética (BKT) sobre conceptos Moodle
upload_s3.py                ← sube el checkpoint entrenado a S3
requirements.txt
evaluation/                 ← suite de evaluación de tesis (ver sección dedicada)
.github/workflows/
  eval-models.yml           ← workflow de comparación de modelos (a demanda)
data/                       ← datasets (ignorado por git; ASSISTments + Moodle)
outputs/                    ← modelos, metadatos y reportes (ignorado por git)
  moodle_kt_dataset.json    ← dataset KT real de Moodle (snapshot versionado)
```

---

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Entrenamiento

### ASSISTments 2015 (validación contra literatura)

```bash
python prepare_data.py        # descarga y preprocesa el dataset (data/assist2015/)
python train.py               # entrena (~15-20 min en M4 Pro con MPS)
```

Salidas: `outputs/sakt_assist2015.pth` (state_dict + metadatos + `concept_index`)
y `outputs/model_meta.json` (n_skills, hiperparámetros, AUC final).

**Métrica esperada:** AUC test ≈ **0.74–0.76** (benchmark SAKT en la literatura).

### Moodle (modelo de producción)

El entrenamiento sobre Moodle depende de **tener data real recolectada**: la
ingesta de SWARD debe correr un tiempo en producción para acumular interacciones
`(concepto, acierto)` antes de poder entrenar.

```bash
# 1. Exportar desde Moodle (API REST) → CSVs pyKT + concept_index.json
MOODLE_URL=... MOODLE_TOKEN=... python export_moodle_to_pykt.py
#    genera data/moodle/{train_valid_sequences,test_sequences}.csv,
#    data/moodle/concept_index.json y outputs/moodle_kt_dataset.json

# 2. Entrenar SAKT sobre Moodle (mismo train.py, dataset por env var)
KT_DATASET=moodle python train.py
#    → outputs/sakt_moodle.pth (incluye concept_index)

# 3. Subir a S3
python upload_s3.py --dataset moodle
```

> **Escala actual** del Moodle de pruebas: ~100 secuencias / ~650 interacciones /
> 23 conceptos / ~55% aciertos. Suficiente para demostrar el pipeline end-to-end;
> para AUC alto se necesita más volumen.

Mientras no exista modelo Moodle entrenado, `ms-recomendacion` cae a una predicción
*mock* para conceptos desconocidos (no rompe).

> **Data sintética (opcional):** `generate_synthetic_kt.py` simula secuencias largas
> con dinámica tipo BKT sobre los conceptos Moodle reales, produciendo data
> fuertemente *aprendible* (AUC alto). Útil para demostrar que el pipeline y la
> trazabilidad del dominio funcionan, pero el AUC mide aprendizaje sobre datos
> simulados, no comportamiento real.

### Variables de entorno relevantes (`train.py`)

| Variable        | Default       | Descripción                                  |
| --------------- | ------------- | -------------------------------------------- |
| `KT_DATASET`    | `assist2015`  | dominio a entrenar (`assist2015` / `moodle`) |
| `KT_DATA_DIR`   | `data/<ds>`   | carpeta del dataset                          |
| `KT_EMB_SIZE`   | `256`         | dimensión de embedding                       |
| `KT_HEADS`      | `8`           | cabezas de atención                          |
| `KT_LAYERS`     | `2`           | capas del encoder                            |
| `KT_BATCH`      | `64`          | tamaño de batch                              |
| `KT_EPOCHS`     | `200`         | épocas                                       |
| `KT_PATIENCE`   | `20`          | early stopping                               |

---

## Destino en S3

```bash
python upload_s3.py                          # assist2015 → s3://sward-models/sakt/assist2015/
python upload_s3.py --dataset moodle         # moodle     → s3://sward-models/sakt/moodle/
python upload_s3.py --dataset moodle --version v2   # → s3://sward-models/sakt/v2/
```

Se suben, bajo el prefijo `sakt/<version>/`:

```
s3://sward-models/sakt/<version>/model.pth          ← checkpoint (state_dict + concept_index)
s3://sward-models/sakt/<version>/model_traced.pt    ← versión trazada (TorchScript)
s3://sward-models/sakt/<version>/model_meta.json    ← n_skills, hiperparámetros, AUC
```

Requiere credenciales AWS con permiso de escritura sobre el bucket `sward-models`.

---

## Conexión con ms-recomendacion

El microservicio `sward-ms-recomendacion` descarga el checkpoint desde S3 al
arrancar y lo usa para inferir el dominio del estudiante. Se configura con:

```bash
SAKT_MODEL_S3_KEY=sakt/moodle/model.pth      # o sakt/assist2015/model.pth
AWS_S3_MODEL_BUCKET=sward-models
```

El `concept_index` embebido en el `.pth` permite al microservicio traducir cada
concepto Moodle (sección del curso) al índice que el modelo espera. Además,
`ms-recomendacion` expone los **pesos de atención** del SAKT como explicación (IA
explicable) de su predicción de dominio — cuya fidelidad se valida en
[`evaluation/xai_faithfulness.py`](#3-fidelidad-de-la-xai--xai_faithfulnesspy).

---

## Suite de evaluación (`evaluation/`)

Experimentos **reproducibles** que respaldan la tesis. Cada script tiene un README
y/o documento de diseño asociado (`README_eval.md`, `README_xai.md`,
`README_pedagogico.md`, `experiment_design.md`, `xai_user_study.md`).

### 1. `fetch_dataset.py` — dataset fresco

Descarga el *training-data* **real** de SWARD (`GET /api/v1/dashboard/training-data`)
y lo convierte a `outputs/moodle_kt_dataset.json` (`concept_index` + `sequences`),
de modo que la evaluación refleje el Moodle actual y no un snapshot viejo.

```bash
SWARD_API_URL=https://<cloudfront> \
TRAZABILIDAD_SERVICE_KEY=<service-key> \
python evaluation/fetch_dataset.py
```

### 2. Comparación de modelos — `compare_models.py`

Validación cruzada **K-fold** (k=5 por defecto) sobre el dataset real de Moodle,
comparando:

- **SAKT** (pyKT) — Self-Attentive Knowledge Tracing.
- **DKT** (pyKT) — Deep Knowledge Tracing (LSTM).
- **Baseline global** — predice siempre el promedio global de aciertos.
- **Baseline por-concepto** — predice el promedio de aciertos de cada concepto
  (dificultad media de cada sección Moodle).

Reporta, promediado sobre folds (media ± desv. estándar): **AUC, ACC, F1, RMSE**,
respetando padding/máscara y el formato KT (predecir el acierto del paso `t+1`).
Salidas: `outputs/model_comparison.csv` y `outputs/model_comparison.md`.

### 3. Fidelidad de la XAI — `xai_faithfulness.py`

Mide si los **pesos de atención** del SAKT que se muestran como "explicación" son
realmente *fieles* (faithful) a la predicción, con las métricas estándar de XAI
(DeYoung et al., 2020, *ERASER*): **comprehensiveness**, **sufficiency** y
comparación contra borrado **aleatorio**.

```bash
python evaluation/xai_faithfulness.py \
    --checkpoint outputs/sakt_moodle.pth \
    --dataset outputs/moodle_kt_dataset.json \
    --muestras 80 --topk 1 2 --out outputs/xai_faithfulness.md
```

### 4. Análisis pedagógico — `pedagogical_analysis.py`

Genera evidencia de que el tutor adaptativo **ayuda a aprender**: curvas de
aprendizaje, preferencia de formato por alumno y comparación pre/post. Autodetecta
varios formatos de entrada (dataset KT, API `training-data`, o lista plana de
interacciones). Salida: `outputs/pedagogical_report.md` con hallazgos, tablas y
*caveats*.

```bash
python evaluation/pedagogical_analysis.py
```

### Cómo correr la comparación en CI

El workflow `.github/workflows/eval-models.yml` (`workflow_dispatch`, **solo a
demanda** — no forma parte del pipeline de despliegue) ejecuta la comparación de
modelos de forma reproducible:

1. Instala torch (CPU) + pyKT + sklearn.
2. Si están los **secrets** `SWARD_API_URL` y `TRAZABILIDAD_SERVICE_KEY`, refresca
   el dataset con `fetch_dataset.py`; si no, usa el snapshot versionado
   `outputs/moodle_kt_dataset.json`.
3. Ejecuta `compare_models.py` y sube `model_comparison.{csv,md}` como *artifact*.

Inputs del workflow: `kfolds` (5), `epochs` (60), `models`
(`sakt,dkt,baseline_global,baseline_concept`).
