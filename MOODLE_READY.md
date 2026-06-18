# Entrenamiento Moodle-ready

El checkpoint ahora incluye `concept_index` (mapa concepto→entero), que
`ms-recomendacion` usa para traducir el concepto (sección Moodle) al índice que
entiende el modelo. Para ASSISTments es identidad; para Moodle es no trivial.

## Estado
- ✅ `train.py` embebe `concept_index` en el `.pth` (contrato con ms-recomendacion).
- ✅ `build_concept_index(conceptos)` asigna un entero estable a cada concepto.
- ⏳ **Re-entrenamiento sobre data Moodle**: pendiente de **tener data real recolectada**
  (huevo-y-gallina: la ingesta — bug #3 — debe correr un tiempo en producción para
  acumular interacciones `(concepto, acierto)` antes de poder entrenar).

## Cómo re-entrenar con data de Moodle (flujo concreto)
Ya hay un pipeline funcional. **El entrenamiento corre en el venv** (torch + pykt).

1. **Exportar** desde Moodle (API REST) → CSVs pyKT + `concept_index.json`:
   ```bash
   MOODLE_URL=... MOODLE_TOKEN=... python export_moodle_to_pykt.py
   # genera data/moodle/{train_valid_sequences,test_sequences}.csv + concept_index.json
   # y outputs/moodle_kt_dataset.json (secuencias legibles)
   ```
   El concepto = **sección del curso**; el acierto = `nota/nota_máx ≥ 0.5`.

2. **Entrenar** SAKT sobre Moodle (mismo `train.py`, dataset por env var):
   ```bash
   KT_DATASET=moodle python train.py
   # → outputs/sakt_moodle.pth (incluye concept_index)
   ```
   `train.py` carga `data/moodle/concept_index.json` y lo embebe en el checkpoint.

3. **Subir a S3** y apuntar `SAKT_MODEL_S3_KEY=sakt/moodle/model.pth` en ms-recomendacion.

> **Escala actual** (data ya existente en el Moodle de pruebas): ~104 secuencias,
> ~650 interacciones, 23 conceptos, ~55% aciertos. Suficiente para demostrar el
> pipeline end-to-end y un modelo de conceptos Moodle; para AUC alto se necesita
> más volumen (más estudiantes/actividades con el tiempo).

> Mientras no haya modelo Moodle, ms-recomendacion cae a predicción mock para
> conceptos desconocidos (no rompe). Ver `project_sward_recomendacion_estado`.
