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

## Cómo re-entrenar con data de Moodle (cuando exista)
1. Exportar las interacciones desde `sward_trazabilidad` (tabla `interactions`):
   `estudiante_id, concept_id, is_correct, fecha` por curso.
2. Construir secuencias por estudiante (ordenadas por `fecha`), formato pyKT:
   `concepts` = lista de conceptos (secciones), `responses` = 0/1.
3. `concept_index = build_concept_index(<todas las secciones únicas>)` y mapear
   cada `concept_id` a su entero; `n_skills = len(concept_index)`.
4. Entrenar SAKT igual que hoy, con `num_c = n_skills`.
5. Guardar el checkpoint **con** `concept_index` (ya lo hace `train.py`).
6. Subir a S3 (`upload_s3.py --version vN`) y apuntar
   `SAKT_MODEL_S3_KEY` en ms-recomendacion.

> Mientras no haya modelo Moodle, ms-recomendacion cae a predicción mock para
> conceptos desconocidos (no rompe). Ver `project_sward_recomendacion_estado`.
