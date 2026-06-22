# Marco de validación pedagógica — SWARD

Este directorio contiene el **marco de validación de impacto** del tutor
adaptativo SWARD: evidencia de que el sistema ayuda a aprender, no solo de que
existe. Es el material que el jurado de tesis pedirá en el capítulo de resultados
y metodología.

## Contenido

| Archivo | Qué es |
|---|---|
| `pedagogical_analysis.py` | Herramienta de análisis (offline, sobre un JSON de interacciones). Genera consola + `outputs/pedagogical_report.md`. |
| `experiment_design.md` | Diseño cuasi-experimental (control vs. tratamiento, pre/post) listo para el capítulo de metodología: hipótesis, variables, métricas, tamaño de muestra, amenazas a la validez y plan estadístico. |
| `README_pedagogico.md` | Este documento. |

## Qué analiza la herramienta

1. **Curvas de aprendizaje.** Evolución del acierto a lo largo de las
   interacciones (por posición global y por n-ésima exposición a cada concepto).
   Responde: ¿el acierto mejora con el refuerzo en el tiempo? Reporta la
   pendiente OLS de la curva global.
2. **Preferencia de formato.** Por alumno determina su formato "fuerte"
   (mayor acierto) y compara el acierto en ese formato vs. los demás mediante
   **Mann-Whitney U** + tamaños de efecto (**Cliff's δ**, Cohen's d). Además
   estima un *efecto posterior*: acierto en la interacción siguiente a una en el
   formato fuerte vs. en otro formato. **Requiere `tipo_recurso` en los datos.**
3. **Pre/post.** Compara desempeño temprano vs. tardío por alumno
   (proxy de mejora). Calcula la **ganancia de aprendizaje normalizada de Hake**
   y un test pareado (Wilcoxon con scipy, o prueba de signos binomial pura).

Toda la salida incluye **CAVEATS** explícitos sobre tamaño de muestra y sobre el
carácter observacional (no causal) del análisis.

## Uso

```bash
# Desde la raíz del repo (sward-model-training/)
python evaluation/pedagogical_analysis.py \
    --input outputs/moodle_kt_dataset.json \
    --output outputs/pedagogical_report.md \
    --early-frac 0.4 --late-frac 0.4
```

Equivalente por variables de entorno: `PEDA_INPUT`, `PEDA_OUTPUT`,
`PEDA_EARLY_FRAC`, `PEDA_LATE_FRAC`.

### Dependencias

- **Ninguna obligatoria.** Los imports científicos (scipy) son **perezosos**:
  si scipy está, se usa para Mann-Whitney y Wilcoxon; si no, hay una
  implementación pura (aproximación normal con corrección por empates y prueba
  de signos exacta). El reporte declara qué motor se usó.
- El archivo pasa `python -m py_compile` sin scipy instalado.

## Formatos de entrada aceptados (autodetectados)

**A) Dataset KT** (`outputs/moodle_kt_dataset.json`):

```json
{
  "concept_index": {"<nombre>": 0, "...": 1},
  "sequences": [
    {"student": 7, "course": 2, "concepts": [0,1,2], "responses": [1,0,1],
     "tipos_recurso": ["practica_generada", "..."],  // opcional
     "es_vista": [false, true]}                        // opcional
  ]
}
```

**B) Salida de `GET /api/v1/dashboard/training-data`** (lista plana):

```json
[{"estudiante_id": 7, "concepto": "SQL Básico", "correcta": 1, "orden": 3,
  "tipo_recurso": "video_generado", "es_vista": false}]
```

El cargador acepta sinónimos de claves (`student`/`estudiante_id`,
`correct`/`correcta`/`nota`, etc.) y convierte notas continuas a acierto binario
con umbral 0.5 (consistente con la ingesta de Moodle).

## Datos necesarios para resultados fuertes

El dataset de ejemplo valida el *pipeline*, pero **no alcanza** para conclusiones
de tesis. Para evidencia sólida se requiere:

1. **`tipo_recurso` por interacción** (`practica_generada`, `lectura_generada`,
   `quiz_generado`, `video_generado`). Sin este campo, el bloque central
   (preferencia de formato) no se ejecuta. Hoy `moodle_kt_dataset.json` no lo
   incluye y el análisis lo reporta como no disponible.
2. **Exposiciones repetidas por concepto** (≥2 por alumno). El dataset actual
   tiene ~1 interacción por concepto, lo que impide curvas de dominio
   intra-concepto reales.
3. **Secuencias más largas** (≥8–10 interacciones/alumno) para una partición
   pre/post informativa. Actualmente ~6.
4. **Grupo de control** (alumnos sin recomendación adaptativa) para pasar de
   correlación a comparación entre condiciones — ver `experiment_design.md`.
5. **Tamaño de muestra** acorde a la potencia: ~64 alumnos por grupo para un
   efecto mediano (ver tabla en `experiment_design.md`).
6. **`es_vista`** y marcas de tiempo para las métricas de engagement.

## Relación con la tesis

- `experiment_design.md` → capítulo de **Metodología**.
- `outputs/pedagogical_report.md` (generado) → capítulo de **Resultados**.
- Los CAVEATS del reporte → sección de **Limitaciones**.
