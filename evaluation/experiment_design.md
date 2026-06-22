# Diseño cuasi-experimental para la validación pedagógica de SWARD

> Capítulo de metodología — Validación de impacto del tutor adaptativo SWARD
> (SAKT + recomendación priorizando el formato de aprendizaje preferido).
> Documento autocontenido, listo para integrarse en la tesis.

## 1. Problema y justificación

SWARD modela el conocimiento del estudiante con un modelo SAKT (Self-Attentive
Knowledge Tracing) y recomienda el siguiente recurso priorizando el **formato**
en el que cada alumno rinde mejor (práctica, lectura, video o quiz). La pregunta
de investigación no es si el sistema funciona técnicamente, sino si **produce
aprendizaje**: el jurado exigirá evidencia de impacto, no solo de existencia.

Como en un contexto académico real rara vez es posible la aleatorización pura
(restricciones éticas, organizativas y de tamaño de cohorte), se propone un
**diseño cuasi-experimental** con grupo de comparación no equivalente y medición
pre/post, complementado con un análisis observacional de la mecánica de
preferencia de formato.

## 2. Preguntas de investigación e hipótesis

**RQ1.** ¿Los estudiantes que usan SWARD con recomendación adaptativa logran
mayor ganancia de aprendizaje que quienes usan el LMS sin adaptación?

- H1₀ (nula): la ganancia de aprendizaje normalizada media del grupo de
  tratamiento es igual a la del grupo de control (g_T = g_C).
- H1₁ (alterna): g_T > g_C.

**RQ2.** ¿El rendimiento de cada estudiante es mayor en su formato "fuerte", y
recibir recursos en ese formato se asocia a mejores aciertos posteriores?

- H2₀: el acierto en el formato fuerte es igual al acierto en los demás formatos.
- H2₁: el acierto en el formato fuerte es mayor.

**RQ3.** ¿La adaptación mejora el **engagement** (recursos completados, recursos
vistos, persistencia en la sesión) respecto al grupo de control?

- H3₀: engagement_T = engagement_C. H3₁: engagement_T > engagement_C.

## 3. Diseño

Diseño **cuasi-experimental con grupo de control no equivalente y pretest-postest**
(Campbell & Stanley, diseño 10), notación:

```
Grupo Tratamiento (T):  O1   X(SWARD adaptativo)   O2
Grupo Control     (C):  O1   X(LMS sin adaptación) O2
```

- `O1`: pretest de conocimientos por concepto (línea base).
- `X`:  intervención durante 4–8 semanas (un módulo/unidad del curso).
- `O2`: postest equivalente al pretest (formas A/B contrabalanceadas).

Variante interna (within-subject) admisible si solo se dispone de una cohorte:
diseño **pre/post intra-sujeto** usando los primeros vs. últimos conceptos de la
secuencia como proxy (lo que ya calcula `pedagogical_analysis.py`), con la
advertencia de que no controla maduración ni efecto de práctica.

### Asignación a grupos

Al no ser aleatorizable a nivel individual sin contaminación, se asigna por
**conglomerados** (secciones/aulas completas) a tratamiento o control. Se
documenta la no equivalencia y se controla estadísticamente con el pretest como
covariable (ANCOVA).

## 4. Variables

| Tipo | Variable | Operacionalización |
|---|---|---|
| Independiente | Uso de recomendación adaptativa | Tratamiento (SWARD) vs. Control (LMS) |
| Independiente (RQ2) | Formato del recurso | `tipo_recurso` ∈ {practica, lectura, video, quiz}_generado |
| Dependiente | Ganancia de aprendizaje | Ganancia normalizada de Hake (ver §5) |
| Dependiente | Acierto | nota/grademax ≥ 0.5 ⇒ 1, si no 0 (consistente con la ingesta) |
| Dependiente | Engagement | recursos completados, % vistos (`es_vista`), nº interacciones/sesión, tasa de abandono |
| Covariable | Conocimiento previo | Puntaje del pretest por concepto |
| Covariable | Exposición | Nº de recursos recibidos (control de dosis) |
| Control | Curso, concepto, docente | Efectos fijos / estratos |

## 5. Métricas

### 5.1 Ganancia de aprendizaje normalizada (Hake)

```
g = (post − pre) / (1 − pre)
```

con `pre` y `post` como proporción de aciertos (0–1). Indefinida si `pre = 1`.
Se reporta media y mediana por grupo. Interpretación de Hake: g < 0.3 baja,
0.3 ≤ g < 0.7 media, g ≥ 0.7 alta.

### 5.2 Tamaño de efecto

- **Cliff's δ** (no paramétrico, robusto a muestras pequeñas y a la binariedad):
  insignificante <0.147, pequeño <0.33, mediano <0.474, grande en otro caso.
- **Cohen's d** (referencial, supone normalidad aproximada).
- **g de Hedges** si se requiere corrección por muestra pequeña.

### 5.3 Engagement

- Recursos completados / recomendados (tasa de finalización).
- Proporción de recursos efectivamente vistos (`es_vista`).
- Longitud media de sesión (nº de interacciones consecutivas).
- Tasa de retorno entre sesiones.

## 6. Tamaño de muestra (potencia)

Para detectar una diferencia entre dos grupos independientes con prueba de dos
colas, α = 0.05 y potencia (1−β) = 0.80, el tamaño aproximado por grupo es:

```
n_por_grupo ≈ 2 · ( (z_{1−α/2} + z_{1−β}) / d )²
            = 2 · ( (1.96 + 0.84) / d )²
```

| Efecto esperado (Cohen's d) | n por grupo | n total |
|---:|---:|---:|
| 0.8 (grande) | ~26 | ~52 |
| 0.5 (mediano) | ~64 | ~128 |
| 0.3 (pequeño) | ~175 | ~350 |

Recomendación: planificar para un efecto **mediano** (≈64 por grupo). El dataset
de ejemplo actual (~26 alumnos por curso, 1 sola exposición por concepto) está
**por debajo** del umbral incluso para detectar efectos grandes con confianza;
sirve para validar el pipeline de análisis, no para conclusiones definitivas.
Para análisis pareados (within-subject) usar el d pareado y la prueba de
Wilcoxon, que ganan potencia respecto al diseño de dos grupos.

## 7. Plan de análisis estadístico

1. **Descriptivos** por grupo: pre, post, Δ, g, engagement (media, DE, mediana, IQR).
2. **Verificación de equivalencia inicial**: comparar `O1` entre grupos
   (Mann-Whitney o t); si difieren, usar ANCOVA con pretest como covariable.
3. **Contraste principal (RQ1)**: ganancia normalizada T vs. C mediante
   **Mann-Whitney U** (no paramétrico; las proporciones de aciertos no son
   normales). Reportar U, p, Cliff's δ e IC.
4. **RQ2 (preferencia de formato)**: comparar acierto en formato fuerte vs.
   resto (Mann-Whitney + Cliff's δ). Definir el formato fuerte en una ventana
   temporal **anterior** y evaluar en una **posterior** para evitar fuga de
   información (split temporal), mitigando regresión a la media.
5. **RQ3 (engagement)**: Mann-Whitney por métrica + corrección de comparaciones
   múltiples (Holm-Bonferroni).
6. **Modelo confirmatorio**: regresión logística/mixta de `correcta` con efectos
   aleatorios por estudiante y por concepto, y predictores: grupo, formato,
   posición en la secuencia, conocimiento previo. Captura la dependencia de
   medidas repetidas que los tests univariados ignoran.
7. **Sensibilidad**: repetir con distintos umbrales de aprobación y fracciones
   pre/post para confirmar robustez de los hallazgos.

Todos los contrastes se ejecutan también con la implementación pura del script
(`pedagogical_analysis.py`) cuando scipy no esté disponible, declarando el motor.

## 8. Amenazas a la validez

**Validez interna**
- *Selección*: grupos no aleatorizados ⇒ posible no equivalencia (mitigación:
  ANCOVA con pretest, emparejamiento por propensión si hay covariables).
- *Maduración / historia*: aprendizaje por el solo paso del tiempo (mitigación:
  grupo de control concurrente).
- *Efecto test*: el pretest sensibiliza (mitigación: formas equivalentes A/B).
- *Difusión del tratamiento*: estudiantes de control acceden a SWARD
  (mitigación: separación por aulas, monitoreo de accesos).
- *Mortalidad experimental*: abandono diferencial (reportar y comparar bajas).

**Validez de constructo**
- El acierto binarizado (umbral 0.5) es un proxy grueso del aprendizaje;
  complementar con la nota continua y, si es posible, una evaluación externa.
- "Formato fuerte" definido con los propios datos puede capturar ruido
  (mitigación: split temporal de definición/evaluación).

**Validez externa**
- Una sola institución/cohorte limita la generalización (declarar alcance).

**Validez estadística**
- Muestra pequeña ⇒ baja potencia y p-valores inestables (priorizar tamaños de
  efecto e intervalos de confianza sobre la significancia binaria).
- Medidas repetidas ⇒ no independencia (usar modelos mixtos en el confirmatorio).

## 9. Consideraciones éticas

Consentimiento informado, anonimización de `estudiante_id`, equivalencia
pedagógica garantizada para el grupo de control (acceso al mismo contenido sin
la capa adaptativa) y oferta de la intervención al grupo de control tras el
estudio (diseño de lista de espera) si los resultados son favorables.

## 10. Cronograma y entregables

| Fase | Duración | Entregable |
|---|---|---|
| Pretest e instrumentación | 1 semana | Línea base `O1`, formularios A/B validados |
| Intervención | 4–8 semanas | Logs de interacciones con `tipo_recurso`, `es_vista` |
| Postest | 1 semana | `O2` |
| Análisis | 2 semanas | Reporte de `pedagogical_analysis.py` + modelo confirmatorio |

El script `evaluation/pedagogical_analysis.py` produce automáticamente el
análisis descriptivo, las curvas de aprendizaje, el contraste de formato y el
pre/post a partir del JSON de interacciones exportado del LMS.
