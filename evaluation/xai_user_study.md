# Estudio con usuarios — Explicabilidad del modelo SAKT en SWARD

> Plantilla lista para anexo de tesis. Complementa la evaluación **cuantitativa** de
> fidelidad (`xai_faithfulness.py`) con la evaluación **centrada en el usuario**:
> ¿las explicaciones por atención son *comprensibles* y *generan confianza* en
> docentes y estudiantes? La fidelidad mide si la explicación es fiel al modelo; el
> estudio con usuarios mide si la explicación es *útil para las personas*. Ambas
> dimensiones son necesarias para sostener el título "IA Explicable".

---

## 1. Objetivo del estudio

Evaluar, con usuarios reales del sistema SWARD, la calidad percibida de las
explicaciones que acompañan a la predicción de dominio del modelo SAKT
(los pesos de atención sobre las interacciones pasadas del estudiante,
presentados en el panel docente y/o de estudiante).

### Preguntas de investigación
- **PI1 (Comprensión).** ¿Los usuarios entienden qué les dice la explicación sobre
  por qué el sistema predijo un determinado nivel de dominio?
- **PI2 (Confianza).** ¿La explicación aumenta la confianza del usuario en la
  recomendación frente a recibir la predicción sin explicación?
- **PI3 (Utilidad para la decisión).** ¿La explicación ayuda al docente a tomar una
  mejor decisión pedagógica (p. ej., a qué estudiante priorizar)?
- **PI4 (Carga cognitiva).** ¿La explicación es fácil de interpretar o resulta
  confusa/abrumadora?

### Hipótesis
- **H1.** La condición *con explicación* obtiene mayor puntaje de confianza (PI2)
  que la condición *sin explicación* (predicción "caja negra").
- **H2.** La comprensión (PI1) media en la relación entre explicación y confianza:
  los usuarios que entienden mejor la explicación confían más.

---

## 2. Diseño del estudio

- **Tipo:** experimento controlado, **entre-sujetos** para la condición principal
  (con vs. sin explicación) y **medidas repetidas** dentro de cada participante
  sobre varios casos de estudiante.
- **Variable independiente principal:** presencia de explicación.
  - **Grupo A (control):** ve sólo la predicción de dominio (probabilidad + nivel).
  - **Grupo B (tratamiento):** ve la predicción **+** la explicación por atención
    (interacciones pasadas resaltadas según su peso, con una leyenda breve).
- **Variables dependientes:** comprensión, confianza, utilidad y carga cognitiva
  (cuestionario Likert, sección 4) y, opcionalmente, **acierto/decisión** en una
  tarea (sección 5).
- **Contrabalanceo:** el orden de los casos de estudiante se aleatoriza por
  participante para controlar efectos de orden/aprendizaje.

### Muestra (n)
- **Población:** docentes de los cursos piloto y, en un segundo bloque,
  estudiantes. Conviene reportar ambos perfiles por separado.
- **Tamaño objetivo:** mínimo **n = 12 por grupo** para análisis exploratorio
  (estudio de tesis con población acotada); **n ≥ 20 por grupo** si se busca
  detectar un efecto mediano (d ≈ 0.5) con potencia 0.80 y α = 0.05 en una prueba
  de dos grupos. Documentar el cálculo de potencia realizado (p. ej. con G*Power) y
  el n finalmente alcanzado, justificando cualquier limitación.
- **Criterios de inclusión:** haber usado Moodle del curso; para docentes, tener a
  cargo el curso evaluado. **Exclusión:** participantes que desarrollaron SWARD.

### Procedimiento (≈ 20–25 min por participante)
1. **Consentimiento informado** y explicación del propósito (sin revelar las
   hipótesis para no inducir respuestas).
2. **Cuestionario demográfico** breve (rol, experiencia docente/uso de Moodle,
   familiaridad con IA, 1 ítem cada uno).
3. **Calibración (tarea de calentamiento):** un caso de ejemplo guiado para que el
   participante entienda la interfaz de su condición (A o B).
4. **Bloque experimental:** se presentan **4–6 casos** de estudiante (perfiles
   reales o realistas extraídos del dataset Moodle). Tras cada caso, el
   participante responde el cuestionario Likert (sección 4) y, si aplica, la tarea
   de decisión (sección 5).
5. **Cierre:** 2–3 preguntas abiertas (sección 6) y agradecimiento.

> **Cegado:** el facilitador no debe sugerir respuestas. Idealmente la asignación a
> grupo A/B se hace de forma automática y el facilitador desconoce la condición.

---

## 3. Material de estímulo

Para cada caso se muestra:
- El **nivel de dominio predicho** (probabilidad y etiqueta, p. ej. "dominio bajo
  del concepto *Árboles AVL*: 38%").
- **Sólo en Grupo B:** la lista de interacciones pasadas del estudiante con su
  **peso de atención** (resaltado/barra), más una leyenda de una frase:
  *"El modelo basó esta predicción principalmente en las interacciones
  resaltadas."*

Seleccionar casos que cubran: (a) predicción de dominio alto, (b) bajo, y
(c) un caso ambiguo (~50%), para evitar que todos los estímulos sean triviales.

---

## 4. Cuestionario (escala Likert de 5 puntos)

Escala: **1 = Totalmente en desacuerdo … 5 = Totalmente de acuerdo**
(salvo los ítems invertidos marcados con **(R)**, que se recodifican al analizar).

### A. Comprensión (PI1)
| # | Ítem |
|---|------|
| C1 | Entiendo por qué el sistema hizo esta predicción de dominio. |
| C2 | La explicación me deja claro qué interacciones del estudiante fueron más influyentes. |
| C3 | Podría explicarle a un colega, con mis palabras, en qué se basó el sistema. |
| C4 | **(R)** La información mostrada me resultó confusa. |

### B. Confianza (PI2)
| # | Ítem |
|---|------|
| T1 | Confío en la predicción de dominio que entrega el sistema. |
| T2 | La explicación aumenta mi confianza en la recomendación. |
| T3 | Me sentiría cómodo/a tomando una decisión pedagógica apoyándome en esta salida. |
| T4 | **(R)** Sospecho que el sistema podría estar equivocándose sin que yo lo note. |

### C. Utilidad para la decisión (PI3)
| # | Ítem |
|---|------|
| U1 | La explicación me ayuda a decidir cómo actuar con este estudiante. |
| U2 | La explicación aporta información que no tendría sólo con la nota/probabilidad. |
| U3 | Usaría esta explicación en mi práctica docente habitual. |

### D. Carga cognitiva / usabilidad (PI4)
| # | Ítem |
|---|------|
| L1 | Interpretar la explicación me exigió poco esfuerzo. |
| L2 | **(R)** Tuve que esforzarme demasiado para entender el gráfico de atención. |
| L3 | La cantidad de información mostrada fue adecuada (ni poca ni excesiva). |

> Sugerencia: para confianza puede añadirse, además, un ítem único validado de la
> literatura (p. ej. *Trust in Automation*) para comparabilidad externa.

---

## 5. Tarea conductual opcional (medida objetiva)

Para no depender sólo de auto-reporte, incluir una tarea con respuesta correcta:

- **Tarea de priorización:** dados 3 estudiantes con su predicción (Grupo B también
  ve la explicación), el docente debe **ordenar a quién apoyar primero**. Se compara
  su orden contra un criterio experto/ground-truth.
- **Métrica:** tasa de acierto y tiempo de decisión. Permite contrastar si la
  explicación mejora *decisiones reales*, no sólo la percepción.

---

## 6. Preguntas abiertas (cualitativo)

1. ¿Qué fue lo más útil de la explicación? ¿Qué le sobró o le faltó?
2. ¿Hubo algún caso en el que la explicación lo/la hizo *desconfiar* del sistema?
   ¿Por qué?
3. ¿Cómo cambiaría la forma en que se muestran las interacciones influyentes?

Analizar mediante **codificación temática** (extraer temas recurrentes; reportar
frecuencia de cada tema y citas representativas).

---

## 7. Análisis de resultados

### 7.1 Preparación
- Recodificar los ítems invertidos **(R)** (`x → 6 − x`).
- Promediar los ítems de cada dimensión para obtener un puntaje por
  constructo (Comprensión, Confianza, Utilidad, Carga).
- **Fiabilidad de la escala:** reportar **α de Cronbach** por dimensión
  (aceptable ≥ 0.70). Si una dimensión queda baja, analizar ítem a ítem.

### 7.2 Estadística
- **Descriptivos:** media, desviación estándar y mediana por dimensión y grupo;
  acompañar con gráficos de cajas (boxplots) o de barras de divergencia (Likert).
- **H1 (con vs. sin explicación):** comparar la dimensión Confianza entre grupos.
  - Datos aproximadamente normales y n suficiente → **t de Student** para muestras
    independientes; reportar tamaño del efecto **d de Cohen**.
  - n pequeño o datos ordinales/no normales → **U de Mann–Whitney**; reportar **r**
    como tamaño del efecto. Dado que los ítems Likert son ordinales y el n de tesis
    suele ser pequeño, **la prueba no paramétrica suele ser la opción más
    defendible**.
- **H2 (mediación):** correlación (Spearman) entre Comprensión y Confianza; si se
  busca formalizar la mediación, un modelo de mediación simple (p. ej. método de
  Baron & Kenny o bootstrap de PROCESS), advirtiendo que requiere n mayor.
- **Casos repetidos por participante:** si se analizan los 4–6 casos de cada
  persona, usar un modelo de **medidas repetidas / efectos mixtos** (participante
  como efecto aleatorio) para no violar la independencia de las observaciones.
- **Comparaciones múltiples:** si se contrastan varias dimensiones, ajustar α
  (p. ej. **corrección de Bonferroni** o control de FDR) y declararlo.

### 7.3 Reporte (para el cuerpo de la tesis)
- Tabla resumen: dimensión × grupo (media ± DE), estadístico de prueba, p-valor,
  tamaño del efecto.
- Triangular lo cuantitativo con lo cualitativo (sección 6): los temas abiertos
  deben *explicar* los números, no contradecirlos sin discusión.
- **Vincular con la fidelidad:** contrastar estos hallazgos con `xai_faithfulness.md`.
  El caso ideal para la tesis es **explicaciones fieles *y* percibidas como
  comprensibles/confiables**. Si la atención resultó poco fiel pero los usuarios
  confían mucho en ella, discutir el riesgo de **confianza injustificada**
  (over-trust) como hallazgo relevante.

---

## 8. Ética y validez

- **Consentimiento informado** y anonimización de los datos de participantes.
- **Datos de estudiantes:** los casos deben anonimizarse/seudonimizarse; si se usan
  perfiles reales del dataset Moodle, garantizar que no sean reidentificables.
- **Amenazas a la validez a documentar:** muestra pequeña y por conveniencia
  (validez externa), efecto de novedad de la herramienta, posible sesgo del
  facilitador, y el hecho de que percibir confianza no implica que la confianza esté
  justificada (de ahí la importancia de cruzar con la fidelidad).

---

## 9. Checklist de ejecución

- [ ] Protocolo y consentimiento aprobados.
- [ ] Casos de estímulo seleccionados (alto/bajo/ambiguo) y validados con un experto.
- [ ] Interfaz A (sin explicación) y B (con explicación) listas y equivalentes salvo
      por la explicación.
- [ ] Asignación aleatoria a grupos y aleatorización del orden de casos.
- [ ] Cuestionario digitalizado con ítems (R) marcados para recodificar.
- [ ] Plan de análisis pre-registrado (pruebas, tamaño de efecto, ajuste de α).
- [ ] Prueba piloto con 2–3 personas antes del estudio formal.
