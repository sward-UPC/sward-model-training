# Evaluación de Explicabilidad (XAI) del SAKT — SWARD

Este directorio contiene las herramientas para **evaluar** la explicabilidad del
modelo SAKT de *knowledge tracing*, el núcleo del título de la tesis
(*"IA Explicable"*). No basta con **mostrar** los pesos de atención en el panel:
hay que **demostrar** que esos pesos explican bien la decisión del modelo.

| Archivo | Qué evalúa | Tipo |
|---|---|---|
| `xai_faithfulness.py` | **Fidelidad** de las explicaciones por atención (¿los pesos determinan la predicción?) | Cuantitativo / automático |
| `xai_user_study.md` | **Comprensión y confianza** de usuarios reales ante las explicaciones | Cualitativo+cuantitativo / con personas |
| `xai_faithfulness.md` | Reporte de resultados generado por el script (tabla + interpretación) | Salida (en `outputs/`) |

---

## ¿Qué es la *faithfulness* (fidelidad) y por qué importa?

En XAI, una explicación es **fiel** (*faithful*) cuando refleja de verdad el proceso
que el modelo usó para decidir, y no una historia plausible pero desconectada del
cómputo real. La atención es la explicación más usada en SWARD, pero existe un
debate conocido en la literatura: *"Attention is not Explanation"* (Jain & Wallace,
2019) mostró que los pesos de atención pueden **no** coincidir con qué entradas
realmente importan. Por eso, mostrar atención sin validarla es insuficiente para
afirmar que el sistema es explicable.

La fidelidad se mide perturbando la entrada y observando la predicción
(DeYoung et al., 2020, *"ERASER"*):

- **Comprehensiveness (exhaustividad).** Si **borro** las interacciones que la
  atención marcó como importantes y la predicción **se desploma**, la explicación
  era exhaustiva: capturaba la evidencia clave.
- **Sufficiency (suficiencia).** Si **conservo sólo** las interacciones importantes
  (borro el resto) y la predicción **apenas cambia**, la explicación era suficiente:
  con eso basta para reproducir la decisión.
- **Comparación contra azar.** El paso decisivo: repetir el borrado con
  interacciones **aleatorias**. La atención sólo es fiel si borrar lo que destaca
  impacta **más** que borrar interacciones cualesquiera. Si no le gana al azar, los
  pesos no son mejor explicación que tirar los dados.

Esto le da a la tesis una afirmación **medible** y defendible: *"las explicaciones
por atención del SAKT son fieles según comprehensiveness/sufficiency, superando al
baseline aleatorio en X"*, en lugar de simplemente *"mostramos atención"*.

---

## Cómo correr el script

> Requiere `torch` y `pykt` instalados (los mismos del entrenamiento). El archivo
> está escrito con **imports perezosos**, así que pasa `python -m py_compile` sin
> esas dependencias, pero la ejecución real sí las necesita.

```bash
cd sward-model-training

# Por defecto usa outputs/sakt_moodle.pth y outputs/moodle_kt_dataset.json
python evaluation/xai_faithfulness.py

# Personalizado
python evaluation/xai_faithfulness.py \
  --checkpoint outputs/sakt_moodle.pth \
  --dataset    outputs/moodle_kt_dataset.json \
  --muestras   100 \
  --topk       1 2 \
  --rand-reps  5 \
  --out        outputs/xai_faithfulness.md
```

Parámetros:
- `--topk`: nº de interacciones de mayor atención que se borran/conservan en cada
  prueba. Como las secuencias del dataset Moodle son cortas (~6–7 interacciones,
  es decir ~5–6 de "pasado"), usar **k pequeño** (1, 2).
- `--muestras`: submuestreo determinista (semilla fija) para acotar el cómputo.
- `--rand-reps`: repeticiones del baseline aleatorio por secuencia (promediadas).

Salida: métricas agregadas por consola **y** un reporte Markdown en `--out`
(tabla + interpretación redactada para el cuerpo/anexo de la tesis).

---

## Cómo interpretar el reporte

La métrica central es **Δ comprehensiveness = atención − azar**:

- **Δ claramente > 0** y la atención **gana al azar en > 50 %** de las secuencias →
  las explicaciones por atención son **fieles**. Es el resultado que respalda el
  componente de IA Explicable.
- **Δ ≈ 0** → la atención no explica mejor que elegir al azar: reportarlo con
  honestidad, ampliar datos y/o complementar con otros métodos de atribución.
- **Sufficiency** cercana a 0 es buena señal (lo retenido basta para reproducir la
  predicción).

El script ya redacta un **veredicto automático** en `xai_faithfulness.md` según
estos umbrales, pero la interpretación final es responsabilidad del autor: debe
acompañarse del **tamaño de muestra** y, idealmente, de una **prueba de
significancia** (Wilcoxon pareado entre comprehensiveness de atención vs azar).

---

## Supuestos y limitaciones

- **"Borrar" = poner a PAD (token 0).** Es la posición vacía que el modelo ya ve en
  el entrenamiento (ver `training/train_sakt.py`), por lo que es el sustituto neutro
  natural. Es una perturbación por **oclusión**; otras opciones (re-muestrear,
  invertir respuesta) medirían matices distintos.
- **Confianza = |p − 0.5|·2.** Comprehensiveness/sufficiency se miden sobre la
  confianza de la salida binaria (sigmoide), no sobre la probabilidad cruda, para no
  depender de qué clase ganó.
- **Atención del último bloque y último paso**, idéntica a la que el microservicio
  expone como explicación (`modelo_sakt._real_prediccion`). Así se evalúa
  exactamente lo que ve el usuario.
- **Dataset pequeño y secuencias cortas.** El dataset Moodle actual tiene ~104
  secuencias de 6–7 interacciones; los resultados son indicativos. Con más datos
  reales de Moodle (y reentrenamiento) las conclusiones serán más robustas.

---

## Estudio con usuarios (complemento obligatorio)

La fidelidad responde *"¿la explicación es fiel al modelo?"*. Falta *"¿es útil y
confiable para las personas?"*. Eso lo cubre `xai_user_study.md`: cuestionario
Likert de comprensión/confianza/utilidad, diseño del experimento (n, grupos,
procedimiento) y plan de análisis. El caso ideal de la tesis es **explicaciones
fieles *y* percibidas como comprensibles**; cualquier divergencia (p. ej. confianza
alta sobre atención poco fiel) es en sí un hallazgo discutible.

---

## Referencias

- Jain, S. & Wallace, B. (2019). *Attention is not Explanation*. NAACL.
- Wiegreffe, S. & Pinter, Y. (2019). *Attention is not not Explanation*. EMNLP.
- DeYoung, J. et al. (2020). *ERASER: A Benchmark to Evaluate Rationalized NLP
  Models* (define comprehensiveness y sufficiency). ACL.
- Pandey, S. & Karypis, G. (2019). *A Self-Attentive model for Knowledge Tracing*
  (SAKT). EDM.
