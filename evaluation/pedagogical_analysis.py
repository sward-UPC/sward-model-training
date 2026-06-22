"""Análisis pedagógico / de impacto para SWARD.

Objetivo
--------
Generar EVIDENCIA de que el tutor adaptativo ayuda a aprender (no solo que el
sistema existe). Lee un JSON de interacciones y produce:

  1. Curvas de aprendizaje  -> ¿el acierto mejora con el refuerzo en el tiempo?
  2. Preferencia de formato  -> ¿cada alumno rinde mejor en su formato "fuerte"?
                                ¿usar ese formato se asocia a mejores aciertos?
  3. Pre/post                -> desempeño temprano vs tardío por alumno (proxy
                                de mejora / ganancia de aprendizaje).

Salida: consola + outputs/pedagogical_report.md (hallazgos + tablas + CAVEATS).

Formatos de entrada soportados (autodetectados)
-----------------------------------------------
A) Dataset KT  (outputs/moodle_kt_dataset.json):
   {
     "concept_index": {"<nombre>": <int>, ...},
     "sequences": [
        {"student": <id>, "course": <id>,
         "concepts": [<int>, ...], "responses": [0/1, ...],
         # opcionales (si la ingesta los registra):
         "tipos_recurso": ["practica_generada", "lectura_generada", ...],
         "es_vista": [false, ...], "fechas": ["...", ...]}
     ]
   }

B) training-data API  (GET /api/v1/dashboard/training-data):
   [{"estudiante_id": <id>, "concepto": <str|int>, "correcta": 0/1,
     "orden": <int>,
     # opcionales:
     "tipo_recurso": "video_generado", "es_vista": false, "fecha": "..."}]

C) Lista plana de interacciones (mismo espíritu que B, claves flexibles).

Cuando NO hay 'tipo_recurso' en los datos, el análisis de preferencia de
formato se omite con un aviso claro (es el bloque que más datos requiere).

Diseño
------
- Imports científicos PEREZOSOS (scipy / numpy). El módulo compila e incluso
  corre sin ellos: hay implementaciones puras de respaldo (Mann-Whitney U con
  aproximación normal, medianas, rangos). Declaramos siempre qué motor se usó.
- Honestidad estadística: con n pequeño reportamos tamaños de efecto y CAVEATS;
  no afirmamos causalidad a partir de datos observacionales.

Uso
---
  python evaluation/pedagogical_analysis.py \
      --input outputs/moodle_kt_dataset.json \
      --output outputs/pedagogical_report.md \
      --early-frac 0.4 --late-frac 0.4

Variables de entorno equivalentes:
  PEDA_INPUT, PEDA_OUTPUT, PEDA_EARLY_FRAC, PEDA_LATE_FRAC
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

# --------------------------------------------------------------------------- #
# Estructuras                                                                  #
# --------------------------------------------------------------------------- #


class Interaccion:
    """Una interacción normalizada (independiente del formato de entrada)."""

    __slots__ = ("estudiante", "concepto", "correcta", "orden", "tipo", "es_vista")

    def __init__(
        self,
        estudiante: Any,
        concepto: Any,
        correcta: int,
        orden: int,
        tipo: Optional[str] = None,
        es_vista: Optional[bool] = None,
    ) -> None:
        self.estudiante = estudiante
        self.concepto = concepto
        self.correcta = int(correcta)
        self.orden = int(orden)
        self.tipo = tipo
        self.es_vista = es_vista

    def __repr__(self) -> str:  # pragma: no cover - debug
        return (
            f"Interaccion(est={self.estudiante!r}, con={self.concepto!r}, "
            f"ok={self.correcta}, orden={self.orden}, tipo={self.tipo!r})"
        )


# --------------------------------------------------------------------------- #
# Carga / normalización                                                        #
# --------------------------------------------------------------------------- #


def _coerce_correcta(v: Any) -> int:
    """Acepta 0/1, bool, 'True'/'False', notas 0..1 o 0..100 (>=0.5 / >=50)."""
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)):
        f = float(v)
        if f in (0.0, 1.0):
            return int(f)
        # nota normalizada (0..1) o porcentual (0..100): umbral de aprobación
        if 0.0 < f < 1.0:
            return 1 if f >= 0.5 else 0
        return 1 if f >= 50.0 else 0
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "si", "sí", "yes", "correcta", "correcto"):
        return 1
    return 0


def cargar_interacciones(ruta: Path) -> tuple[list[Interaccion], dict[int, str]]:
    """Carga y normaliza interacciones desde cualquiera de los formatos.

    Devuelve (lista_interacciones, mapa_indice_concepto->nombre).
    """
    raw = json.loads(Path(ruta).read_text(encoding="utf-8"))
    nombre_concepto: dict[int, str] = {}
    inter: list[Interaccion] = []

    # --- Formato A: dataset KT con 'sequences' ----------------------------- #
    if isinstance(raw, dict) and "sequences" in raw:
        idx = raw.get("concept_index", {}) or {}
        nombre_concepto = {int(v): k for k, v in idx.items()}
        for seq in raw["sequences"]:
            est = seq.get("student", seq.get("estudiante_id"))
            conceptos = seq.get("concepts", [])
            respuestas = seq.get("responses", [])
            tipos = seq.get("tipos_recurso") or seq.get("tipos") or []
            vistas = seq.get("es_vista") or []
            for i, (c, r) in enumerate(zip(conceptos, respuestas)):
                inter.append(
                    Interaccion(
                        estudiante=est,
                        concepto=c,
                        correcta=_coerce_correcta(r),
                        orden=i,  # el orden en la secuencia ya es temporal
                        tipo=(tipos[i] if i < len(tipos) else None),
                        es_vista=(vistas[i] if i < len(vistas) else None),
                    )
                )
        return inter, nombre_concepto

    # --- Formato B/C: lista plana ------------------------------------------ #
    if isinstance(raw, list):
        registros: Iterable[dict] = raw
    elif isinstance(raw, dict) and "interactions" in raw:
        registros = raw["interactions"]
    elif isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
        registros = raw["data"]
    else:
        raise ValueError(
            "Formato de entrada no reconocido. Se espera dict con 'sequences' "
            "o una lista de interacciones planas."
        )

    for rec in registros:
        est = rec.get("estudiante_id", rec.get("student", rec.get("user_id")))
        con = rec.get("concepto", rec.get("concept", rec.get("skill")))
        ok = rec.get("correcta", rec.get("correct", rec.get("nota", rec.get("response"))))
        orden = rec.get("orden", rec.get("order", rec.get("seq", 0)))
        tipo = rec.get("tipo_recurso", rec.get("tipo", rec.get("resource_type")))
        vista = rec.get("es_vista", rec.get("is_view"))
        inter.append(
            Interaccion(
                estudiante=est,
                concepto=con,
                correcta=_coerce_correcta(ok),
                orden=int(orden) if orden is not None else 0,
                tipo=tipo,
                es_vista=vista,
            )
        )
    return inter, nombre_concepto


# --------------------------------------------------------------------------- #
# Utilidades estadísticas (puras + lazy)                                       #
# --------------------------------------------------------------------------- #


def _try_scipy():
    """Importa scipy.stats si está disponible; si no, None."""
    try:  # import perezoso
        from scipy import stats  # type: ignore

        return stats
    except Exception:
        return None


def media(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def mediana(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2.0


def desv_std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = media(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def mann_whitney_u_puro(a: list[float], b: list[float]) -> tuple[float, float, str]:
    """Mann-Whitney U con aproximación normal (corrección por empates).

    Devuelve (U, p_dos_colas, motor). Fallback cuando no hay scipy. Válido como
    aproximación con n moderado; con n muy pequeño el p-valor es orientativo.
    """
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan"), "puro"
    combinado = [(v, 0) for v in a] + [(v, 1) for v in b]
    combinado.sort(key=lambda t: t[0])

    # rangos con promedio de empates
    rangos = [0.0] * len(combinado)
    i = 0
    empates: dict[float, int] = defaultdict(int)
    while i < len(combinado):
        j = i
        while j < len(combinado) and combinado[j][0] == combinado[i][0]:
            j += 1
        rango_prom = (i + 1 + j) / 2.0  # rangos 1-based: promedio de [i+1 .. j]
        for k in range(i, j):
            rangos[k] = rango_prom
        empates[combinado[i][0]] = j - i
        i = j

    r1 = sum(rangos[k] for k in range(len(combinado)) if combinado[k][1] == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    n = n1 + n2
    mu_u = n1 * n2 / 2.0
    # corrección por empates en la varianza
    sum_t = sum(t**3 - t for t in empates.values())
    var_u = (n1 * n2 / 12.0) * ((n + 1) - sum_t / (n * (n - 1))) if n > 1 else 0.0
    if var_u <= 0:
        return u, float("nan"), "puro"
    z = (u - mu_u) / math.sqrt(var_u)
    # p dos colas vía función de error
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return u, max(0.0, min(1.0, p)), "puro (aprox. normal)"


def mann_whitney(a: list[float], b: list[float]) -> dict[str, Any]:
    """Mann-Whitney U usando scipy si está; si no, implementación pura."""
    stats = _try_scipy()
    if stats is not None and a and b:
        try:
            res = stats.mannwhitneyu(a, b, alternative="two-sided")
            return {"U": float(res.statistic), "p": float(res.pvalue), "motor": "scipy"}
        except Exception:
            pass
    u, p, motor = mann_whitney_u_puro(a, b)
    return {"U": u, "p": p, "motor": motor}


def cliffs_delta(a: list[float], b: list[float]) -> float:
    """Cliff's delta: tamaño de efecto no paramétrico en [-1, 1].

    >0 indica que 'a' tiende a ser mayor que 'b'. Interpretación habitual:
    |d|<0.147 insignificante, <0.33 pequeño, <0.474 mediano, si no grande.
    """
    if not a or not b:
        return float("nan")
    mayor = menor = 0
    for x in a:
        for y in b:
            if x > y:
                mayor += 1
            elif x < y:
                menor += 1
    return (mayor - menor) / (len(a) * len(b))


def cohen_d(a: list[float], b: list[float]) -> float:
    """Cohen's d con desviación combinada (paramétrico, referencial)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    na, nb = len(a), len(b)
    sa, sb = desv_std(a), desv_std(b)
    sp2 = ((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2)
    if sp2 <= 0:
        return float("nan")
    return (media(a) - media(b)) / math.sqrt(sp2)


def interpreta_cliffs(d: float) -> str:
    if d != d:  # nan
        return "n/d"
    ad = abs(d)
    if ad < 0.147:
        return "insignificante"
    if ad < 0.33:
        return "pequeño"
    if ad < 0.474:
        return "mediano"
    return "grande"


# --------------------------------------------------------------------------- #
# 1) Curvas de aprendizaje                                                     #
# --------------------------------------------------------------------------- #


def curvas_de_aprendizaje(
    inter: list[Interaccion], nombre_concepto: dict[int, str]
) -> dict[str, Any]:
    """Acierto promedio por POSICIÓN de interacción dentro de cada concepto.

    Para cada concepto agrupamos las respuestas de todos los alumnos por su
    n-ésima exposición a ese concepto (1ra, 2da, ...). Si la mayoría de los
    alumnos tiene una sola exposición por concepto, no hay curva intra-concepto
    posible y se reporta como tal (caveat).

    También calculamos una curva GLOBAL por posición de interacción (1ra, 2da,
    ... del alumno) como proxy de progreso a lo largo de la sesión/curso.
    """
    # --- por concepto, por n-ésima exposición ----------------------------- #
    # estructura: concepto -> posicion(1-based) -> [aciertos]
    por_concepto: dict[Any, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    # contador de exposición por (estudiante, concepto)
    exp: dict[tuple, int] = defaultdict(int)

    por_estudiante = _agrupar_por_estudiante(inter)
    for _est, items in por_estudiante.items():
        for it in items:
            exp[(it.estudiante, it.concepto)] += 1
            pos = exp[(it.estudiante, it.concepto)]
            por_concepto[it.concepto][pos].append(it.correcta)

    curvas_concepto = {}
    multi_exposicion = 0
    for con, posiciones in por_concepto.items():
        max_pos = max(posiciones)
        if max_pos >= 2:
            multi_exposicion += 1
        nombre = nombre_concepto.get(con, str(con)) if isinstance(con, int) else str(con)
        curvas_concepto[nombre] = {
            pos: {"n": len(v), "acierto": media([float(x) for x in v])}
            for pos, v in sorted(posiciones.items())
        }

    # --- curva global por posición de interacción del alumno --------------- #
    pos_global: dict[int, list[int]] = defaultdict(list)
    for _est, items in por_estudiante.items():
        for i, it in enumerate(items, start=1):
            pos_global[i].append(it.correcta)
    curva_global = {
        pos: {"n": len(v), "acierto": media([float(x) for x in v])}
        for pos, v in sorted(pos_global.items())
    }

    # tendencia simple (pendiente OLS) de la curva global
    pendiente = _pendiente_ols(
        [float(p) for p in curva_global],
        [curva_global[p]["acierto"] for p in curva_global],
    )

    return {
        "curvas_concepto": curvas_concepto,
        "curva_global": curva_global,
        "pendiente_global": pendiente,
        "conceptos_con_refuerzo": multi_exposicion,
        "conceptos_totales": len(por_concepto),
    }


def _pendiente_ols(xs: list[float], ys: list[float]) -> float:
    """Pendiente de regresión lineal simple (mínimos cuadrados)."""
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = media(xs), media(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return float("nan")
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return num / den


# --------------------------------------------------------------------------- #
# 2) Preferencia de formato                                                    #
# --------------------------------------------------------------------------- #


def analisis_preferencia_formato(inter: list[Interaccion]) -> dict[str, Any]:
    """¿Cada alumno rinde mejor en su formato "fuerte"?

    Requiere que las interacciones tengan 'tipo' (tipo_recurso). Si no hay
    ninguna con tipo, devuelve {'disponible': False}.

    Procedimiento:
      1. Por alumno, calcula acierto promedio por formato.
      2. Define el formato "fuerte" del alumno (mayor acierto, con mínimo de
         observaciones).
      3. Compara, vía Mann-Whitney + Cliff's delta, los aciertos en el formato
         fuerte vs. el resto de formatos (a nivel de interacción).
    """
    con_tipo = [it for it in inter if it.tipo]
    if not con_tipo:
        return {"disponible": False, "motivo": "ninguna interacción tiene 'tipo_recurso'"}

    por_estudiante = _agrupar_por_estudiante(con_tipo)
    MIN_OBS = 2  # mínimo de observaciones por formato para considerarlo

    aciertos_fuerte: list[float] = []
    aciertos_resto: list[float] = []
    detalle_alumnos = []
    formatos_vistos: set[str] = set()

    for est, items in por_estudiante.items():
        por_formato: dict[str, list[int]] = defaultdict(list)
        for it in items:
            por_formato[it.tipo].append(it.correcta)
            formatos_vistos.add(it.tipo)
        candidatos = {f: v for f, v in por_formato.items() if len(v) >= MIN_OBS}
        if len(candidatos) < 2:
            continue  # no se puede comparar fuerte vs resto en este alumno
        fuerte = max(candidatos, key=lambda f: media([float(x) for x in candidatos[f]]))
        for f, v in por_formato.items():
            vals = [float(x) for x in v]
            if f == fuerte:
                aciertos_fuerte.extend(vals)
            else:
                aciertos_resto.extend(vals)
        detalle_alumnos.append(
            {
                "estudiante": est,
                "formato_fuerte": fuerte,
                "acierto_fuerte": media([float(x) for x in por_formato[fuerte]]),
                "acierto_resto": media(
                    [float(x) for f, v in por_formato.items() if f != fuerte for x in v]
                ),
            }
        )

    resultado = {
        "disponible": True,
        "n_alumnos_comparables": len(detalle_alumnos),
        "formatos": sorted(formatos_vistos),
        "acierto_medio_fuerte": media(aciertos_fuerte),
        "acierto_medio_resto": media(aciertos_resto),
        "n_obs_fuerte": len(aciertos_fuerte),
        "n_obs_resto": len(aciertos_resto),
        "detalle_alumnos": detalle_alumnos,
    }
    if aciertos_fuerte and aciertos_resto:
        mw = mann_whitney(aciertos_fuerte, aciertos_resto)
        resultado["mann_whitney"] = mw
        resultado["cliffs_delta"] = cliffs_delta(aciertos_fuerte, aciertos_resto)
        resultado["cliffs_interpretacion"] = interpreta_cliffs(resultado["cliffs_delta"])
        resultado["cohen_d"] = cohen_d(aciertos_fuerte, aciertos_resto)

    # --- efecto "posterior": ¿una interacción en el formato fuerte se asocia a
    #     mejor acierto en la SIGUIENTE interacción del alumno? -------------- #
    resultado["efecto_posterior"] = _efecto_posterior_formato(por_estudiante)
    return resultado


def _efecto_posterior_formato(
    por_estudiante: dict[Any, list[Interaccion]],
) -> dict[str, Any]:
    """Acierto en la interacción t+1 según si t fue en el formato fuerte global.

    Aproximación observacional (no causal): define para cada alumno su formato
    fuerte y mide el acierto de la interacción siguiente a una en formato fuerte
    vs siguiente a una en otro formato.
    """
    sig_tras_fuerte: list[float] = []
    sig_tras_otro: list[float] = []
    for _est, items in por_estudiante.items():
        por_formato: dict[str, list[int]] = defaultdict(list)
        for it in items:
            por_formato[it.tipo].append(it.correcta)
        candidatos = {f: v for f, v in por_formato.items() if len(v) >= 2}
        if len(candidatos) < 2:
            continue
        fuerte = max(candidatos, key=lambda f: media([float(x) for x in candidatos[f]]))
        for i in range(len(items) - 1):
            sig = float(items[i + 1].correcta)
            if items[i].tipo == fuerte:
                sig_tras_fuerte.append(sig)
            else:
                sig_tras_otro.append(sig)
    out: dict[str, Any] = {
        "acierto_siguiente_tras_fuerte": media(sig_tras_fuerte),
        "acierto_siguiente_tras_otro": media(sig_tras_otro),
        "n_tras_fuerte": len(sig_tras_fuerte),
        "n_tras_otro": len(sig_tras_otro),
    }
    if sig_tras_fuerte and sig_tras_otro:
        out["mann_whitney"] = mann_whitney(sig_tras_fuerte, sig_tras_otro)
        out["cliffs_delta"] = cliffs_delta(sig_tras_fuerte, sig_tras_otro)
    return out


# --------------------------------------------------------------------------- #
# 3) Pre/post                                                                  #
# --------------------------------------------------------------------------- #


def analisis_pre_post(
    inter: list[Interaccion], early_frac: float, late_frac: float
) -> dict[str, Any]:
    """Compara desempeño temprano (pre) vs tardío (post) por alumno.

    Para cada alumno con suficientes interacciones, toma la fracción inicial
    (pre) y la final (post) de su secuencia ordenada. La GANANCIA NORMALIZADA
    de aprendizaje (estilo Hake) se aproxima como:

        g = (post - pre) / (1 - pre)   si pre < 1, indefinida si pre == 1.

    Reporta pre, post y g por alumno + agregados + test pareado no paramétrico
    (Wilcoxon vía scipy si está; si no, signo + Mann-Whitney como respaldo).
    """
    por_estudiante = _agrupar_por_estudiante(inter)
    MIN_INTER = 4  # mínimo para partir en pre/post de forma sensata

    pres: list[float] = []
    posts: list[float] = []
    ganancias: list[float] = []
    detalle = []
    descartados = 0

    for est, items in por_estudiante.items():
        n = len(items)
        if n < MIN_INTER:
            descartados += 1
            continue
        n_pre = max(1, int(round(n * early_frac)))
        n_post = max(1, int(round(n * late_frac)))
        # evita solapamiento: si se solapan, recorta al mínimo viable
        if n_pre + n_post > n:
            n_pre = n // 2
            n_post = n - n_pre
        pre_vals = [float(it.correcta) for it in items[:n_pre]]
        post_vals = [float(it.correcta) for it in items[-n_post:]]
        pre, post = media(pre_vals), media(post_vals)
        pres.append(pre)
        posts.append(post)
        g = (post - pre) / (1 - pre) if pre < 1.0 else float("nan")
        if g == g:
            ganancias.append(g)
        detalle.append(
            {"estudiante": est, "n": n, "pre": pre, "post": post, "ganancia_norm": g}
        )

    resultado: dict[str, Any] = {
        "n_alumnos": len(detalle),
        "descartados_por_pocas_interacciones": descartados,
        "early_frac": early_frac,
        "late_frac": late_frac,
        "pre_medio": media(pres),
        "post_medio": media(posts),
        "delta_medio": (media(posts) - media(pres)) if pres else float("nan"),
        "ganancia_norm_media": media(ganancias) if ganancias else float("nan"),
        "ganancia_norm_mediana": mediana(ganancias) if ganancias else float("nan"),
        "n_mejoran": sum(1 for d in detalle if d["post"] > d["pre"]),
        "n_empeoran": sum(1 for d in detalle if d["post"] < d["pre"]),
        "n_iguales": sum(1 for d in detalle if d["post"] == d["pre"]),
        "detalle": detalle,
    }

    # test pareado: Wilcoxon (scipy) o prueba de signos (puro)
    if pres and posts:
        resultado["test_pareado"] = _test_pareado(pres, posts)
        resultado["cohen_d_pareado"] = cohen_d(posts, pres)
    return resultado


def _test_pareado(pre: list[float], post: list[float]) -> dict[str, Any]:
    stats = _try_scipy()
    difs = [b - a for a, b in zip(pre, post)]
    if stats is not None and any(d != 0 for d in difs):
        try:
            res = stats.wilcoxon(post, pre)
            return {"prueba": "Wilcoxon (scipy)", "estadistico": float(res.statistic),
                    "p": float(res.pvalue)}
        except Exception:
            pass
    # prueba de signos pura (binomial exacta, dos colas)
    pos = sum(1 for d in difs if d > 0)
    neg = sum(1 for d in difs if d < 0)
    n = pos + neg
    if n == 0:
        return {"prueba": "signos (puro)", "p": float("nan"),
                "nota": "sin diferencias no nulas"}
    k = min(pos, neg)
    # p dos colas = 2 * suma_{i=0}^{k} C(n,i) 0.5^n  (acotado a 1)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return {"prueba": "signos (binomial exacta, puro)", "n_pares_no_nulos": n,
            "positivos": pos, "negativos": neg, "p": min(1.0, p)}


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _agrupar_por_estudiante(inter: list[Interaccion]) -> dict[Any, list[Interaccion]]:
    """Agrupa por estudiante y ordena cada lista por 'orden'."""
    g: dict[Any, list[Interaccion]] = defaultdict(list)
    for it in inter:
        g[it.estudiante].append(it)
    for est in g:
        g[est].sort(key=lambda it: it.orden)
    return g


def _fmt(x: float, dec: int = 3) -> str:
    if isinstance(x, float) and x != x:
        return "n/d"
    if isinstance(x, float):
        return f"{x:.{dec}f}"
    return str(x)


# --------------------------------------------------------------------------- #
# Reporte                                                                      #
# --------------------------------------------------------------------------- #


def construir_reporte(
    inter: list[Interaccion],
    nombre_concepto: dict[int, str],
    curvas: dict[str, Any],
    formato: dict[str, Any],
    prepost: dict[str, Any],
    ruta_entrada: Path,
) -> str:
    n_inter = len(inter)
    estudiantes = {it.estudiante for it in inter}
    conceptos = {it.concepto for it in inter}
    acierto_global = media([float(it.correcta) for it in inter]) if inter else float("nan")
    n_est = len(estudiantes)
    scipy_on = _try_scipy() is not None

    L: list[str] = []
    a = L.append
    a("# Reporte de análisis pedagógico — SWARD\n")
    a("> Generado por `evaluation/pedagogical_analysis.py`. "
      "Análisis **observacional** sobre datos de interacciones; "
      "no establece causalidad por sí solo (ver CAVEATS).\n")
    a(f"- Fuente de datos: `{ruta_entrada}`")
    a(f"- Motor estadístico: {'scipy' if scipy_on else 'implementación pura (sin scipy)'}")
    a(f"- Interacciones: **{n_inter}** | Estudiantes: **{n_est}** | "
      f"Conceptos: **{len(conceptos)}**")
    a(f"- Acierto global: **{_fmt(acierto_global)}**\n")

    # --- Tamaño de muestra: caveat principal ------------------------------- #
    a("## Advertencia sobre tamaño de muestra (LEER PRIMERO)\n")
    avisos = []
    if n_est < 30:
        avisos.append(
            f"- Solo **{n_est} estudiantes**: por debajo del umbral habitual (n≈30) "
            "para inferencia robusta. Los p-valores son **orientativos**.")
    medias_por_est = [
        len(v) for v in _agrupar_por_estudiante(inter).values()
    ]
    med_long = media([float(x) for x in medias_por_est]) if medias_por_est else 0
    if med_long < 8:
        avisos.append(
            f"- Secuencias cortas (~{_fmt(med_long,1)} interacciones/alumno): "
            "limita curvas de aprendizaje intra-concepto y la partición pre/post.")
    if curvas["conceptos_con_refuerzo"] == 0:
        avisos.append(
            "- **Ningún concepto tiene exposiciones repetidas** (cada alumno ve "
            "cada concepto una sola vez): no se pueden trazar curvas de dominio "
            "intra-concepto; se usa la curva global por posición como proxy débil.")
    if not formato.get("disponible"):
        avisos.append(
            "- Los datos **no incluyen `tipo_recurso`**: el análisis de preferencia "
            "de formato (núcleo de la hipótesis adaptativa) NO pudo ejecutarse. "
            "Se requiere registrar el formato de cada interacción.")
    if not avisos:
        avisos.append("- Sin alertas críticas de muestra detectadas.")
    L.extend(avisos)
    a("")

    # --- 1. Curvas de aprendizaje ------------------------------------------ #
    a("## 1. Curvas de aprendizaje\n")
    a("Proxy de progreso: acierto promedio según la **posición** de la "
      "interacción dentro de la secuencia del alumno.\n")
    a("| Posición | n | Acierto |")
    a("|---:|---:|---:|")
    for pos, d in curvas["curva_global"].items():
        a(f"| {pos} | {d['n']} | {_fmt(d['acierto'])} |")
    a("")
    a(f"- Pendiente OLS de la curva global: **{_fmt(curvas['pendiente_global'],4)}** "
      "(positiva = el acierto tiende a subir a lo largo de la secuencia).")
    a(f"- Conceptos con refuerzo (≥2 exposiciones): "
      f"**{curvas['conceptos_con_refuerzo']}/{curvas['conceptos_totales']}**.")
    if curvas["conceptos_con_refuerzo"] > 0:
        a("\nCurvas intra-concepto (primeras exposiciones):\n")
        a("| Concepto | Pos | n | Acierto |")
        a("|:--|---:|---:|---:|")
        for nombre, posiciones in list(curvas["curvas_concepto"].items()):
            if max(posiciones) >= 2:
                for pos, d in posiciones.items():
                    a(f"| {nombre} | {pos} | {d['n']} | {_fmt(d['acierto'])} |")
    a("")

    # --- 2. Preferencia de formato ----------------------------------------- #
    a("## 2. Análisis de preferencia de formato\n")
    if not formato.get("disponible"):
        a(f"**No disponible.** Motivo: {formato.get('motivo')}.\n")
        a("Para habilitarlo, cada interacción debe registrar `tipo_recurso` "
          "(p. ej. `practica_generada`, `lectura_generada`, `quiz_generado`, "
          "`video_generado`). Ver sección de datos requeridos en el README.\n")
    else:
        a(f"- Alumnos comparables (≥2 formatos con ≥2 obs): "
          f"**{formato['n_alumnos_comparables']}**")
        a(f"- Formatos observados: {', '.join(formato['formatos'])}")
        a(f"- Acierto medio en formato **fuerte**: **{_fmt(formato['acierto_medio_fuerte'])}** "
          f"(n={formato['n_obs_fuerte']})")
        a(f"- Acierto medio en **otros** formatos: **{_fmt(formato['acierto_medio_resto'])}** "
          f"(n={formato['n_obs_resto']})")
        if "mann_whitney" in formato:
            mw = formato["mann_whitney"]
            a(f"- Mann-Whitney U={_fmt(mw['U'],1)}, p={_fmt(mw['p'],4)} "
              f"(motor: {mw['motor']})")
            a(f"- Tamaño de efecto Cliff's δ={_fmt(formato['cliffs_delta'])} "
              f"({formato['cliffs_interpretacion']}); Cohen's d={_fmt(formato.get('cohen_d'))}")
        ep = formato.get("efecto_posterior", {})
        if ep.get("n_tras_fuerte") and ep.get("n_tras_otro"):
            a("\nEfecto posterior (acierto en la interacción siguiente):")
            a(f"- Tras formato fuerte: **{_fmt(ep['acierto_siguiente_tras_fuerte'])}** "
              f"(n={ep['n_tras_fuerte']})")
            a(f"- Tras otro formato: **{_fmt(ep['acierto_siguiente_tras_otro'])}** "
              f"(n={ep['n_tras_otro']})")
            if "mann_whitney" in ep:
                a(f"- Mann-Whitney p={_fmt(ep['mann_whitney']['p'],4)}; "
                  f"Cliff's δ={_fmt(ep.get('cliffs_delta'))}")
    a("")

    # --- 3. Pre/post ------------------------------------------------------- #
    a("## 3. Análisis pre/post (proxy de mejora)\n")
    a(f"- Fracción temprana (pre): {prepost['early_frac']} | "
      f"tardía (post): {prepost['late_frac']}")
    a(f"- Alumnos analizados: **{prepost['n_alumnos']}** "
      f"(descartados por <4 interacciones: {prepost['descartados_por_pocas_interacciones']})")
    a(f"- Pre medio: **{_fmt(prepost['pre_medio'])}** | "
      f"Post medio: **{_fmt(prepost['post_medio'])}** | "
      f"Δ medio: **{_fmt(prepost['delta_medio'])}**")
    a(f"- Ganancia normalizada (Hake) media: **{_fmt(prepost['ganancia_norm_media'])}** | "
      f"mediana: **{_fmt(prepost['ganancia_norm_mediana'])}**")
    a(f"- Mejoran: {prepost['n_mejoran']} | Empeoran: {prepost['n_empeoran']} | "
      f"Iguales: {prepost['n_iguales']}")
    if "test_pareado" in prepost:
        tp = prepost["test_pareado"]
        a(f"- Test pareado: {tp['prueba']}, p={_fmt(tp.get('p'),4)}")
        a(f"- Cohen's d pareado: {_fmt(prepost.get('cohen_d_pareado'))}")
    a("")

    # --- CAVEATS ----------------------------------------------------------- #
    a("## CAVEATS metodológicos\n")
    a("- **Observacional, no causal.** Sin grupo de control aleatorizado no se "
      "puede atribuir la mejora a la recomendación adaptativa. Ver "
      "`experiment_design.md` para el diseño cuasi-experimental propuesto.")
    a("- **Ganancia pre/post intra-sujeto** puede reflejar maduración, práctica "
      "o efecto test, no necesariamente la intervención.")
    a("- **Selección de formato fuerte** se define con los mismos datos que se "
      "evalúan (riesgo de sobreajuste / regresión a la media). Idealmente se "
      "define el formato fuerte en una ventana y se evalúa en otra posterior.")
    a("- Con muestras pequeñas, los **tamaños de efecto** (Cliff's δ) son más "
      "informativos que los p-valores.")
    a("- Acierto binario derivado de notas con umbral 0.5 (consistente con la "
      "ingesta): pierde granularidad respecto a la nota continua.")
    a("")
    return "\n".join(L)


def imprimir_consola(reporte_md: str) -> None:
    # Consola: versión sin sintaxis de tabla pesada, pero útil tal cual.
    print(reporte_md)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    base = Path(__file__).resolve().parent.parent  # sward-model-training/
    p = argparse.ArgumentParser(
        description="Análisis pedagógico / de impacto para SWARD (offline, sobre JSON)."
    )
    p.add_argument(
        "--input",
        default=os.environ.get("PEDA_INPUT", str(base / "outputs" / "moodle_kt_dataset.json")),
        help="JSON de interacciones (dataset KT o salida de training-data API).",
    )
    p.add_argument(
        "--output",
        default=os.environ.get("PEDA_OUTPUT", str(base / "outputs" / "pedagogical_report.md")),
        help="Ruta del reporte Markdown de salida.",
    )
    p.add_argument(
        "--early-frac",
        type=float,
        default=float(os.environ.get("PEDA_EARLY_FRAC", "0.4")),
        help="Fracción inicial de la secuencia usada como 'pre' (def. 0.4).",
    )
    p.add_argument(
        "--late-frac",
        type=float,
        default=float(os.environ.get("PEDA_LATE_FRAC", "0.4")),
        help="Fracción final de la secuencia usada como 'post' (def. 0.4).",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    ruta_in = Path(args.input)
    if not ruta_in.exists():
        print(f"[ERROR] No existe el archivo de entrada: {ruta_in}")
        return 2

    inter, nombre_concepto = cargar_interacciones(ruta_in)
    if not inter:
        print("[ERROR] No se cargaron interacciones (archivo vacío o formato inesperado).")
        return 2

    curvas = curvas_de_aprendizaje(inter, nombre_concepto)
    formato = analisis_preferencia_formato(inter)
    prepost = analisis_pre_post(inter, args.early_frac, args.late_frac)

    reporte = construir_reporte(
        inter, nombre_concepto, curvas, formato, prepost, ruta_in
    )

    ruta_out = Path(args.output)
    ruta_out.parent.mkdir(parents=True, exist_ok=True)
    ruta_out.write_text(reporte, encoding="utf-8")

    imprimir_consola(reporte)
    print(f"\n[OK] Reporte escrito en: {ruta_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
