"""Evaluación reproducible de modelos de Knowledge Tracing para la tesis SWARD.

Compara, mediante validación cruzada K-fold (k=5 por defecto), varios modelos de
Knowledge Tracing sobre el dataset real de Moodle exportado por
`export_moodle_to_pykt.py` (`outputs/moodle_kt_dataset.json`):

  - SAKT  (pyKT)  — Self-Attentive Knowledge Tracing.
  - DKT   (pyKT)  — Deep Knowledge Tracing (LSTM).
  - Baseline "global"      — predice siempre el promedio global de aciertos del
                             conjunto de entrenamiento (clasificador trivial).
  - Baseline "por-concepto"— predice el promedio de aciertos de cada concepto
                             (memoriza la dificultad media de cada sección Moodle);
                             si un concepto no se vio en train, cae al promedio global.

Para cada modelo reporta, promediado sobre los folds (media ± desviación
estándar): AUC, ACC, F1 y RMSE. Todas las métricas se calculan únicamente sobre
las posiciones VÁLIDAS de la secuencia (se respeta el padding/máscara), y la
predicción en el paso t corresponde a la respuesta del paso t+1 (formato KT:
"dado el pasado, predecir el siguiente acierto").

Salidas:
  - Tabla impresa en consola.
  - outputs/model_comparison.csv  (datos crudos por modelo).
  - outputs/model_comparison.md   (tabla markdown lista para la tesis).

Parametrización (CLI o variables de entorno), con valores por defecto sensatos:
  DATASET   / --dataset    JSON de entrada (default outputs/moodle_kt_dataset.json)
  KFOLDS    / --kfolds      número de folds de la CV (default 5)
  EPOCHS    / --epochs       épocas de entrenamiento de los modelos neuronales (default 60)
  SEQ_LEN   / --seq-len      longitud máxima de secuencia / padding (default 64)
  EMB_SIZE  / --emb-size     dimensión de embedding (default 64)
  HEADS     / --heads        cabezas de atención SAKT (default 4)
  LAYERS    / --layers       capas del encoder SAKT (default 2)
  DROPOUT   / --dropout      dropout (default 0.2)
  LR        / --lr           learning rate (default 1e-3)
  BATCH     / --batch        tamaño de batch (default 16)
  SEED      / --seed         semilla global para reproducibilidad (default 42)
  MODELS    / --models       lista separada por comas (default sakt,dkt,baseline_global,baseline_concept)
  OUTDIR    / --outdir       carpeta de salida (default outputs)

NOTA metodológica: torch/pyKT se importan DENTRO de las funciones (lazy) para
que el archivo se pueda `python -m py_compile` sin tener pyKT instalado y para
que los baselines (que no requieren pyKT) puedan correr de forma aislada.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

# ── Constante de padding (idéntica a export_moodle_to_pykt.py / train.py) ──────
PAD = -1


# ──────────────────────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────────────────────
def parse_config(argv: list[str] | None = None) -> dict:
    """Lee la configuración desde CLI con fallback a variables de entorno.

    Prioridad: argumento CLI explícito > variable de entorno > default.
    """

    def _env(name: str, default: str) -> str:
        return os.environ.get(name, default)

    p = argparse.ArgumentParser(description="Comparación de modelos KT (tesis SWARD).")
    p.add_argument("--dataset", default=_env("DATASET", "outputs/moodle_kt_dataset.json"))
    p.add_argument("--kfolds", type=int, default=int(_env("KFOLDS", "5")))
    p.add_argument("--epochs", type=int, default=int(_env("EPOCHS", "60")))
    p.add_argument("--seq-len", type=int, default=int(_env("SEQ_LEN", "64")))
    p.add_argument("--emb-size", type=int, default=int(_env("EMB_SIZE", "64")))
    p.add_argument("--heads", type=int, default=int(_env("HEADS", "4")))
    p.add_argument("--layers", type=int, default=int(_env("LAYERS", "2")))
    p.add_argument("--dropout", type=float, default=float(_env("DROPOUT", "0.2")))
    p.add_argument("--lr", type=float, default=float(_env("LR", "1e-3")))
    p.add_argument("--batch", type=int, default=int(_env("BATCH", "16")))
    p.add_argument("--seed", type=int, default=int(_env("SEED", "42")))
    p.add_argument(
        "--models",
        default=_env("MODELS", "sakt,dkt,baseline_global,baseline_concept"),
    )
    p.add_argument("--outdir", default=_env("OUTDIR", "outputs"))
    args = p.parse_args(argv)

    cfg = vars(args)
    cfg["models"] = [m.strip() for m in cfg["models"].split(",") if m.strip()]
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Carga de datos
# ──────────────────────────────────────────────────────────────────────────────
def cargar_dataset(path: str) -> tuple[list[dict], dict[str, int]]:
    """Carga el JSON de Moodle y devuelve (secuencias, concept_index).

    Cada secuencia es {concepts: [int...], responses: [0/1...]} con al menos 2
    interacciones (necesario para tener un par (pasado, siguiente)).
    """
    data = json.loads(Path(path).read_text())
    concept_index = data.get("concept_index", {})
    seqs = [
        {"concepts": list(map(int, s["concepts"])), "responses": list(map(int, s["responses"]))}
        for s in data["sequences"]
        if len(s["responses"]) >= 2
    ]
    return seqs, concept_index


def n_skills_de(seqs: list[dict]) -> int:
    """Número de conceptos distintos = max(indice) + 1 (índices 0..n-1)."""
    maxc = 0
    for s in seqs:
        for c in s["concepts"]:
            if c > maxc:
                maxc = c
    return maxc + 1


# ──────────────────────────────────────────────────────────────────────────────
# Particionado K-fold reproducible (sin sklearn para no depender de él aquí)
# ──────────────────────────────────────────────────────────────────────────────
def kfold_indices(n: int, k: int, seed: int) -> list[list[int]]:
    """Devuelve k listas de índices (folds) tras barajar de forma determinística.

    Usa random.Random(seed) para que la partición sea idéntica entre corridas.
    """
    import random

    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    folds: list[list[int]] = [[] for _ in range(k)]
    for i, v in enumerate(idx):
        folds[i % k].append(v)
    return folds


# ──────────────────────────────────────────────────────────────────────────────
# Métricas (sklearn si está disponible; si no, implementaciones puras)
# ──────────────────────────────────────────────────────────────────────────────
def _auc_puro(y_true: list[int], y_score: list[float]) -> float | None:
    """AUC-ROC por el estadístico de Mann-Whitney U (maneja empates con rango medio).

    Devuelve None si solo hay una clase (AUC indefinido).
    """
    pares = sorted(zip(y_score, y_true), key=lambda x: x[0])
    n = len(pares)
    # Rangos promedio para empates en el score.
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pares[j + 1][0] == pares[i][0]:
            j += 1
        rango_medio = (i + j) / 2.0 + 1.0  # rangos 1-indexados
        for t in range(i, j + 1):
            ranks[t] = rango_medio
        i = j + 1
    suma_rangos_pos = sum(r for r, (_, lab) in zip(ranks, pares) if lab == 1)
    n_pos = sum(1 for _, lab in pares if lab == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    return (suma_rangos_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def calcular_metricas(y_true: list[int], y_score: list[float], umbral: float = 0.5) -> dict:
    """AUC, ACC, F1 y RMSE. Usa sklearn si está; si no, implementaciones puras.

    - y_true: etiquetas 0/1 de las posiciones válidas.
    - y_score: probabilidad predicha de acierto en [0, 1].
    AUC puede ser None si el fold no contiene ambas clases.
    """
    y_pred = [1 if s >= umbral else 0 for s in y_score]

    # RMSE (siempre puro, es trivial y exacto).
    rmse = math.sqrt(sum((s - t) ** 2 for s, t in zip(y_score, y_true)) / len(y_true))

    try:
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else None
    except Exception:
        # ── Implementaciones puras (fallback sin sklearn) ─────────────────────
        n = len(y_true)
        acc = sum(1 for p, t in zip(y_pred, y_true) if p == t) / n
        tp = sum(1 for p, t in zip(y_pred, y_true) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(y_pred, y_true) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(y_pred, y_true) if p == 0 and t == 1)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        auc = _auc_puro(y_true, y_score)

    return {"AUC": auc, "ACC": acc, "F1": f1, "RMSE": rmse}


def _media_std(valores: list[float]) -> tuple[float, float]:
    """Media y desviación estándar poblacional, ignorando None."""
    xs = [v for v in valores if v is not None]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, math.sqrt(var)


# ──────────────────────────────────────────────────────────────────────────────
# Pares (entrada, objetivo) por secuencia con máscara
# ──────────────────────────────────────────────────────────────────────────────
def pares_validos(seq: dict) -> list[tuple[int, int, int]]:
    """Lista de (concepto_query, respuesta_pasada, respuesta_objetivo) por paso.

    Para cada t en 0..L-2: el modelo conoce (concepts[t], responses[t]) y debe
    predecir responses[t+1] del concepto concepts[t+1]. Solo posiciones válidas.
    """
    c, r = seq["concepts"], seq["responses"]
    return [(c[t + 1], r[t], r[t + 1]) for t in range(len(c) - 1)]


# ──────────────────────────────────────────────────────────────────────────────
# Baselines (no requieren pyKT ni torch)
# ──────────────────────────────────────────────────────────────────────────────
def predecir_baseline_global(train: list[dict], test: list[dict]) -> tuple[list[int], list[float]]:
    """Predice para todas las posiciones el promedio global de aciertos de train."""
    todos = [r for s in train for r in s["responses"]]
    p = sum(todos) / len(todos) if todos else 0.5
    y_true, y_score = [], []
    for s in test:
        for _, _, tgt in pares_validos(s):
            y_true.append(tgt)
            y_score.append(p)
    return y_true, y_score


def predecir_baseline_concept(
    train: list[dict], test: list[dict], n_skills: int
) -> tuple[list[int], list[float]]:
    """Predice el promedio de aciertos por concepto (dificultad media de cada sección).

    Para cada concepto se usa su tasa de acierto en train; conceptos no vistos en
    train caen al promedio global (suavizado natural ante datos escasos).
    """
    suma = [0.0] * n_skills
    cuenta = [0] * n_skills
    todos = []
    for s in train:
        for c, r in zip(s["concepts"], s["responses"]):
            if 0 <= c < n_skills:
                suma[c] += r
                cuenta[c] += 1
            todos.append(r)
    global_p = sum(todos) / len(todos) if todos else 0.5
    prob_concepto = [
        (suma[i] / cuenta[i]) if cuenta[i] else global_p for i in range(n_skills)
    ]

    y_true, y_score = [], []
    for s in test:
        for cq, _, tgt in pares_validos(s):
            p = prob_concepto[cq] if 0 <= cq < n_skills else global_p
            y_true.append(tgt)
            y_score.append(p)
    return y_true, y_score


# ──────────────────────────────────────────────────────────────────────────────
# Construcción de tensores paddeados para los modelos neuronales (pyKT)
# ──────────────────────────────────────────────────────────────────────────────
def _tensorizar(seqs: list[dict], seq_len: int):
    """Convierte secuencias a tensores (q, r, qry, tgt, mask) paddeados a seq_len.

    Formato KT: q/r = pasado (t), qry/tgt = siguiente (t+1). El padding usa 0 en
    los índices/respuestas (válido como embedding) y mask=0 para esas posiciones,
    de modo que NUNCA entran al cálculo de pérdida ni de métricas.
    """
    import torch

    qs, rs, qrys, tgts, masks = [], [], [], [], []
    for s in seqs:
        c, r = s["concepts"], s["responses"]
        # Cantidad de pares (pasado t → siguiente t+1), recortada a seq_len.
        L = min(len(c) - 1, seq_len)
        if L <= 0:
            continue
        # q/r = paso t (pasado); qry/tgt = paso t+1 (a predecir).
        q = [c[t] for t in range(L)]
        rp = [r[t] for t in range(L)]
        qry = [c[t + 1] for t in range(L)]
        tgt = [r[t + 1] for t in range(L)]
        pad = seq_len - L
        qs.append(q + [0] * pad)
        rs.append(rp + [0] * pad)
        qrys.append(qry + [0] * pad)
        tgts.append([float(x) for x in tgt] + [0.0] * pad)
        masks.append([1] * L + [0] * pad)

    return (
        torch.LongTensor(qs),
        torch.LongTensor(rs),
        torch.LongTensor(qrys),
        torch.FloatTensor(tgts),
        torch.FloatTensor(masks),
    )


def _preparar_pykt():
    """Monkeypatch de turtle (pyKT lo importa) y fija el device de pyKT a CPU.

    pyKT importa `turtle` (requiere Tk) en algunos modelos y usa una variable de
    módulo `device` en pykt.models.utils para ut_mask/pos_encode. Forzamos CPU.
    """
    import sys
    import types

    if "turtle" not in sys.modules:
        m = types.ModuleType("turtle")
        m.forward = lambda *a, **k: None
        sys.modules["turtle"] = m

    import pykt.models.utils as _u

    _u.device = "cpu"


def _entrenar_y_predecir_neuronal(
    nombre: str,
    train: list[dict],
    test: list[dict],
    n_skills: int,
    cfg: dict,
) -> tuple[list[int], list[float]]:
    """Entrena un modelo neuronal de pyKT (sakt|dkt) en CPU y predice sobre test.

    Devuelve (y_true, y_score) solo de las posiciones válidas (según la máscara).
    Reproducible: fija la semilla de torch a cfg['seed'] antes de instanciar.
    """
    import torch
    from torch import nn

    _preparar_pykt()
    torch.manual_seed(cfg["seed"])

    seq_len = cfg["seq_len"]
    q_tr, r_tr, qry_tr, tgt_tr, m_tr = _tensorizar(train, seq_len)
    q_te, r_te, qry_te, tgt_te, m_te = _tensorizar(test, seq_len)

    if nombre == "sakt":
        from pykt.models.sakt import SAKT

        model = SAKT(
            num_c=n_skills,
            seq_len=seq_len,
            emb_size=cfg["emb_size"],
            num_attn_heads=cfg["heads"],
            dropout=cfg["dropout"],
            num_en=cfg["layers"],
            emb_type="qid",
        )

        def _forward(q, r, qry):
            # SAKT devuelve p de forma (batch, seq) ya con sigmoid.
            return model(q, r, qry)

    elif nombre == "dkt":
        from pykt.models.dkt import DKT

        model = DKT(
            num_c=n_skills,
            emb_size=cfg["emb_size"],
            dropout=cfg["dropout"],
            emb_type="qid",
        )

        def _forward(q, r, qry):
            # DKT devuelve y de forma (batch, seq, num_c) con sigmoid; se toma la
            # probabilidad del concepto consultado (qry) en cada paso (gather).
            y = model(q, r)  # (B, S, C)
            idx = qry.unsqueeze(-1).clamp(min=0, max=n_skills - 1)
            return y.gather(-1, idx).squeeze(-1)  # (B, S)

    else:
        raise ValueError(f"Modelo neuronal desconocido: {nombre}")

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    bce = nn.BCELoss(reduction="none")

    n = q_tr.shape[0]
    batch = max(1, cfg["batch"])
    for _ in range(cfg["epochs"]):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            b = perm[i : i + batch]
            opt.zero_grad()
            out = _forward(q_tr[b], r_tr[b], qry_tr[b]).view(tgt_tr[b].shape)
            perdida = bce(out, tgt_tr[b]) * m_tr[b]
            perdida = perdida.sum() / m_tr[b].sum().clamp(min=1)
            perdida.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        if q_te.shape[0] > 0:
            out = _forward(q_te, r_te, qry_te).view(tgt_te.shape)
            sel = m_te == 1
            y_score = out[sel].clamp(0.0, 1.0).tolist()
            y_true = [int(x) for x in tgt_te[sel].tolist()]
    return y_true, y_score


# ──────────────────────────────────────────────────────────────────────────────
# Evaluación de un modelo con CV K-fold
# ──────────────────────────────────────────────────────────────────────────────
def evaluar_modelo(nombre: str, seqs: list[dict], n_skills: int, cfg: dict) -> dict:
    """Corre la CV K-fold para `nombre` y devuelve métricas agregadas (media/std)."""
    folds = kfold_indices(len(seqs), cfg["kfolds"], cfg["seed"])
    por_fold: list[dict] = []

    for fi in range(cfg["kfolds"]):
        test_idx = set(folds[fi])
        test = [seqs[i] for i in test_idx]
        train = [seqs[i] for i in range(len(seqs)) if i not in test_idx]
        if not train or not test:
            continue

        if nombre == "baseline_global":
            y_true, y_score = predecir_baseline_global(train, test)
        elif nombre == "baseline_concept":
            y_true, y_score = predecir_baseline_concept(train, test, n_skills)
        elif nombre in ("sakt", "dkt"):
            y_true, y_score = _entrenar_y_predecir_neuronal(nombre, train, test, n_skills, cfg)
        else:
            raise ValueError(f"Modelo desconocido: {nombre}")

        if not y_true:
            continue
        met = calcular_metricas(y_true, y_score)
        met["n"] = len(y_true)
        por_fold.append(met)
        auc_str = f"{met['AUC']:.4f}" if met["AUC"] is not None else "n/a"
        print(
            f"  [{nombre}] fold {fi + 1}/{cfg['kfolds']} | "
            f"n={met['n']:4d} AUC={auc_str} ACC={met['ACC']:.4f} "
            f"F1={met['F1']:.4f} RMSE={met['RMSE']:.4f}"
        )

    resumen = {"modelo": nombre, "folds": len(por_fold)}
    for k in ("AUC", "ACC", "F1", "RMSE"):
        m, s = _media_std([f[k] for f in por_fold])
        resumen[f"{k}_mean"] = m
        resumen[f"{k}_std"] = s
    return resumen


# ──────────────────────────────────────────────────────────────────────────────
# Salidas (consola, CSV, markdown)
# ──────────────────────────────────────────────────────────────────────────────
NOMBRE_LARGO = {
    "sakt": "SAKT (pyKT)",
    "dkt": "DKT (pyKT)",
    "baseline_global": "Baseline global",
    "baseline_concept": "Baseline por-concepto",
}


def _fmt(m: float, s: float) -> str:
    if m != m:  # NaN
        return "n/a"
    return f"{m:.4f} ± {s:.4f}"


def imprimir_tabla(resultados: list[dict], cfg: dict) -> None:
    print("\n=== Comparación de modelos KT (CV {}-fold, seed={}) ===".format(
        cfg["kfolds"], cfg["seed"]))
    cols = ["Modelo", "AUC", "ACC", "F1", "RMSE", "Folds"]
    anchos = [24, 17, 17, 17, 17, 5]
    fila = "".join(c.ljust(w) for c, w in zip(cols, anchos))
    print(fila)
    print("-" * sum(anchos))
    for r in resultados:
        vals = [
            NOMBRE_LARGO.get(r["modelo"], r["modelo"]),
            _fmt(r["AUC_mean"], r["AUC_std"]),
            _fmt(r["ACC_mean"], r["ACC_std"]),
            _fmt(r["F1_mean"], r["F1_std"]),
            _fmt(r["RMSE_mean"], r["RMSE_std"]),
            str(r["folds"]),
        ]
        print("".join(v.ljust(w) for v, w in zip(vals, anchos)))


def escribir_csv(resultados: list[dict], path: Path) -> None:
    campos = [
        "modelo", "folds",
        "AUC_mean", "AUC_std", "ACC_mean", "ACC_std",
        "F1_mean", "F1_std", "RMSE_mean", "RMSE_std",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for r in resultados:
            w.writerow({k: r.get(k, "") for k in campos})


def escribir_markdown(resultados: list[dict], cfg: dict, meta: dict, path: Path) -> None:
    lineas = [
        "# Comparación de modelos de Knowledge Tracing — SWARD",
        "",
        f"- Dataset: `{cfg['dataset']}`",
        f"- Secuencias: {meta['n_seqs']} | Conceptos: {meta['n_skills']} | "
        f"Interacciones evaluables: {meta['n_pares']}",
        f"- Validación cruzada: {cfg['kfolds']}-fold | Semilla: {cfg['seed']} | "
        f"Épocas (neuronales): {cfg['epochs']}",
        f"- Hiperparámetros neuronales: emb_size={cfg['emb_size']}, heads={cfg['heads']}, "
        f"layers={cfg['layers']}, dropout={cfg['dropout']}, lr={cfg['lr']}, batch={cfg['batch']}",
        "",
        "Métricas: media ± desviación estándar sobre los folds. Calculadas solo "
        "sobre posiciones válidas (padding enmascarado). AUC = n/a si un fold no "
        "contiene ambas clases.",
        "",
        "| Modelo | AUC | ACC | F1 | RMSE | Folds |",
        "|---|---|---|---|---|---|",
    ]
    for r in resultados:
        lineas.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                NOMBRE_LARGO.get(r["modelo"], r["modelo"]),
                _fmt(r["AUC_mean"], r["AUC_std"]),
                _fmt(r["ACC_mean"], r["ACC_std"]),
                _fmt(r["F1_mean"], r["F1_std"]),
                _fmt(r["RMSE_mean"], r["RMSE_std"]),
                r["folds"],
            )
        )
    lineas.append("")
    lineas.append(
        "_AUC: área bajo la curva ROC (capacidad de ordenar aciertos vs. fallos). "
        "ACC: exactitud con umbral 0.5. F1: media armónica de precisión y recall "
        "de la clase 'acierto'. RMSE: error cuadrático medio de la probabilidad._"
    )
    Path(path).write_text("\n".join(lineas) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> None:
    cfg = parse_config(argv)
    print("=== Evaluación reproducible de modelos KT (SWARD) ===")
    print("Configuración:", json.dumps(cfg, ensure_ascii=False))

    seqs, _concept_index = cargar_dataset(cfg["dataset"])
    if len(seqs) < cfg["kfolds"]:
        raise SystemExit(
            f"Pocas secuencias ({len(seqs)}) para {cfg['kfolds']} folds. "
            "Reduce --kfolds o usa un dataset mayor."
        )
    n_skills = n_skills_de(seqs)
    n_pares = sum(len(pares_validos(s)) for s in seqs)
    print(
        f"Secuencias: {len(seqs)} | Conceptos: {n_skills} | "
        f"Interacciones evaluables (pares t→t+1): {n_pares}\n"
    )

    resultados: list[dict] = []
    for nombre in cfg["models"]:
        print(f"→ Evaluando '{nombre}'...")
        resultados.append(evaluar_modelo(nombre, seqs, n_skills, cfg))

    imprimir_tabla(resultados, cfg)

    outdir = Path(cfg["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "model_comparison.csv"
    md_path = outdir / "model_comparison.md"
    meta = {"n_seqs": len(seqs), "n_skills": n_skills, "n_pares": n_pares}
    escribir_csv(resultados, csv_path)
    escribir_markdown(resultados, cfg, meta, md_path)
    print(f"\nResultados escritos en:\n  - {csv_path}\n  - {md_path}")


if __name__ == "__main__":
    main()
