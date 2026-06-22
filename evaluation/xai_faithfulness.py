"""Evaluación de FIDELIDAD (faithfulness) de las explicaciones por atención del SAKT.

Contexto de tesis (SWARD — "IA Explicable" sobre knowledge tracing):
    El microservicio de recomendación muestra al docente/estudiante los *pesos de
    atención* del modelo SAKT como "explicación" de su predicción de dominio. Pero
    enseñar pesos NO demuestra que esos pesos expliquen realmente la decisión del
    modelo. Este script aporta la evidencia cuantitativa que falta: mide si la
    atención es *fiel* (faithful) a la predicción mediante las métricas estándar de
    la literatura de XAI (DeYoung et al., 2020, "ERASER"):

      - Comprehensiveness (exhaustividad): al BORRAR las interacciones de mayor
        atención, ¿cuánto cae la confianza de la predicción? Si la atención es fiel,
        la caída debe ser grande.
      - Sufficiency (suficiencia): al CONSERVAR SOLO las interacciones de mayor
        atención (borrando el resto), ¿qué tan poco cambia la predicción? Si la
        atención es fiel, debe cambiar poco (lo retenido basta para explicar).
      - Comparación vs. ALEATORIO: se repite borrando k interacciones al azar. La
        atención es fiel sólo si su comprehensiveness supera claramente al azar.

Reusa el patrón de carga del checkpoint y el monkeypatch `_last_attn` de
`sward-ms-recomendacion/src/domain/services/modelo_sakt.py` (copiado/adaptado a
propósito: este script es autónomo y NO importa el microservicio).

Uso:
    python evaluation/xai_faithfulness.py \
        --checkpoint outputs/sakt_moodle.pth \
        --dataset outputs/moodle_kt_dataset.json \
        --muestras 80 --topk 1 2 --out outputs/xai_faithfulness.md

Restricción de diseño: torch/pykt se importan de forma *perezosa* (lazy) dentro de
las funciones, de modo que el archivo pasa `python -m py_compile` sin pyKT/torch
instalados. La inferencia real sí requiere ambos.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import types as _types
from dataclasses import dataclass, field

# Token de "borrado" para perturbar una interacción. 0 es el valor de padding que
# el modelo ya ve durante el entrenamiento (ver train_sakt.py), por lo que es el
# sustituto neutro natural: equivale a "esta interacción no ocurrió".
PAD_TOKEN = 0


# --------------------------------------------------------------------------- #
# Carga del modelo (adaptado de modelo_sakt.py::ModeloSAKT._cargar_modelo)
# --------------------------------------------------------------------------- #
def _mock_turtle() -> None:
    """pyKT bug: qdkt.py importa `turtle` (requiere Tk). Mock antes de cargar pyKT."""
    if "turtle" not in sys.modules:
        _mock = _types.ModuleType("turtle")
        _mock.forward = lambda *a, **kw: None  # type: ignore[attr-defined]
        sys.modules["turtle"] = _mock


@dataclass
class ModeloCargado:
    """SAKT entrenado listo para inferencia + captura de atención."""

    model: object
    seq_len: int
    n_skills: int
    concept_index: dict


def cargar_sakt(checkpoint_path: str) -> ModeloCargado:
    """Carga el checkpoint pyKT-SAKT y parchea el bloque para capturar `_last_attn`.

    Idéntico en espíritu a ModeloSAKT._cargar_modelo, pero local: lee desde disco
    (no S3) y no hace fallback a mock (queremos fallar ruidoso si algo va mal).
    """
    import torch  # lazy

    # El checkpoint contiene tensores + primitivas → weights_only=True es seguro.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    n_skills = checkpoint["n_skills"]
    seq_len = checkpoint["seq_len"]
    emb_size = checkpoint["emb_size"]
    n_heads = checkpoint["n_heads"]
    dropout = checkpoint["dropout"]
    n_layers = checkpoint["n_layers"]
    concept_index = checkpoint.get("concept_index", {}) or {}

    _mock_turtle()

    import pykt.models.utils as _pykt_utils
    from pykt.models.sakt import SAKT, Blocks
    from pykt.models.utils import ut_mask

    _pykt_utils.device = "cpu"

    # Monkeypatch del bloque de atención para CAPTURAR los pesos (el forward de
    # pyKT los descarta). Idéntico al stock salvo que guarda _last_attn.
    def _blocks_forward_capture(self, q=None, k=None, v=None):
        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        causal_mask = ut_mask(seq_len=k.shape[0])
        attn_emb, attn_w = self.attn(
            q, k, v, attn_mask=causal_mask, need_weights=True
        )
        self._last_attn = attn_w.detach()  # (batch, tgt_len, src_len)
        attn_emb = self.attn_dropout(attn_emb)
        attn_emb, q = attn_emb.permute(1, 0, 2), q.permute(1, 0, 2)
        attn_emb = self.attn_layer_norm(q + attn_emb)
        emb = self.FFN(attn_emb)
        emb = self.FFN_dropout(emb)
        emb = self.FFN_layer_norm(attn_emb + emb)
        return emb

    Blocks.forward = _blocks_forward_capture

    model = SAKT(
        num_c=n_skills,
        seq_len=seq_len,
        emb_size=emb_size,
        num_attn_heads=n_heads,
        dropout=dropout,
        num_en=n_layers,
        emb_type="qid",
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return ModeloCargado(
        model=model,
        seq_len=seq_len,
        n_skills=n_skills,
        concept_index=concept_index,
    )


# --------------------------------------------------------------------------- #
# Inferencia + extracción de atención (adaptado de _real_prediccion)
# --------------------------------------------------------------------------- #
def _predecir(modelo: ModeloCargado, concepts: list, responses: list):
    """Devuelve (prob_ultimo_paso, pesos_atencion_sobre_pasado).

    Replica EXACTAMENTE el formato de inferencia de modelo_sakt._real_prediccion:
        q   = concepts[:-1]   (interacciones pasadas)
        r   = responses[:-1]
        qry = concepts[1:]    (consulta desplazada) → out[-1] = P(correcto último)
    `pesos` tiene longitud L-1 y suma 1 (atención normalizada sobre el pasado).
    """
    import torch  # lazy

    seq_len = modelo.seq_len
    concepts = list(concepts[-seq_len:])
    responses = list(responses[-seq_len:])
    L = len(concepts)
    if L < 2:
        raise ValueError("Se requieren >= 2 interacciones para evaluar faithfulness.")

    q = concepts[:-1]
    r = responses[:-1]
    qry = concepts[1:]

    pad = seq_len - (L - 1)
    q_t = torch.LongTensor([[PAD_TOKEN] * pad + q])
    r_t = torch.LongTensor([[PAD_TOKEN] * pad + r])
    qry_t = torch.LongTensor([[PAD_TOKEN] * pad + qry])

    with torch.no_grad():
        out = modelo.model(q_t, r_t, qry_t)  # SAKT aplica sigmoid internamente
        prob = float(out[0, -1].item())

    # Pesos de atención REALES del último paso sobre las L-1 interacciones pasadas.
    pesos = [1.0 / (L - 1)] * (L - 1)
    try:
        attn = modelo.model.blocks[-1]._last_attn  # (1, seq_len, seq_len)
        fila = attn[0, -1, -(L - 1):].tolist()
        total = sum(fila)
        if total > 0:
            pesos = [w / total for w in fila]
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] no se pudo extraer atención real, uso uniforme: {e}")

    return prob, pesos


def _predecir_perturbado(
    modelo: ModeloCargado,
    concepts: list,
    responses: list,
    indices_a_borrar: set,
):
    """Predice tras BORRAR (poner a PAD) las interacciones pasadas indicadas.

    `indices_a_borrar` indexa el VECTOR DE PASADO (0..L-2), alineado con `pesos`.
    Borrar = neutralizar la interacción: concepto y respuesta pasan a PAD_TOKEN, lo
    que el modelo interpreta como "posición vacía/padding".
    """
    import torch  # lazy

    seq_len = modelo.seq_len
    concepts = list(concepts[-seq_len:])
    responses = list(responses[-seq_len:])
    L = len(concepts)

    q = list(concepts[:-1])
    r = list(responses[:-1])
    qry = concepts[1:]  # la consulta NO se altera (es lo que predecimos)

    for i in indices_a_borrar:
        if 0 <= i < len(q):
            q[i] = PAD_TOKEN
            r[i] = PAD_TOKEN

    pad = seq_len - (L - 1)
    q_t = torch.LongTensor([[PAD_TOKEN] * pad + q])
    r_t = torch.LongTensor([[PAD_TOKEN] * pad + r])
    qry_t = torch.LongTensor([[PAD_TOKEN] * pad + qry])

    with torch.no_grad():
        out = modelo.model(q_t, r_t, qry_t)
        return float(out[0, -1].item())


def _predecir_conservando(
    modelo: ModeloCargado,
    concepts: list,
    responses: list,
    indices_a_conservar: set,
):
    """Predice CONSERVANDO sólo las interacciones indicadas (borra el resto).

    Es el complemento de _predecir_perturbado y sirve para `sufficiency`.
    """
    n_pasado = len(concepts[-modelo.seq_len:]) - 1
    a_borrar = {i for i in range(n_pasado) if i not in indices_a_conservar}
    return _predecir_perturbado(modelo, concepts, responses, a_borrar)


# --------------------------------------------------------------------------- #
# Métricas de faithfulness (DeYoung et al., 2020 — ERASER)
# --------------------------------------------------------------------------- #
def _confianza(prob: float) -> float:
    """Confianza de la predicción binaria = |p - 0.5| * 2 ∈ [0, 1].

    Comprehensiveness/sufficiency se definen sobre la *probabilidad de la clase
    predicha*. Para una salida binaria con sigmoide, medimos el cambio en la
    probabilidad de la clase ganadora; usar la confianza centrada en 0.5 evita
    que la métrica dependa de cuál clase (correcto/incorrecto) ganó.
    """
    return abs(prob - 0.5) * 2.0


def _topk_indices(pesos: list, k: int) -> set:
    """Índices de las k interacciones de MAYOR atención."""
    orden = sorted(range(len(pesos)), key=lambda i: pesos[i], reverse=True)
    return set(orden[:k])


def _aleatorio_indices(n: int, k: int, rng) -> set:
    return set(rng.sample(range(n), min(k, n)))


@dataclass
class ResultadoSecuencia:
    """Métricas por secuencia para un valor de k dado."""

    k: int
    n_pasado: int
    prob_base: float
    conf_base: float
    # Comprehensiveness: caída de confianza al borrar top-k de atención.
    comp_attn: float
    # Comprehensiveness al borrar k aleatorias (baseline).
    comp_rand: float
    # Sufficiency: cambio al conservar SÓLO el top-k de atención.
    suff_attn: float
    suff_rand: float


@dataclass
class Agregado:
    """Promedios sobre todas las secuencias evaluadas, por k."""

    k: int
    n: int = 0
    comp_attn: list = field(default_factory=list)
    comp_rand: list = field(default_factory=list)
    suff_attn: list = field(default_factory=list)
    suff_rand: list = field(default_factory=list)
    # Cuántas veces el top-k de atención fue MÁS influyente que el azar.
    gana_a_azar: int = 0


def evaluar_secuencia(
    modelo: ModeloCargado,
    concepts: list,
    responses: list,
    k: int,
    rng,
    n_rand_reps: int = 5,
) -> ResultadoSecuencia | None:
    """Calcula comprehensiveness y sufficiency (atención vs azar) para una secuencia."""
    n_pasado = len(concepts[-modelo.seq_len:]) - 1
    if n_pasado <= k:
        # Sin "resto" no hay contraste posible (borrar/conservar k = todo).
        return None

    prob_base, pesos = _predecir(modelo, concepts, responses)
    conf_base = _confianza(prob_base)

    # --- Comprehensiveness: borrar top-k de atención ---
    top = _topk_indices(pesos, k)
    prob_sin_top = _predecir_perturbado(modelo, concepts, responses, top)
    comp_attn = conf_base - _confianza(prob_sin_top)

    # --- Sufficiency: conservar SÓLO top-k de atención ---
    prob_solo_top = _predecir_conservando(modelo, concepts, responses, top)
    suff_attn = conf_base - _confianza(prob_solo_top)

    # --- Baseline aleatorio (promedio de varias repeticiones) ---
    comp_rand_vals, suff_rand_vals = [], []
    for _ in range(n_rand_reps):
        rand = _aleatorio_indices(n_pasado, k, rng)
        comp_rand_vals.append(
            conf_base - _confianza(_predecir_perturbado(modelo, concepts, responses, rand))
        )
        suff_rand_vals.append(
            conf_base - _confianza(_predecir_conservando(modelo, concepts, responses, rand))
        )
    comp_rand = statistics.mean(comp_rand_vals)
    suff_rand = statistics.mean(suff_rand_vals)

    return ResultadoSecuencia(
        k=k,
        n_pasado=n_pasado,
        prob_base=prob_base,
        conf_base=conf_base,
        comp_attn=comp_attn,
        comp_rand=comp_rand,
        suff_attn=suff_attn,
        suff_rand=suff_rand,
    )


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def cargar_secuencias(dataset_path: str) -> list:
    """Carga las secuencias del dataset Moodle (formato {concepts, responses})."""
    data = json.load(open(dataset_path))
    seqs = data["sequences"] if isinstance(data, dict) else data
    out = []
    for s in seqs:
        concepts = s.get("concepts")
        responses = s.get("responses")
        if concepts and responses and len(concepts) == len(responses) and len(concepts) >= 3:
            out.append((concepts, responses))
    return out


# --------------------------------------------------------------------------- #
# Reporte
# --------------------------------------------------------------------------- #
def _media(xs: list) -> float:
    return statistics.mean(xs) if xs else 0.0


def _desv(xs: list) -> float:
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def construir_reporte(agregados: list, n_seqs: int, checkpoint: str, dataset: str) -> str:
    """Genera el Markdown de resultados (tabla + interpretación de tesis)."""
    lineas = []
    lineas.append("# Fidelidad de las explicaciones por atención del SAKT (XAI)\n")
    lineas.append(
        "Métricas de *faithfulness* (DeYoung et al., 2020, ERASER) que miden si los "
        "pesos de atención que SWARD muestra como explicación realmente determinan la "
        "predicción de dominio del modelo SAKT.\n"
    )
    lineas.append(f"- Checkpoint evaluado: `{checkpoint}`")
    lineas.append(f"- Dataset: `{dataset}`")
    lineas.append(f"- Secuencias válidas evaluadas: **{n_seqs}**\n")

    lineas.append("## Resultados agregados\n")
    lineas.append(
        "| k | n | Comprehensiveness (atención) | Comprehensiveness (azar) | Δ comp. (atención − azar) | Sufficiency (atención) | Sufficiency (azar) | % gana al azar |"
    )
    lineas.append("|---|---|---|---|---|---|---|---|")
    for a in agregados:
        ca, cr = _media(a.comp_attn), _media(a.comp_rand)
        sa, sr = _media(a.suff_attn), _media(a.suff_rand)
        delta = ca - cr
        pct = (a.gana_a_azar / a.n * 100) if a.n else 0.0
        lineas.append(
            f"| {a.k} | {a.n} | {ca:+.4f} ± {_desv(a.comp_attn):.3f} | "
            f"{cr:+.4f} | **{delta:+.4f}** | {sa:+.4f} | {sr:+.4f} | {pct:.1f}% |"
        )
    lineas.append("")

    lineas.append("## Cómo leer cada métrica\n")
    lineas.append(
        "- **Comprehensiveness (exhaustividad)**: confianza_base − confianza tras "
        "BORRAR las top-k interacciones por atención. Valor **alto y positivo** = al "
        "quitar lo que el modelo dijo que importaba, la predicción se desmorona ⇒ "
        "esos pesos sí concentran la evidencia. Es la métrica central de fidelidad.\n"
    )
    lineas.append(
        "- **Comprehensiveness (azar)**: lo mismo borrando k interacciones al azar "
        "(promedio de varias repeticiones). Es el grupo de control.\n"
    )
    lineas.append(
        "- **Δ comprehensiveness = atención − azar**: el resultado clave. Si es "
        "claramente **> 0**, las explicaciones por atención son **fieles** (borrar lo "
        "que destacan impacta más que borrar interacciones cualesquiera). Si ≈ 0, la "
        "atención no es mejor explicación que elegir al azar.\n"
    )
    lineas.append(
        "- **Sufficiency (suficiencia)**: confianza_base − confianza CONSERVANDO sólo "
        "el top-k de atención. Valor **cercano a 0** = lo retenido basta para "
        "reproducir la predicción ⇒ la explicación es suficiente. Cuanto menor en "
        "magnitud, mejor.\n"
    )
    lineas.append(
        "- **% gana al azar**: proporción de secuencias donde el top-k de atención "
        "fue más influyente (mayor comprehensiveness) que el promedio aleatorio. "
        ">50 % indica ventaja sistemática de la atención.\n"
    )

    # Veredicto automático basado en el menor k disponible.
    if agregados:
        a0 = agregados[0]
        delta0 = _media(a0.comp_attn) - _media(a0.comp_rand)
        pct0 = (a0.gana_a_azar / a0.n * 100) if a0.n else 0.0
        lineas.append("## Interpretación para la tesis\n")
        if delta0 > 0.02 and pct0 >= 55:
            veredicto = (
                f"Para k={a0.k}, borrar las interacciones de mayor atención reduce la "
                f"confianza **{delta0:+.4f}** más que borrar interacciones al azar, y la "
                f"atención supera al azar en el **{pct0:.0f}%** de las secuencias. La "
                "evidencia respalda que las explicaciones por atención del SAKT son "
                "**razonablemente fieles**: los pesos que SWARD muestra no son un "
                "adorno, identifican interacciones que efectivamente sostienen la "
                "predicción de dominio. Esto da sustento empírico al componente de IA "
                "Explicable del sistema."
            )
        elif delta0 > 0:
            veredicto = (
                f"Para k={a0.k}, la atención supera al azar de forma **modesta** "
                f"(Δ comprehensiveness = {delta0:+.4f}; gana al azar en {pct0:.0f}% de "
                "los casos). Hay señal de fidelidad pero débil: se recomienda reportar "
                "estos números con cautela, complementar con un *estudio con usuarios* "
                "(ver `xai_user_study.md`) y, de ser posible, reentrenar con más datos "
                "reales de Moodle, ya que el dataset actual es pequeño."
            )
        else:
            veredicto = (
                f"Para k={a0.k}, la atención **no** supera de forma consistente al azar "
                f"(Δ comprehensiveness = {delta0:+.4f}). Esto NO invalida el sistema, "
                "pero sí advierte que la atención por sí sola podría no ser una "
                "explicación fiel para este checkpoint/dataset. Conviene: (1) ampliar el "
                "dataset, (2) reportar el límite honestamente en la tesis, y (3) "
                "considerar métodos de explicación complementarios (p. ej. atribuciones "
                "por gradientes/oclusión) además de la atención."
            )
        lineas.append(veredicto + "\n")
        lineas.append(
            "> Nota metodológica: el dataset Moodle disponible tiene secuencias cortas "
            "(~6–7 interacciones), por lo que los valores de k útiles son pequeños y el "
            "tamaño de muestra es limitado. Las conclusiones deben acompañarse del "
            "tamaño de muestra y, en lo posible, de una prueba de significancia "
            "(p. ej. Wilcoxon pareado entre comprehensiveness de atención vs azar).\n"
        )
    return "\n".join(lineas)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        default=os.path.join(base, "outputs", "sakt_moodle.pth"),
        help="Ruta al checkpoint SAKT (.pth) entrenado.",
    )
    p.add_argument(
        "--dataset",
        default=os.path.join(base, "outputs", "moodle_kt_dataset.json"),
        help="Dataset JSON con {concept_index, sequences:[{concepts,responses}]}.",
    )
    p.add_argument(
        "--muestras",
        type=int,
        default=100,
        help="Máx. de secuencias a evaluar (submuestreo determinista).",
    )
    p.add_argument(
        "--topk",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Valores de k (nº de interacciones top-atención a borrar/conservar).",
    )
    p.add_argument(
        "--rand-reps",
        type=int,
        default=5,
        help="Repeticiones del baseline aleatorio por secuencia.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out",
        default=os.path.join(base, "outputs", "xai_faithfulness.md"),
        help="Ruta del reporte Markdown de salida.",
    )
    return p.parse_args(argv)


def main(argv=None):
    import random

    args = parse_args(argv)
    rng = random.Random(args.seed)

    print(f"[xai] Cargando SAKT desde {args.checkpoint} ...")
    modelo = cargar_sakt(args.checkpoint)
    print(
        f"[xai] Modelo OK | n_skills={modelo.n_skills} seq_len={modelo.seq_len} "
        f"conceptos={len(modelo.concept_index)}"
    )

    seqs = cargar_secuencias(args.dataset)
    print(f"[xai] Secuencias válidas en dataset: {len(seqs)}")
    if len(seqs) > args.muestras:
        seqs = rng.sample(seqs, args.muestras)
    print(f"[xai] Evaluando {len(seqs)} secuencias | topk={args.topk}")

    agregados = {k: Agregado(k=k) for k in args.topk}
    n_evaluadas = 0
    for concepts, responses in seqs:
        usada = False
        for k in args.topk:
            res = evaluar_secuencia(
                modelo, concepts, responses, k, rng, n_rand_reps=args.rand_reps
            )
            if res is None:
                continue
            usada = True
            a = agregados[k]
            a.n += 1
            a.comp_attn.append(res.comp_attn)
            a.comp_rand.append(res.comp_rand)
            a.suff_attn.append(res.suff_attn)
            a.suff_rand.append(res.suff_rand)
            if res.comp_attn > res.comp_rand:
                a.gana_a_azar += 1
        if usada:
            n_evaluadas += 1

    ordenados = [agregados[k] for k in sorted(agregados)]

    # --- Consola ---
    print("\n=== RESULTADOS DE FIDELIDAD (faithfulness) ===")
    print(f"Secuencias evaluadas: {n_evaluadas}")
    for a in ordenados:
        if a.n == 0:
            print(f"  k={a.k}: sin secuencias suficientemente largas (omitida).")
            continue
        ca, cr = _media(a.comp_attn), _media(a.comp_rand)
        sa, sr = _media(a.suff_attn), _media(a.suff_rand)
        pct = a.gana_a_azar / a.n * 100
        print(
            f"  k={a.k} (n={a.n}): "
            f"comprehensiveness atención={ca:+.4f} vs azar={cr:+.4f} "
            f"(Δ={ca - cr:+.4f}) | sufficiency atención={sa:+.4f} vs azar={sr:+.4f} "
            f"| gana al azar={pct:.1f}%"
        )

    # --- Markdown ---
    reporte = construir_reporte(ordenados, n_evaluadas, args.checkpoint, args.dataset)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(reporte)
    print(f"\n[xai] Reporte escrito en {args.out}")


if __name__ == "__main__":
    main()
