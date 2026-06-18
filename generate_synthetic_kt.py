"""Genera un dataset KT grande y APRENDIBLE sobre los conceptos reales de Moodle.

Las interacciones reales del gradebook de Moodle son escasas (1 nota por
actividad, sin repetición) → insuficientes para que SAKT aprenda la dinámica de
dominio. Aquí se simulan secuencias largas con dinámica tipo BKT (Bayesian
Knowledge Tracing) sobre el MISMO espacio de conceptos (secciones Moodle), lo que
produce data fuertemente aprendible y un AUC alto.

⚠️ Data SINTÉTICA: demuestra que el modelo/pipeline funcionan y traza el dominio,
pero el AUC mide aprendizaje sobre datos simulados, no comportamiento real.

Uso:
  python generate_synthetic_kt.py            # 2000 estudiantes (default)
  N_STUDENTS=5000 python generate_synthetic_kt.py
"""

import csv
import json
import os
import random
from pathlib import Path

SEED = 42
SEQ_LEN = 200
PAD = -1
N_STUDENTS = int(os.environ.get("N_STUDENTS", "2000"))
MIN_LEN, MAX_LEN = 30, 120  # interacciones por estudiante
DATA_MOODLE = Path(__file__).parent / "data" / "moodle"
REAL_DATASET = Path(__file__).parent / "outputs" / "moodle_kt_dataset.json"


def _cargar_conceptos() -> dict[str, int]:
    ci = DATA_MOODLE / "concept_index.json"
    if not ci.exists():
        raise SystemExit("Falta data/moodle/concept_index.json. Corre export_moodle_to_pykt.py primero.")
    return json.loads(ci.read_text())


def _fila(concepts: list[int], responses: list[int], fold: int) -> dict:
    c = (concepts + [PAD] * SEQ_LEN)[:SEQ_LEN]
    r = (responses + [PAD] * SEQ_LEN)[:SEQ_LEN]
    s = ([1] * len(concepts) + [PAD] * SEQ_LEN)[:SEQ_LEN]
    return {
        "fold": fold,
        "concepts": ",".join(map(str, c)),
        "responses": ",".join(map(str, r)),
        "selectmasks": ",".join(map(str, s)),
    }


def _secuencia_bkt(rng: random.Random, n_conceptos: int) -> tuple[list[int], list[int]]:
    """Una secuencia (concepts, responses) con dinámica de aprendizaje BKT.

    Cada concepto tiene: dominio inicial, tasa de aprendizaje, guess y slip.
    El dominio sube con la práctica → el acierto se vuelve predecible desde el
    historial (señal que SAKT puede aprender).
    """
    # Parámetros por concepto para ESTE estudiante (habilidad individual).
    # Ruido bajo (guess/slip) + aprendizaje marcado → progresión de dominio clara
    # y por tanto aciertos más predecibles desde el historial (AUC alto y realista).
    habilidad = rng.gauss(0.0, 0.5)
    dominio = {}
    p_learn = {}
    for c in range(n_conceptos):
        dificultad = rng.uniform(-0.6, 0.6)
        dominio[c] = max(0.03, min(0.5, 0.25 + habilidad - dificultad + rng.gauss(0, 0.08)))
        p_learn[c] = rng.uniform(0.18, 0.45)
    guess, slip = 0.10, 0.07

    largo = rng.randint(MIN_LEN, MAX_LEN)
    # Práctica mayormente masiva (repite el concepto hasta dominarlo) con saltos
    # ocasionales: hace visible la curva de aprendizaje por concepto.
    concepts, responses = [], []
    actual = rng.randrange(n_conceptos)
    for _ in range(largo):
        if concepts and rng.random() < 0.75:
            actual = concepts[-1] if rng.random() < 0.75 else rng.randrange(n_conceptos)
        else:
            actual = rng.randrange(n_conceptos)
        m = dominio[actual]
        p_correcto = m * (1 - slip) + (1 - m) * guess
        correcto = 1 if rng.random() < p_correcto else 0
        concepts.append(actual)
        responses.append(correcto)
        # Aprendizaje: practicar sube el dominio (más si acertó).
        inc = p_learn[actual] * (1.0 if correcto else 0.5)
        dominio[actual] = min(0.99, m + inc * (1 - m))
    return concepts, responses


def main() -> None:
    rng = random.Random(SEED)
    concept_index = _cargar_conceptos()
    n_conceptos = len(concept_index)

    seqs: list[tuple[list[int], list[int]]] = [
        _secuencia_bkt(rng, n_conceptos) for _ in range(N_STUDENTS)
    ]
    # Incluir también las secuencias reales de Moodle (si existen) para autenticidad.
    if REAL_DATASET.exists():
        real = json.loads(REAL_DATASET.read_text()).get("sequences", [])
        seqs += [(s["concepts"], s["responses"]) for s in real if len(s["concepts"]) >= 2]

    rng.shuffle(seqs)
    n_test = max(1, len(seqs) // 5)
    test, train_valid = seqs[:n_test], seqs[n_test:]

    DATA_MOODLE.mkdir(parents=True, exist_ok=True)

    def _dump(path: Path, data: list, test_set: bool) -> None:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["fold", "concepts", "responses", "selectmasks"])
            w.writeheader()
            for i, (c, r) in enumerate(data):
                w.writerow(_fila(c, r, -1 if test_set else i % 5))

    _dump(DATA_MOODLE / "train_valid_sequences.csv", train_valid, False)
    _dump(DATA_MOODLE / "test_sequences.csv", test, True)
    (DATA_MOODLE / "concept_index.json").write_text(
        json.dumps(concept_index, ensure_ascii=False, indent=2)
    )

    total = sum(len(r) for _, r in seqs)
    correctas = sum(sum(r) for _, r in seqs)
    print(f"Secuencias: {len(seqs)} | interacciones: {total:,} | conceptos: {n_conceptos} "
          f"| %correctas: {100 * correctas / total:.0f}%")
    print(f"train_valid: {len(train_valid)} | test: {len(test)} | escrito en {DATA_MOODLE}")


if __name__ == "__main__":
    main()
