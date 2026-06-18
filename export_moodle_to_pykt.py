"""Exporta interacciones reales de Moodle al formato de entrenamiento KT.

Lee, vía la API REST de Moodle, las notas por estudiante y actividad de cada
curso, deriva el concepto = sección del curso, y construye secuencias
(concepto, acierto) por estudiante ordenadas en el tiempo (orden de secciones).

Salida: outputs/moodle_kt_dataset.json con:
  - concept_index: {nombre_concepto: int}
  - sequences: [{student, course, concepts: [int...], responses: [0/1...]}]

Uso:
  MOODLE_URL=... MOODLE_TOKEN=... python export_moodle_to_pykt.py
"""

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

MOODLE_URL = os.environ["MOODLE_URL"].rstrip("/")
MOODLE_TOKEN = os.environ["MOODLE_TOKEN"]
COURSE_IDS = [int(c) for c in os.environ.get("COURSE_IDS", "2,3,4,5").split(",")]
ENDPOINT = f"{MOODLE_URL}/webservice/rest/server.php"
OUT = Path(__file__).parent / "outputs" / "moodle_kt_dataset.json"
APROBADO = 0.5  # graderaw/grademax >= 0.5 => acierto (igual que get_events)


def _call(fn: str, **params) -> object:
    params.update(wstoken=MOODLE_TOKEN, moodlewsrestformat="json", wsfunction=fn)
    url = f"{ENDPOINT}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def _seccion_por_instancia(course_id: int) -> tuple[dict[tuple[str, int], str], list[str]]:
    """Mapa (modname, instance)->seccion y la lista ordenada de secciones."""
    data = _call("core_course_get_contents", courseid=course_id)
    mapa: dict[tuple[str, int], str] = {}
    orden: list[str] = []
    for s in data if isinstance(data, list) else []:
        nombre = (s.get("name") or "General").strip()
        orden.append(nombre)
        for m in s.get("modules", []):
            if m.get("modname") and m.get("instance") is not None:
                mapa[(m["modname"], int(m["instance"]))] = nombre
    return mapa, orden


SEQ_LEN = 200
PAD = -1
DATA_MOODLE = Path(__file__).parent / "data" / "moodle"


def _fila_csv(concepts: list[int], responses: list[int], fold: int) -> dict:
    """Una secuencia en el formato CSV de pyKT (paddeada a SEQ_LEN)."""
    c = (concepts + [PAD] * SEQ_LEN)[:SEQ_LEN]
    r = (responses + [PAD] * SEQ_LEN)[:SEQ_LEN]
    s = ([1] * len(concepts) + [PAD] * SEQ_LEN)[:SEQ_LEN]
    return {
        "fold": fold,
        "concepts": ",".join(map(str, c)),
        "responses": ",".join(map(str, r)),
        "selectmasks": ",".join(map(str, s)),
    }


def _escribir_csv_pykt(sequences: list[dict], concept_index: dict[str, int]) -> None:
    """Escribe los CSV (train_valid/test) + concept_index.json que consume train.py.

    Split ~80/20 determinístico (sin random para reproducibilidad). En
    train_valid el fold rota 0..4 (train.py usa fold 4 como validación);
    el set de test va con fold -1.
    """
    import csv

    DATA_MOODLE.mkdir(parents=True, exist_ok=True)
    n_test = max(1, len(sequences) // 5)
    test, train_valid = sequences[:n_test], sequences[n_test:]

    def _dump(path: Path, seqs: list[dict], test_set: bool) -> None:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["fold", "concepts", "responses", "selectmasks"])
            w.writeheader()
            for i, s in enumerate(seqs):
                fold = -1 if test_set else (i % 5)
                w.writerow(_fila_csv(s["concepts"], s["responses"], fold))

    _dump(DATA_MOODLE / "train_valid_sequences.csv", train_valid, False)
    _dump(DATA_MOODLE / "test_sequences.csv", test, True)
    (DATA_MOODLE / "concept_index.json").write_text(
        json.dumps(concept_index, ensure_ascii=False, indent=2)
    )
    print(f"CSV pyKT + concept_index escritos en {DATA_MOODLE}")


def main() -> None:
    concept_index: dict[str, int] = {}
    sequences: list[dict] = []

    for cid in COURSE_IDS:
        secciones, orden = _seccion_por_instancia(cid)
        rank = {nombre: i for i, nombre in enumerate(orden)}
        users = _call("core_enrol_get_enrolled_users", courseid=cid)
        estudiantes = [
            u for u in users
            if any(r.get("shortname") == "student" for r in u.get("roles", []))
        ]
        for u in estudiantes:
            d = _call("gradereport_user_get_grade_items", courseid=cid, userid=u["id"])
            inter = []  # (rank_seccion, concepto, acierto)
            for ut in d.get("usergrades", []):
                for it in ut.get("gradeitems", []):
                    mod = it.get("itemmodule")
                    gr, gm = it.get("graderaw"), it.get("grademax")
                    if mod not in ("assign", "quiz", "lesson") or gr is None or not gm:
                        continue
                    concepto = secciones.get((mod, int(it.get("iteminstance") or 0)))
                    if not concepto:
                        continue
                    acierto = 1 if (gr / gm) >= APROBADO else 0
                    inter.append((rank.get(concepto, 999), concepto, acierto))
            inter.sort(key=lambda x: x[0])  # orden temporal por sección
            if len(inter) < 2:
                continue
            for _, concepto, _ in inter:
                if concepto not in concept_index:
                    concept_index[concepto] = len(concept_index)
            sequences.append({
                "student": u["id"],
                "course": cid,
                "concepts": [concept_index[c] for _, c, _ in inter],
                "responses": [a for _, _, a in inter],
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"concept_index": concept_index, "sequences": sequences}, indent=2,
        ensure_ascii=False,
    ))
    _escribir_csv_pykt(sequences, concept_index)
    total = sum(len(s["responses"]) for s in sequences)
    correctas = sum(sum(s["responses"]) for s in sequences)
    print(f"Secuencias: {len(sequences)} | interacciones: {total} | "
          f"conceptos: {len(concept_index)} | %correctas: {100 * correctas / total:.0f}%")
    print(f"Guardado en {OUT}")


if __name__ == "__main__":
    main()
