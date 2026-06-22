"""Descarga el training-data REAL de SWARD y lo convierte al formato que consume
el evaluador (`outputs/moodle_kt_dataset.json`: concept_index + sequences).

Así la tabla comparativa de la tesis refleja la data actual (Moodle ya poblado),
no un snapshot viejo. Requiere los mismos secrets que el entrenamiento:
    SWARD_API_URL              ej. https://<cloudfront>
    TRAZABILIDAD_SERVICE_KEY   service-key del endpoint interno

Uso: SWARD_API_URL=... TRAZABILIDAD_SERVICE_KEY=... python evaluation/fetch_dataset.py
"""

import json
import os
import urllib.request
from collections import defaultdict
from pathlib import Path

API = os.environ["SWARD_API_URL"].rstrip("/")
KEY = os.environ["TRAZABILIDAD_SERVICE_KEY"]
OUT = Path(os.environ.get("OUTDIR", "outputs")) / "moodle_kt_dataset.json"


def main() -> None:
    req = urllib.request.Request(
        f"{API}/api/v1/dashboard/training-data",
        headers={"X-Service-Key": KEY},
    )
    with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 (URL de confianza)
        rows = json.load(r)

    # Agrupa por estudiante y ordena temporalmente (igual que el pipeline de entreno).
    por_est: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for x in rows:
        por_est[str(x["estudiante_id"])].append(
            (str(x.get("orden", "")), str(x["concepto"]), 1 if x["correcta"] else 0)
        )

    conceptos = sorted({c for filas in por_est.values() for _, c, _ in filas})
    cidx = {c: i for i, c in enumerate(conceptos)}

    sequences = []
    for est, filas in por_est.items():
        filas.sort(key=lambda t: t[0])
        sequences.append(
            {
                "student": est,
                "course": "",
                "concepts": [cidx[c] for _, c, _ in filas],
                "responses": [resp for _, _, resp in filas],
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"concept_index": cidx, "sequences": sequences}, ensure_ascii=False)
    )
    print(
        f"Dataset fresco: {len(sequences)} secuencias, {len(cidx)} conceptos, "
        f"{len(rows)} interacciones → {OUT}"
    )


if __name__ == "__main__":
    main()
