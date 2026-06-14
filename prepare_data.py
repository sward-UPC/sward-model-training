"""
Descarga y preprocesa ASSISTments 2015 al formato que espera pyKT (KTDataset).

Uso:
    python prepare_data.py

Genera: data/assist2015/train_valid_sequences.csv
        data/assist2015/test_sequences.csv
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/assist2015")
SEQ_LEN = 200
PAD = -1
N_FOLDS = 5
TEST_RATIO = 0.1
SEED = 42

# Google Drive file ID oficial de ASSISTments 2015 Skill Builder
# Fuente: https://sites.google.com/site/assistmentsdata/datasets/2015-assistments-skill-builder-data
GDRIVE_FILE_ID = "0B_hO8cnpcIMgUGZzRnh3bHJrSjQ"


def download_raw():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / "assist2015.csv"
    if raw_path.exists():
        print(f"Dataset crudo ya existe: {raw_path}")
        return raw_path

    print("Descargando ASSISTments 2015 desde Google Drive...")
    try:
        import gdown
        gdown.download(
            f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
            str(raw_path),
            quiet=False,
        )
        if raw_path.exists() and raw_path.stat().st_size > 1000:
            print(f"Descargado: {raw_path}")
            return raw_path
    except Exception as e:
        print(f"gdown falló: {e}")

    print("\nNo se pudo descargar automáticamente.")
    print("Descarga manualmente el archivo y colócalo en data/raw/assist2015.csv")
    print("Fuente: https://sites.google.com/site/assistmentsdata/datasets/2015-assistments-skill-builder-data")
    raise SystemExit(1)


def preprocess(raw_path: Path) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_out = OUT_DIR / "train_valid_sequences.csv"
    test_out = OUT_DIR / "test_sequences.csv"

    if train_out.exists() and test_out.exists():
        print("Datos ya preprocesados.")
        df = pd.read_csv(train_out)
        all_skills = []
        for row in df["concepts"]:
            all_skills.extend([int(x) for x in str(row).split(",") if x.strip() and x != str(PAD)])
        return max(all_skills) + 1

    print("Preprocesando ASSISTments 2015...")
    df = pd.read_csv(raw_path, encoding="latin1", low_memory=False)

    # Normalizar nombres de columnas
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Columnas del dataset 2015 Skill Builder: user_id, log_id, sequence_id, correct
    # sequence_id es el identificador de habilidad (skill builder)
    user_col = next(c for c in df.columns if "user" in c)
    skill_col = next(
        (c for c in df.columns if c in ("skill_id", "kc_id", "sequence_id", "problem_id")),
        None,
    )
    if skill_col is None:
        skill_col = next(c for c in df.columns if "skill" in c or "sequence" in c or "problem" in c)
    correct_col = next(c for c in df.columns if "correct" in c)
    print(f"Columnas usadas → usuario: {user_col}, habilidad: {skill_col}, correcto: {correct_col}")

    df = df[[user_col, skill_col, correct_col]].dropna()
    df[skill_col] = df[skill_col].astype(float).astype(int)
    df[correct_col] = df[correct_col].astype(int).clip(0, 1)
    df[user_col] = df[user_col].astype(int)

    # Remapear skill_ids a 0-indexed
    skills = sorted(df[skill_col].unique())
    skill2idx = {s: i for i, s in enumerate(skills)}
    df[skill_col] = df[skill_col].map(skill2idx)
    n_skills = len(skills)
    print(f"Skills únicos: {n_skills}, Estudiantes: {df[user_col].nunique()}, Interacciones: {len(df)}")

    # Construir secuencias por estudiante
    random.seed(SEED)
    np.random.seed(SEED)

    sequences = []
    for uid, group in tqdm(df.groupby(user_col), desc="Construyendo secuencias"):
        concepts = group[skill_col].tolist()
        responses = group[correct_col].tolist()
        # Partir en ventanas de SEQ_LEN
        for start in range(0, len(concepts), SEQ_LEN):
            c_chunk = concepts[start : start + SEQ_LEN]
            r_chunk = responses[start : start + SEQ_LEN]
            if len(c_chunk) < 2:
                continue
            # Padding
            pad_len = SEQ_LEN - len(c_chunk)
            c_chunk += [PAD] * pad_len
            r_chunk += [PAD] * pad_len
            # selectmasks: 1 para posiciones válidas, -1 para padding
            smasks = [1] * (SEQ_LEN - pad_len) + [PAD] * pad_len
            sequences.append((c_chunk, r_chunk, smasks))

    random.shuffle(sequences)
    n_test = max(1, int(len(sequences) * TEST_RATIO))
    test_seqs = sequences[:n_test]
    train_seqs = sequences[n_test:]

    def seqs_to_df(seqs, folds_list):
        rows = []
        for i, (concepts, responses, smasks) in enumerate(seqs):
            fold = folds_list[i % len(folds_list)]
            rows.append({
                "fold": fold,
                "concepts": ",".join(map(str, concepts)),
                "responses": ",".join(map(str, responses)),
                "selectmasks": ",".join(map(str, smasks)),
            })
        return pd.DataFrame(rows)

    train_df = seqs_to_df(train_seqs, list(range(N_FOLDS)))
    test_df = seqs_to_df(test_seqs, [-1])

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    print(f"Train/val secuencias: {len(train_df)}")
    print(f"Test secuencias: {len(test_df)}")
    print(f"Archivos guardados en {OUT_DIR}")
    return n_skills


if __name__ == "__main__":
    raw = download_raw()
    n_skills = preprocess(raw)
    print(f"\nn_skills={n_skills} — listo para entrenar.")
    print("Siguiente paso: python train.py")
