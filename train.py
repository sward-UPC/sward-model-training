"""
Entrenamiento SAKT con ASSISTments 2015 usando pyKT.

Uso:
    python train.py

El modelo entrenado se guarda en outputs/sakt_assist2015.pth
Luego subir a S3:
    python upload_s3.py
    # o manualmente:
    aws s3 cp outputs/sakt_assist2015.pth s3://sward-models/sakt/v1.0/model.pth
"""

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from tqdm import tqdm

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Dispositivo: {DEVICE}")

# ── Hiperparámetros ───────────────────────────────────────────────────────────
DATASET = "assist2015"
SEQ_LEN = 200
EMB_SIZE = 256
NUM_HEADS = 8
DROPOUT = 0.2
NUM_ENCODER_LAYERS = 2
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
PATIENCE = 5

OUTPUT_DIR = Path("outputs")
DATA_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


# ── Preprocesamiento de datos ─────────────────────────────────────────────────
def prepare_data():
    """Descarga y preprocesa ASSISTments 2015 con pyKT."""
    from pykt.datasets.data_preprocess import process_raw_data

    dataset_dir = DATA_DIR / DATASET
    train_file = dataset_dir / "train_valid_sequences.csv"

    if not train_file.exists():
        print(f"Preprocesando {DATASET} con pyKT...")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        process_raw_data(DATASET, str(dataset_dir))
        print("Preprocesamiento completado.")
    else:
        print(f"Datos ya preprocesados en {dataset_dir}")

    return str(dataset_dir)


# ── DataLoaders (pyKT) ────────────────────────────────────────────────────────
def get_loaders(data_dir: str):
    """Inicializa los DataLoaders usando pyKT."""
    from pykt.datasets.init_dataset import init_dataset4train

    loaders = init_dataset4train(
        dataset_name=DATASET,
        model_name="sakt",
        dpath=data_dir,
        fname="train_valid_sequences.csv",
        emb_type="qid",
        seq_len=SEQ_LEN,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        batch_size=BATCH_SIZE,
    )
    # init_dataset4train devuelve: train, valid, test, test_window (y a veces más)
    train_loader, valid_loader, test_loader = loaders[0], loaders[1], loaders[2]
    return train_loader, valid_loader, test_loader


# ── Modelo (pyKT SAKT) ────────────────────────────────────────────────────────
def build_model(n_skills: int):
    """Instancia el modelo SAKT de pyKT."""
    from pykt.models.sakt import SAKT

    model = SAKT(
        num_c=n_skills,
        seq_len=SEQ_LEN,
        emb_size=EMB_SIZE,
        num_attn_heads=NUM_HEADS,
        dropout=DROPOUT,
        num_en=NUM_ENCODER_LAYERS,
        emb_type="qid",
    )
    return model.to(DEVICE)


# ── Loop de entrenamiento ─────────────────────────────────────────────────────
def get_n_skills(data_dir: str) -> int:
    """Lee el número de skills del archivo de metadatos generado por pyKT."""
    import pandas as pd

    meta_path = Path(data_dir) / "keyid2idx.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        return len(meta.get("concepts", meta))

    # Fallback: leer del CSV
    df = pd.read_csv(Path(data_dir) / "train_valid_sequences.csv")
    all_skills = []
    for row in df["concepts"]:
        all_skills.extend([int(x) for x in str(row).split(",") if x.strip()])
    return max(all_skills) + 1


def extract_batch(batch):
    """
    Extrae tensores del batch de pyKT.
    pyKT devuelve un dict 'dcur' con las secuencias.
    """
    dcur = batch
    if isinstance(batch, (list, tuple)):
        dcur = batch[0]

    q = dcur.get("qseqs", dcur.get("questions")).long().to(DEVICE)
    r = dcur.get("rseqs", dcur.get("responses")).long().to(DEVICE)
    qry = dcur.get("qshfseqs", q).long().to(DEVICE)       # query: pregunta siguiente
    target = dcur.get("rshfseqs", r).float().to(DEVICE)   # target: respuesta siguiente
    mask = dcur.get("smasks", torch.ones_like(r, dtype=torch.bool)).bool().to(DEVICE)

    return q, r, qry, target, mask


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, all_preds, all_targets = 0.0, [], []

    for batch in tqdm(loader, desc="Train", leave=False):
        q, r, qry, target, mask = extract_batch(batch)

        optimizer.zero_grad()
        # pyKT SAKT devuelve (y, reg_loss) — y son probabilidades (post-sigmoid)
        y, reg_loss = model(q, r, qry)

        active_y = y[mask]
        active_t = target[mask]

        loss = criterion(active_y, active_t) + reg_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(active_y.detach().cpu().numpy())
        all_targets.extend(active_t.cpu().numpy())

    auc = roc_auc_score(all_targets, all_preds) if len(set(all_targets)) > 1 else 0.0
    return total_loss / len(loader), auc


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_targets = 0.0, [], []

    for batch in tqdm(loader, desc="Eval", leave=False):
        q, r, qry, target, mask = extract_batch(batch)

        y, reg_loss = model(q, r, qry)
        active_y = y[mask]
        active_t = target[mask]

        total_loss += criterion(active_y, active_t).item()
        all_preds.extend(active_y.cpu().numpy())
        all_targets.extend(active_t.cpu().numpy())

    auc = roc_auc_score(all_targets, all_preds) if len(set(all_targets)) > 1 else 0.0
    return total_loss / len(loader), auc


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=== Entrenamiento SAKT — ASSISTments 2015 ===\n")

    data_dir = prepare_data()
    n_skills = get_n_skills(data_dir)
    print(f"Skills únicos: {n_skills}\n")

    train_loader, valid_loader, test_loader = get_loaders(data_dir)

    model = build_model(n_skills)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros del modelo: {total_params:,}\n")

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    # pyKT SAKT ya aplica sigmoid, usamos BCELoss (no BCEWithLogitsLoss)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_val_auc = 0.0
    patience_counter = 0
    best_model_path = OUTPUT_DIR / "sakt_assist2015.pth"

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_auc = eval_epoch(model, valid_loader, criterion)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss={train_loss:.4f} AUC={train_auc:.4f} | "
            f"Val loss={val_loss:.4f} AUC={val_auc:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "n_skills": n_skills,
                    "seq_len": SEQ_LEN,
                    "emb_size": EMB_SIZE,
                    "n_heads": NUM_HEADS,
                    "dropout": DROPOUT,
                    "n_layers": NUM_ENCODER_LAYERS,
                    "val_auc": round(val_auc, 4),
                    "epoch": epoch,
                },
                best_model_path,
            )
            print(f"  ✓ Mejor modelo guardado (val_auc={val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping en epoch {epoch}")
                break

    # Evaluación final en test
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_auc = eval_epoch(model, test_loader, criterion)

    print(f"\n=== Resultado final ===")
    print(f"Test AUC: {test_auc:.4f}  (esperado: ~0.74-0.76)")
    print(f"Modelo: {best_model_path}")
    print(f"\nSiguiente paso:")
    print(f"  python upload_s3.py")

    meta = {
        "dataset": DATASET,
        "n_skills": n_skills,
        "seq_len": SEQ_LEN,
        "emb_size": EMB_SIZE,
        "n_heads": NUM_HEADS,
        "dropout": DROPOUT,
        "n_layers": NUM_ENCODER_LAYERS,
        "test_auc": round(test_auc, 4),
        "best_val_auc": round(best_val_auc, 4),
    }
    with open(OUTPUT_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadatos: {OUTPUT_DIR}/model_meta.json")


if __name__ == "__main__":
    main()
