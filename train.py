"""
Entrenamiento SAKT con ASSISTments 2015 usando pyKT.

Uso:
    python prepare_data.py   ← solo la primera vez
    python train.py

El modelo se guarda en outputs/sakt_assist2015.pth
Siguiente paso: python upload_s3.py
"""

# pyKT bug 1: qdkt.py importa turtle (requiere Tk). Mock antes de que pyKT cargue.
import sys
import types as _types

if "turtle" not in sys.modules:
    _mock = _types.ModuleType("turtle")
    _mock.forward = lambda *a, **kw: None
    sys.modules["turtle"] = _mock


import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.utils.data import DataLoader
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
DATA_DIR = Path("data/assist2015")
SEQ_LEN = 200
EMB_SIZE = 256
NUM_HEADS = 8
DROPOUT = 0.2
NUM_ENCODER_LAYERS = 2
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
PATIENCE = 5
N_FOLDS = 5          # fold 0..4 → usamos fold 4 como validación

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PAD = -1


# ── Metadatos del dataset ─────────────────────────────────────────────────────
def get_n_skills() -> int:
    import pandas as pd

    df = pd.read_csv(DATA_DIR / "train_valid_sequences.csv")
    skills = set()
    for row in df["concepts"]:
        for x in str(row).split(","):
            x = x.strip()
            if x and x != str(PAD):
                skills.add(int(x))
    return max(skills) + 1


# ── Dataset propio (mismo formato CSV que pyKT, sin torch.cuda) ───────────────
class KTDatasetCPU(torch.utils.data.Dataset):
    """
    Lee los CSV generados por prepare_data.py en el formato que espera pyKT
    (columnas: fold, concepts, responses, selectmasks) pero usa tensores CPU.
    Compatible con el SAKT de pyKT: devuelve el mismo dict dcur.
    """

    def __init__(self, file_path: str, folds: set):
        import pandas as pd

        df = pd.read_csv(file_path)
        df = df[df["fold"].isin(folds)]

        self.cseqs, self.rseqs, self.smasks = [], [], []
        for _, row in df.iterrows():
            c = [int(x) for x in str(row["concepts"]).split(",")]
            r = [int(x) for x in str(row["responses"]).split(",")]
            s = [int(x) for x in str(row["selectmasks"]).split(",")]
            self.cseqs.append(c)
            self.rseqs.append(r)
            self.smasks.append(s)

        self.cseqs = torch.LongTensor(self.cseqs)
        self.rseqs = torch.FloatTensor(self.rseqs)
        self.smasks = torch.LongTensor(self.smasks)
        # mask: ambas posiciones (t y t+1) deben ser válidas (≠ PAD)
        self.masks = (self.cseqs[:, :-1] != PAD) & (self.cseqs[:, 1:] != PAD)
        # smasks: posiciones a evaluar (1 = válido)
        self.smasks = (self.smasks[:, 1:] != PAD)

    def __len__(self):
        return len(self.cseqs)

    def __getitem__(self, idx):
        # pyKT KTDataset devuelve dcur con pares (t, t+1) ya shiftados
        return {
            "cseqs":      self.cseqs[idx, :-1] * self.masks[idx],
            "rseqs":      self.rseqs[idx, :-1] * self.masks[idx],
            "shft_cseqs": self.cseqs[idx, 1:]  * self.masks[idx],
            "shft_rseqs": self.rseqs[idx, 1:]  * self.masks[idx],
            "masks":      self.masks[idx],
            "smasks":     self.smasks[idx],
        }


# ── DataLoaders ───────────────────────────────────────────────────────────────
def get_loaders(n_skills: int):
    train_file = str(DATA_DIR / "train_valid_sequences.csv")
    test_file = str(DATA_DIR / "test_sequences.csv")

    train_ds = KTDatasetCPU(train_file, folds=set(range(N_FOLDS - 1)))
    valid_ds = KTDatasetCPU(train_file, folds={N_FOLDS - 1})
    test_ds = KTDatasetCPU(test_file, folds={-1})

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, valid_loader, test_loader


# ── Modelo SAKT (pyKT) ────────────────────────────────────────────────────────
def build_model(n_skills: int):
    from pykt.models.sakt import SAKT
    import pykt.models.utils as _pykt_utils

    # pyKT calcula device como "cpu"/"cuda" sin saber de MPS.
    # Parcheamos la variable de módulo para que pos_encode y ut_mask
    # generen tensores en el mismo device que el modelo.
    _pykt_utils.device = str(DEVICE)

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


# ── Extracción de batch (formato pyKT KTDataset) ──────────────────────────────
def extract_batch(dcur):
    # KTDataset devuelve dict con claves: cseqs, rseqs, shft_cseqs, shft_rseqs, masks, smasks
    q = dcur["cseqs"].long().to(DEVICE)          # conceptos pasados
    r = dcur["rseqs"].long().to(DEVICE)          # respuestas pasadas (float en pyKT, pero usamos long)
    qry = dcur["shft_cseqs"].long().to(DEVICE)   # concepto siguiente (query)
    target = dcur["shft_rseqs"].float().to(DEVICE)
    mask = dcur["smasks"].bool().to(DEVICE)
    return q, r, qry, target, mask


# ── Epoch de entrenamiento ────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, epoch: int):
    model.train()
    total_loss, all_preds, all_targets = 0.0, [], []
    t0 = time.time()

    bar = tqdm(loader, desc=f"  Epoch {epoch:02d} [train]", unit="batch", dynamic_ncols=True)
    for step, dcur in enumerate(bar, 1):
        q, r, qry, target, mask = extract_batch(dcur)

        optimizer.zero_grad()
        # pyKT SAKT.forward devuelve solo p (ya con sigmoid); qtest=False
        y = model(q, r, qry)

        active_y = y[mask]
        active_t = target[mask]

        loss = criterion(active_y, active_t)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(active_y.detach().cpu().numpy())
        all_targets.extend(active_t.cpu().numpy())

        if step % 20 == 0 and len(set(all_targets)) > 1:
            running_auc = roc_auc_score(all_targets, all_preds)
            bar.set_postfix(loss=f"{total_loss/step:.4f}", auc=f"{running_auc:.4f}")

    elapsed = time.time() - t0
    auc = roc_auc_score(all_targets, all_preds) if len(set(all_targets)) > 1 else 0.0
    return total_loss / len(loader), auc, elapsed


# ── Epoch de evaluación ───────────────────────────────────────────────────────
@torch.no_grad()
def eval_epoch(model, loader, criterion, desc="[eval]"):
    model.eval()
    total_loss, all_preds, all_targets = 0.0, [], []

    for dcur in tqdm(loader, desc=f"  {desc}", unit="batch", dynamic_ncols=True, leave=False):
        q, r, qry, target, mask = extract_batch(dcur)

        y = model(q, r, qry)
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

    if not (DATA_DIR / "train_valid_sequences.csv").exists():
        print("Datos no encontrados. Ejecuta primero: python prepare_data.py")
        raise SystemExit(1)

    n_skills = get_n_skills()
    print(f"Skills únicos: {n_skills}")

    train_loader, valid_loader, test_loader = get_loaders(n_skills)
    print(f"Batches — train: {len(train_loader)}, val: {len(valid_loader)}, test: {len(test_loader)}\n")

    model = build_model(n_skills)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros del modelo: {total_params:,}\n")

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.BCELoss()  # pyKT SAKT ya aplica sigmoid
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_val_auc = 0.0
    patience_counter = 0
    best_model_path = OUTPUT_DIR / "sakt_assist2015.pth"
    train_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_auc, t_elapsed = train_epoch(model, train_loader, optimizer, criterion, epoch)
        val_loss, val_auc = eval_epoch(model, valid_loader, criterion, desc="[val]  ")
        scheduler.step(val_loss)

        avg_time = (time.time() - train_start) / epoch
        eta_sec = avg_time * (EPOCHS - epoch)
        eta_str = f"{int(eta_sec // 60)}m{int(eta_sec % 60):02d}s"
        marker = "✓ MEJOR" if val_auc > best_val_auc else f"  sin mejora {patience_counter + 1}/{PATIENCE}"

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss={train_loss:.4f} AUC={train_auc:.4f} | "
            f"val loss={val_loss:.4f} AUC={val_auc:.4f} | "
            f"{t_elapsed:.0f}s | ETA {eta_str} | {marker}"
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
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping en epoch {epoch}.")
                break

    total_time = time.time() - train_start
    print(f"\nEntrenamiento completado en {int(total_time // 60)}m{int(total_time % 60):02d}s")

    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_auc = eval_epoch(model, test_loader, criterion, desc="[test] ")

    print(f"\n=== Resultado final ===")
    print(f"Test AUC: {test_auc:.4f}  (esperado: ~0.74-0.76)")
    print(f"Modelo guardado: {best_model_path}")
    print(f"\nSiguiente paso: python upload_s3.py")

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


if __name__ == "__main__":
    main()
