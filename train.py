"""
Entrenamiento SAKT con ASSISTments 2015 usando pyKT.

Uso:
    python prepare_data.py   ← solo la primera vez
    python train.py

El modelo se guarda en outputs/sakt_assist2015.pth
Siguiente paso: python upload_s3.py
"""

# pyKT bug 1: qdkt.py importa turtle (requiere Tk). Mock antes de que pyKT cargue.
import os
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
# Dataset configurable por env: assist2015 (default) o moodle (data real exportada
# por export_moodle_to_pykt.py). Permite re-entrenar SAKT sobre conceptos de Moodle.
DATASET = os.environ.get("KT_DATASET", "assist2015")
DATA_DIR = Path(os.environ.get("KT_DATA_DIR", f"data/{DATASET}"))
SEQ_LEN = 200
EMB_SIZE = 256
NUM_HEADS = 8
DROPOUT = 0.2
NUM_ENCODER_LAYERS = 2
BATCH_SIZE = 64
EPOCHS = 200
LR = 1e-3
PATIENCE = 20
N_FOLDS = 5          # fold 0..4 → usamos fold 4 como validación

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PAD = -1


# ── Metadatos del dataset ─────────────────────────────────────────────────────
def build_concept_index(conceptos: list[str]) -> dict[str, int]:
    """Mapa concepto(str)→índice entero estable (orden de aparición).

    Lo consume ms-recomendacion para traducir el concepto (sección Moodle) al
    índice que entiende el modelo. Para ASSISTments es prácticamente identidad.
    """
    index: dict[str, int] = {}
    for c in conceptos:
        if c not in index:
            index[c] = len(index)
    return index


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


# ── Modelo SAKT (pyKT) con key_padding_mask ──────────────────────────────────
def build_model(n_skills: int):
    from pykt.models.sakt import SAKT, Blocks
    from pykt.models.utils import ut_mask
    import pykt.models.utils as _pykt_utils

    _pykt_utils.device = str(DEVICE)

    # Monkey-patch: añade key_padding_mask al bloque de atención de pyKT.
    # El checkpoint resultante es idéntico — solo cambia el forward de entrenamiento.
    def _blocks_forward(self, q=None, k=None, v=None, key_padding_mask=None):
        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        causal_mask = ut_mask(seq_len=k.shape[0])
        attn_emb, _ = self.attn(q, k, v, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
        attn_emb = self.attn_dropout(attn_emb)
        attn_emb, q = attn_emb.permute(1, 0, 2), q.permute(1, 0, 2)
        attn_emb = self.attn_layer_norm(q + attn_emb)
        emb = self.FFN(attn_emb)
        emb = self.FFN_dropout(emb)
        emb = self.FFN_layer_norm(attn_emb + emb)
        return emb

    def _sakt_forward(self, q, r, qry, qtest=False, key_padding_mask=None):
        qshftemb, xemb = self.base_emb(q, r, qry)
        for i in range(self.num_en):
            xemb = self.blocks[i](qshftemb, xemb, xemb, key_padding_mask=key_padding_mask)
        p = torch.sigmoid(self.pred(self.dropout_layer(xemb))).squeeze(-1)
        return p if not qtest else (p, xemb)

    import types
    Blocks.forward = _blocks_forward
    SAKT.forward = _sakt_forward

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
    q = dcur["cseqs"].long().to(DEVICE)
    r = dcur["rseqs"].long().to(DEVICE)
    qry = dcur["shft_cseqs"].long().to(DEVICE)
    target = dcur["shft_rseqs"].float().to(DEVICE)
    mask = dcur["smasks"].bool().to(DEVICE)
    # key_padding_mask: True en posiciones PAD → la atención las ignora.
    # dcur["masks"] es True donde la posición es válida, así que invertimos.
    key_padding_mask = (~dcur["masks"]).to(DEVICE)
    return q, r, qry, target, mask, key_padding_mask


# ── Epoch de entrenamiento ────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, epoch: int):
    model.train()
    total_loss, all_preds, all_targets = 0.0, [], []
    t0 = time.time()

    bar = tqdm(loader, desc=f"  Epoch {epoch:02d} [train]", unit="batch", dynamic_ncols=True)
    for step, dcur in enumerate(bar, 1):
        q, r, qry, target, mask, key_padding_mask = extract_batch(dcur)

        optimizer.zero_grad()
        y = model(q, r, qry, key_padding_mask=key_padding_mask)

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
        q, r, qry, target, mask, key_padding_mask = extract_batch(dcur)

        y = model(q, r, qry, key_padding_mask=key_padding_mask)
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

    # Índice concepto→entero embebido en el checkpoint (contrato con ms-recomendacion).
    # Si el dataset trae concept_index.json (Moodle: secciones→entero) se usa ese;
    # en ASSISTments los conceptos ya son enteros 0..n-1 → índice identidad.
    ci_path = DATA_DIR / "concept_index.json"
    if ci_path.exists():
        concept_index = json.loads(ci_path.read_text())
        print(f"concept_index cargado de {ci_path} ({len(concept_index)} conceptos)")
    else:
        concept_index = build_concept_index([str(i) for i in range(n_skills)])

    train_loader, valid_loader, test_loader = get_loaders(n_skills)
    print(f"Batches — train: {len(train_loader)}, val: {len(valid_loader)}, test: {len(test_loader)}\n")

    model = build_model(n_skills)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros del modelo: {total_params:,}\n")

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.BCELoss()  # pyKT SAKT ya aplica sigmoid
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5, min_lr=1e-6
    )

    best_val_auc = 0.0
    patience_counter = 0
    best_model_path = OUTPUT_DIR / f"sakt_{DATASET}.pth"
    train_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_auc, t_elapsed = train_epoch(model, train_loader, optimizer, criterion, epoch)
        val_loss, val_auc = eval_epoch(model, valid_loader, criterion, desc="[val]  ")
        scheduler.step(val_auc)

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
                    "concept_index": concept_index,
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

    # Export TorchScript: ms-recomendacion lo carga sin necesitar pyKT.
    # Resetear device global de pyKT a cpu antes de trazar (pos_encode lo usa).
    import pykt.models.utils as _pykt_utils_inf
    _pykt_utils_inf.device = "cpu"
    model_cpu = model.cpu().eval()
    q_ex = torch.zeros(1, SEQ_LEN, dtype=torch.long)
    r_ex = torch.zeros(1, SEQ_LEN, dtype=torch.long)
    qry_ex = torch.zeros(1, SEQ_LEN, dtype=torch.long)
    traced = torch.jit.trace(model_cpu, (q_ex, r_ex, qry_ex))
    traced_path = OUTPUT_DIR / "sakt_assist2015_traced.pt"
    torch.jit.save(traced, traced_path)
    print(f"Modelo TorchScript guardado: {traced_path}")

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
