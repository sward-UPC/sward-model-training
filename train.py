"""
Entrenamiento SAKT con ASSISTments 2015.

Uso:
    python train.py

El modelo entrenado se guarda en outputs/sakt_assist2015.pth
Luego subirlo a S3:
    aws s3 cp outputs/sakt_assist2015.pth s3://sward-models/sakt/v1.0/model.pth
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ── Device ──────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Dispositivo: {DEVICE}")

# ── Hiperparámetros ──────────────────────────────────────────────────────────
SEQ_LEN = 200
EMB_SIZE = 256
NUM_HEADS = 8
DROPOUT = 0.2
NUM_ENCODER_LAYERS = 2
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
PATIENCE = 5  # early stopping

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Dataset ──────────────────────────────────────────────────────────────────
class KTDataset(Dataset):
    """
    Dataset de Knowledge Tracing a partir de secuencias pyKT.
    Cada fila del CSV tiene la secuencia completa de un estudiante.
    """

    def __init__(self, df, n_skills: int, seq_len: int):
        self.samples = []
        for _, row in df.iterrows():
            skills = [int(x) for x in str(row["concepts"]).split(",")]
            answers = [int(x) for x in str(row["responses"]).split(",")]
            # Partir en ventanas de seq_len
            for start in range(0, len(skills), seq_len):
                s = skills[start : start + seq_len]
                a = answers[start : start + seq_len]
                if len(s) < 2:
                    continue
                self.samples.append((s, a, n_skills))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        skills, answers, n_skills = self.samples[idx]
        seq_len = len(skills)

        q = torch.zeros(SEQ_LEN, dtype=torch.long)
        r = torch.zeros(SEQ_LEN, dtype=torch.long)
        mask = torch.zeros(SEQ_LEN, dtype=torch.bool)

        q[:seq_len] = torch.tensor(skills, dtype=torch.long)
        r[:seq_len] = torch.tensor(answers, dtype=torch.long)
        mask[:seq_len] = True

        # Entrada: pares (skill, respuesta) del pasado
        # Target: respuesta del step siguiente
        return q[:-1], r[:-1], q[1:], r[1:], mask[1:]


# ── Modelo SAKT ──────────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        out, _ = self.attn(q, k, v, key_padding_mask=mask)
        return self.norm(q + self.dropout(out))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.norm(x + self.dropout(self.net(x)))


class SAKT(nn.Module):
    """
    Self-Attentive Knowledge Tracing (Pandey & Karypis, 2019).
    https://arxiv.org/abs/1907.06837
    """

    def __init__(self, n_skills: int, seq_len: int, emb_size: int, n_heads: int, dropout: float, n_layers: int = 2):
        super().__init__()
        self.n_skills = n_skills
        self.emb_size = emb_size

        # Embedding de interacciones: skill × 2 (correcto/incorrecto)
        self.interaction_emb = nn.Embedding(n_skills * 2 + 2, emb_size)
        # Embedding de ejercicios (para el query)
        self.exercise_emb = nn.Embedding(n_skills + 1, emb_size)
        # Embedding posicional
        self.pos_emb = nn.Embedding(seq_len, emb_size)

        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        MultiHeadAttention(emb_size, n_heads, dropout),
                        FeedForward(emb_size, emb_size * 4, dropout),
                    ]
                )
                for _ in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(emb_size, 1)

    def forward(self, q_past, r_past, q_next, padding_mask=None):
        bs, seq = q_past.shape
        pos = torch.arange(seq, device=q_past.device).unsqueeze(0)

        # Interacción pasada: (skill_id + n_skills * respuesta)
        interaction = q_past + self.n_skills * r_past
        interaction = torch.clamp(interaction, 0, self.n_skills * 2 + 1)

        x = self.interaction_emb(interaction) + self.pos_emb(pos)
        q = self.exercise_emb(q_next) + self.pos_emb(pos)

        # Causal mask: no ver el futuro
        causal = torch.triu(torch.ones(seq, seq, device=q_past.device), diagonal=1).bool()

        for attn, ff in self.layers:
            q = attn(q, x, x, mask=None)
            q = ff(q)

        logits = self.out(self.dropout(q)).squeeze(-1)
        return logits


# ── Datos (pyKT) ─────────────────────────────────────────────────────────────
def load_pykt_data():
    """Carga ASSISTments 2015 usando pyKT."""
    try:
        import pandas as pd
        from pykt.datasets.data_preprocess import process_raw_data

        data_dir = Path("data/assist2015")
        data_dir.mkdir(parents=True, exist_ok=True)

        # pyKT descarga y preprocesa si no existe
        train_path = data_dir / "train_valid_sequences.csv"
        test_path = data_dir / "test_sequences.csv"

        if not train_path.exists():
            print("Descargando y preprocesando ASSISTments 2015 con pyKT...")
            process_raw_data("assist2015", str(data_dir))

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Detectar número de skills
        all_skills = []
        for df in [train_df, test_df]:
            for row in df["concepts"]:
                all_skills.extend([int(x) for x in str(row).split(",")])
        n_skills = max(all_skills) + 1

        print(f"Skills únicos: {n_skills}")
        print(f"Estudiantes train: {len(train_df)}, test: {len(test_df)}")

        return train_df, test_df, n_skills

    except ImportError:
        raise ImportError("Instala pykt-toolkit: pip install pykt-toolkit")


# ── Entrenamiento ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, all_preds, all_targets = 0.0, [], []

    for q_past, r_past, q_next, r_next, mask in tqdm(loader, desc="Train", leave=False):
        q_past, r_past, q_next = q_past.to(DEVICE), r_past.to(DEVICE), q_next.to(DEVICE)
        r_next, mask = r_next.to(DEVICE), mask.to(DEVICE)

        optimizer.zero_grad()
        logits = model(q_past, r_past, q_next)

        # Solo calcular loss en posiciones no-padding
        active_logits = logits[mask]
        active_labels = r_next[mask].float()

        loss = criterion(active_logits, active_labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(torch.sigmoid(active_logits).detach().cpu().numpy())
        all_targets.extend(active_labels.cpu().numpy())

    auc = roc_auc_score(all_targets, all_preds) if len(set(all_targets)) > 1 else 0.0
    return total_loss / len(loader), auc


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_targets = 0.0, [], []

    for q_past, r_past, q_next, r_next, mask in tqdm(loader, desc="Eval", leave=False):
        q_past, r_past, q_next = q_past.to(DEVICE), r_past.to(DEVICE), q_next.to(DEVICE)
        r_next, mask = r_next.to(DEVICE), mask.to(DEVICE)

        logits = model(q_past, r_past, q_next)
        active_logits = logits[mask]
        active_labels = r_next[mask].float()

        total_loss += criterion(active_logits, active_labels).item()
        all_preds.extend(torch.sigmoid(active_logits).cpu().numpy())
        all_targets.extend(active_labels.cpu().numpy())

    auc = roc_auc_score(all_targets, all_preds) if len(set(all_targets)) > 1 else 0.0
    return total_loss / len(loader), auc


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=== Entrenamiento SAKT — ASSISTments 2015 ===\n")

    train_df, test_df, n_skills = load_pykt_data()

    # Split train/val (80/20)
    val_size = int(len(train_df) * 0.2)
    val_df = train_df.iloc[:val_size]
    train_df = train_df.iloc[val_size:]

    train_ds = KTDataset(train_df, n_skills, SEQ_LEN)
    val_ds = KTDataset(val_df, n_skills, SEQ_LEN)
    test_ds = KTDataset(test_df, n_skills, SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)

    print(f"Train samples: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}\n")

    model = SAKT(
        n_skills=n_skills,
        seq_len=SEQ_LEN,
        emb_size=EMB_SIZE,
        n_heads=NUM_HEADS,
        dropout=DROPOUT,
        n_layers=NUM_ENCODER_LAYERS,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros del modelo: {total_params:,}\n")

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_val_auc = 0.0
    patience_counter = 0
    best_model_path = OUTPUT_DIR / "sakt_assist2015.pth"

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_auc = eval_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d}/{EPOCHS} | Train loss={train_loss:.4f} AUC={train_auc:.4f} | Val loss={val_loss:.4f} AUC={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "n_skills": n_skills,
                    "seq_len": SEQ_LEN,
                    "emb_size": EMB_SIZE,
                    "n_heads": NUM_HEADS,
                    "dropout": DROPOUT,
                    "n_layers": NUM_ENCODER_LAYERS,
                    "val_auc": val_auc,
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
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Modelo guardado en: {best_model_path}")
    print(f"\nPara subir a S3:")
    print(f"  aws s3 cp {best_model_path} s3://sward-models/sakt/v1.0/model.pth")

    # Guardar metadatos del modelo
    meta = {
        "dataset": "assist2015",
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
    print(f"Metadatos en: {OUTPUT_DIR}/model_meta.json")


if __name__ == "__main__":
    main()
