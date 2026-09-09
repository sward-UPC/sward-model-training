"""Evalúa un checkpoint ya entrenado sobre el conjunto de prueba.

train.py calcula el AUC de test al final del entrenamiento. Si la corrida se
detiene antes —porque convergió y el early stopping no llega a dispararse por
mejoras marginales que reinician la paciencia— ese número se pierde aunque el
checkpoint del mejor epoch sí esté guardado.

Este script lo recupera sin reentrenar. La configuración debe coincidir con la
del entrenamiento: se lee del propio checkpoint y se aplica antes de importar
train.py, porque ese módulo fija sus constantes al importarse.

Uso:
    python evaluate_checkpoint.py [ruta_checkpoint]
"""

import os
import sys

import torch

RUTA = sys.argv[1] if len(sys.argv) > 1 else "outputs/sakt_assist2015.pth"

ck = torch.load(RUTA, map_location="cpu", weights_only=True)

# La configuración vive dentro del checkpoint: se replica en el entorno para que
# train.py construya exactamente el mismo modelo y el mismo tamaño de ventana.
os.environ["KT_DATASET"] = ck.get("dataset", "assist2015")
os.environ["KT_SEQ_LEN"] = str(ck["seq_len"])
os.environ["KT_EMB_SIZE"] = str(ck["emb_size"])
os.environ["KT_HEADS"] = str(ck["n_heads"])
os.environ["KT_LAYERS"] = str(ck["n_layers"])

import train as T  # noqa: E402  (debe importarse después de fijar el entorno)

print(f"Checkpoint : {RUTA}")
print(f"Dataset    : {os.environ['KT_DATASET']}")
print(f"Config     : seq_len={ck['seq_len']} emb={ck['emb_size']} "
      f"heads={ck['n_heads']} layers={ck['n_layers']}")
print(f"Conceptos  : {ck['n_skills']}")
print()

modelo = T.build_model(ck["n_skills"])
modelo.load_state_dict(ck["model_state_dict"])

_, val_loader, test_loader = T.get_loaders(ck["n_skills"])
criterio = torch.nn.BCELoss()

val_loss, val_auc = T.eval_epoch(modelo, val_loader, criterio, "[val] ")
test_loss, test_auc = T.eval_epoch(modelo, test_loader, criterio, "[test]")

print(f"Validación : loss={val_loss:.4f}  AUC={val_auc:.4f}")
print(f"Prueba     : loss={test_loss:.4f}  AUC={test_auc:.4f}")
print()
print("Referencia: pyKT reporta ~0.72 para SAKT sobre ASSISTments 2015.")
print("La diferencia se atribuye a que no se realizó búsqueda de hiperparámetros.")
