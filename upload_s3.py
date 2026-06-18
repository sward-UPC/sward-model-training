"""
Sube el modelo entrenado a S3.

Uso:
    python upload_s3.py                         # assist2015 → sakt/assist2015/
    python upload_s3.py --dataset moodle        # sakt_moodle.pth → sakt/moodle/
    python upload_s3.py --dataset moodle --version v2  # → sakt/v2/
"""

import argparse
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

BUCKET = "sward-models"
OUTPUTS = Path("outputs")


def upload(dataset: str, version: str) -> None:
    model_path = OUTPUTS / f"sakt_{dataset}.pth"
    if not model_path.exists():
        print(f"No se encontró el modelo en {model_path}. Ejecuta train.py primero.")
        return

    s3 = boto3.client("s3")
    prefix = f"sakt/{version}"
    archivos = [
        (model_path, f"{prefix}/model.pth"),
        (OUTPUTS / f"sakt_{dataset}_traced.pt", f"{prefix}/model_traced.pt"),
        (OUTPUTS / "model_meta.json", f"{prefix}/model_meta.json"),
    ]
    for local, s3_key in archivos:
        if not local.exists():
            continue
        try:
            s3.upload_file(str(local), BUCKET, s3_key)
            print(f"✓ Subido: s3://{BUCKET}/{s3_key}")
        except ClientError as e:
            print(f"Error al subir {local}: {e}")
            return

    print(f"\nModelo disponible en: s3://{BUCKET}/{prefix}/model.pth")
    print("Apunta SAKT_MODEL_S3_KEY a esa key en ms-recomendacion y redeploy.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="assist2015", help="Dataset del modelo (assist2015 | moodle)")
    parser.add_argument("--version", default=None, help="Prefijo en S3 (default: = dataset)")
    args = parser.parse_args()
    upload(args.dataset, args.version or args.dataset)
