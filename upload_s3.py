"""
Sube el modelo entrenado a S3.

Uso:
    python upload_s3.py
    python upload_s3.py --version v2.0
"""

import argparse
import json
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

BUCKET = "sward-models"
DEFAULT_VERSION = "v1.0"
MODEL_PATH = Path("outputs/sakt_assist2015.pth")
META_PATH = Path("outputs/model_meta.json")


def upload(version: str):
    if not MODEL_PATH.exists():
        print(f"No se encontró el modelo en {MODEL_PATH}. Ejecuta train.py primero.")
        return

    s3 = boto3.client("s3")
    prefix = f"sakt/{version}"

    for local, s3_key in [(MODEL_PATH, f"{prefix}/model.pth"), (META_PATH, f"{prefix}/model_meta.json")]:
        if not local.exists():
            continue
        try:
            s3.upload_file(str(local), BUCKET, s3_key)
            print(f"✓ Subido: s3://{BUCKET}/{s3_key}")
        except ClientError as e:
            print(f"Error al subir {local}: {e}")
            return

    print(f"\nModelo disponible en: s3://{BUCKET}/{prefix}/model.pth")
    print("ms-recomendacion lo cargará automáticamente al reiniciar.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=DEFAULT_VERSION, help="Versión del modelo (ej: v1.0)")
    args = parser.parse_args()
    upload(args.version)
