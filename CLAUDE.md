# CLAUDE.md — sward-model-training

## Qué es este repo
Script de entrenamiento offline del modelo SAKT (Self-Attentive Knowledge Tracing) con dataset ASSISTments 2015. El checkpoint `.pth` resultante se sube a S3 y es consumido por `sward-ms-recomendacion` para inferencia en tiempo real.

## Stack
- Python 3.11
- pyKT (framework de knowledge tracing)
- PyTorch >= 2.3 (MPS para Apple Silicon)
- Dataset: ASSISTments 2015 (~683K interacciones, 19K estudiantes, 100 skills)

## Estructura
```
train.py         ← entrenamiento SAKT con ASSISTments 2015
upload_s3.py     ← sube outputs/sakt_assist2015.pth a S3
outputs/         ← modelo y metadatos (ignorado por git)
data/            ← dataset descargado por pyKT (ignorado por git)
```

## Comandos clave
```bash
# Instalar dependencias
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Entrenar (~15-20 min en M4 Pro con MPS)
python train.py

# Subir a S3
python upload_s3.py
python upload_s3.py --version v2.0   # nueva versión
```

## Salidas
- `outputs/sakt_assist2015.pth` → checkpoint del modelo (state_dict + metadatos)
- `outputs/model_meta.json` → n_skills, hiperparámetros, AUC final

## Destino en S3
```
s3://sward-models/sakt/v1.0/model.pth
s3://sward-models/sakt/v1.0/model_meta.json
```

## Variable de entorno en ms-recomendacion
```
SAKT_MODEL_S3_KEY=sakt/v1.0/model.pth
AWS_S3_MODEL_BUCKET=sward-models
```

## Métricas esperadas (ASSISTments 2015)
- AUC test: ~0.74-0.76 (benchmark SAKT en literatura)

## Decisiones de diseño
- SAKT implementado desde cero (no depende del API interno de pyKT para el modelo)
  para tener control total del checkpoint y la inferencia en ms-recomendacion
- pyKT se usa solo para descarga y preprocesamiento del dataset
- MPS habilitado automáticamente en Apple Silicon
