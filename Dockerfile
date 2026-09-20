# Entorno reproducible de entrenamiento y evaluacion del SAKT.
#
# El repo pide Python 3.11 y pykt-toolkit no da garantias en versiones mas
# nuevas, asi que fijamos 3.11 aqui en vez de depender del Python del equipo.
#
# Se instala torch CPU explicitamente: la rueda por defecto arrastra CUDA
# (~2.5 GB) que no sirve en una maquina sin GPU NVIDIA.

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir torch>=2.3.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

# Sin comando por defecto: se invoca el script que toque
# (prepare_data.py, train.py, evaluation/compare_models.py...).
CMD ["python", "--version"]
