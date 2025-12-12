# ✂️ AnimeCut Serverless v10.0 - STABLE COMPATIBLE EDITION
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Cache Buster
ENV BUILD_DATE="2025-12-12_STABLE_V1"

# ==================== SISTEMA OPERACIONAL ====================
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME="/runpod-volume/.cache/huggingface"

# Bibliotecas essenciais de runtime (FFmpeg, fontes, opencv libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    fonts-dejavu-core \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ==================== PYTHON & DEPENDÊNCIAS ====================
COPY requirements.txt .

# --no-cache-dir é vital para garantir downloads frescos
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==================== CÓDIGO ====================
COPY handler.py .

CMD ["python3", "-u", "handler.py"]
