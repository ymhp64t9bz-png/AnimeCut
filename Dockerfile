# ✂️ AnimeCut Serverless v10.0 - STABLE COMPATIBLE EDITION
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Cache Buster
ENV BUILD_DATE="2025-12-12_STABLE_V1"

# ==================== SISTEMA OPERACIONAL ====================
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME="/runpod-volume/.cache/huggingface"

# Bibliotecas essenciais de runtime E BUILD (necessário para compilar av 10.x)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    pkg-config \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    fonts-dejavu-core \
    imagemagick \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ==================== PYTHON & DEPENDÊNCIAS ====================
COPY requirements.txt .

# --no-cache-dir é vital para garantir downloads frescos
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==================== MODEL BAKING (OTIMIZAÇÃO) ====================
# Cria diretório de cache
RUN mkdir -p /runpod-volume/.cache/huggingface

# "Assa" o modelo Whisper na imagem para start instantâneo (sem download no boot)
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', download_root='/runpod-volume/.cache/huggingface')"

# ==================== CÓDIGO ====================
COPY handler.py .

CMD ["python3", "-u", "handler.py"]
