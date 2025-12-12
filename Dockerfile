# ✂️ AnimeCut Serverless V2 TURBO - STATE OF THE ART
# Base Image com PyTorch 2.2.1 + CUDA 12.1 (Base Solida)
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Diretório
WORKDIR /app

# Cache & Vars
ENV BUILD_DATE="2025-12-12_TURBO_V1"
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME="/runpod-volume/.cache/huggingface"

# ==================== 1. DEPENDÊNCIAS DE SISTEMA (FFmpeg, Audio, Build) ====================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    pkg-config \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Atualiza pip
RUN pip install --upgrade pip

# ==================== 2. ARSENAL PYTHON ====================

# Utilitários Básicos
RUN pip install --no-cache-dir \
    runpod>=1.6.0 \
    boto3>=1.34.0 \
    requests \
    tqdm \
    colorama

# Mídia & Visão (MoviePy 2.0 + YOLO Anime + OpenCV)
RUN pip install --no-cache-dir \
    "moviepy>=2.0.0.dev2" \
    imageio-ffmpeg>=0.5.1 \
    opencv-python-headless \
    Pillow \
    librosa \
    soundfile \
    ultralytics \
    proglog>=0.1.10

# Áudio Cleaning (DeepFilterNet)
RUN pip install --no-cache-dir deepfilternet

# ==================== 3. INSANELY FAST WHISPER STACK ====================
# Instala as dependências para usar Flash Attention 2 e Transformers otimizados
RUN pip install --no-cache-dir \
    transformers \
    optimum \
    accelerate \
    scipy

# COMPILAÇÃO CRÍTICA DO FLASH ATTENTION 2
# Isso pode demorar uns minutos no build, mas garante velocidade extrema no Whisper.
RUN pip install --no-cache-dir --no-build-isolation flash-attn

# ==================== 4. TOOLS PRO (Upscale) ====================
RUN pip install --no-cache-dir \
    basicsr>=1.4.2 \
    facexlib>=0.2.5 \
    gfpgan>=1.3.8 \
    realesrgan>=0.3.0

# ==================== 5. MODEL BAKING (CACHE) ====================
# Cria diretório
RUN mkdir -p /runpod-volume/.cache/huggingface

# Baixa YOLOv8n (Nano) para cache (é minúsculo)
RUN pip install ultralytics && \
    python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Nota: O modelo Whisper V3 via Transformers será baixado na primeira execução ou podemos tentar pré-carregar
# Mas como insanity-fast-whisper usa pipeline, vamos deixar o handler gerenciar o load inicial via HF_HOME

# ==================== CÓDIGO ====================
COPY handler.py .

CMD [ "python3", "-u", "handler.py" ]
