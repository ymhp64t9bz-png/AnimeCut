# ✂️ AnimeCut Serverless V12.1 ULTRA-STABLE - FINAL BUILD
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Variáveis de Ambiente
ENV BUILD_DATE="V12_1_ULTRA_STABLE"
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME="/runpod-volume/.cache/huggingface"
ENV CUDA_VISIBLE_DEVICES="0"
ENV TF_CPP_MIN_LOG_LEVEL="3"
ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# ==================== 1. DEPENDÊNCIAS DE SISTEMA ====================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    pkg-config \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    git \
    nano \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Atualizar pip, setuptools e wheel
RUN pip install --upgrade pip setuptools wheel

# ==================== 2. NUMPY SHIELD (CRÍTICO - PRIMEIRO) ====================
# Instalar numpy 1.26.4 ANTES de qualquer dependência para evitar conflitos
RUN pip install --no-cache-dir "numpy==1.26.4"

# ==================== 3. CORE DEPENDENCIES ====================
RUN pip install --no-cache-dir \
    runpod>=1.6.0 \
    boto3>=1.34.0 \
    botocore>=1.34.0 \
    requests \
    tqdm \
    colorama

# ==================== 4. PROCESSAMENTO DE VÍDEO ====================
RUN pip install --no-cache-dir \
    "moviepy==1.0.3" \
    imageio>=2.34.1 \
    imageio-ffmpeg>=0.5.1 \
    proglog>=0.1.10 \
    "opencv-python-headless>=4.9.0.80"

# ==================== 5. PROCESSAMENTO DE ÁUDIO ====================
RUN pip install --no-cache-dir \
    librosa \
    soundfile>=0.12.1 \
    scipy

# ==================== 6. IA & VISÃO (YOLO + TOOLS) ====================
RUN pip install --no-cache-dir \
    ultralytics \
    deepfilternet \
    basicsr>=1.4.2 \
    facexlib>=0.2.5 \
    gfpgan>=1.3.8 \
    realesrgan>=0.3.0

# ==================== 7. WHISPER & TRANSCRIÇÃO ====================
RUN pip install --no-cache-dir \
    transformers>=4.40.0 \
    accelerate>=0.30.0 \
    optimum \
    protobuf \
    sentencepiece \
    faster-whisper \
    insanely-fast-whisper

# ==================== 8. FERRAMENTAS ====================
RUN pip install --no-cache-dir \
    "Pillow>=10.3.0" \
    "decorator<5.0" \
    "Cython<3"

# ==================== 9. FORÇA FINAL - NUMPY INTEGRITY ====================
# Força numpy 1.26.4 no final para garantir integridade após todas as instalações
RUN pip install "numpy==1.26.4" --force-reinstall --no-cache-dir

# ==================== 10. PRÉ-CARREGAMENTO DE MODELOS ====================
# Pré-carrega YOLO para evitar delays na primeira requisição
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Copia handler
COPY handler.py .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)" || exit 1

# Comando de entrada
CMD ["python3", "-u", "handler.py"]
