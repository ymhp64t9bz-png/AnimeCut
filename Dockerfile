# ✂️ AnimeCut Serverless V12.1 ULTRA-STABLE - RunPod Build
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Variáveis de Ambiente
ENV BUILD_DATE="V12_1_RUNPOD_FINAL"
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
    && rm -rf /var/lib/apt/lists/*

# Atualizar pip
RUN pip install --upgrade pip setuptools wheel

# ==================== 2. INSTALAÇÃO VIA REQUIREMENTS ====================
# Copiar requirements e instalar todas as dependências de uma vez
COPY requirements.txt .

# Instalar numpy PRIMEIRO para evitar conflitos
RUN pip install --no-cache-dir "numpy==1.26.4"

# Instalar todas as outras dependências
RUN pip install --no-cache-dir -r requirements.txt

# ==================== 3. FORÇA FINAL - NUMPY INTEGRITY ====================
RUN pip install "numpy==1.26.4" --force-reinstall --no-cache-dir

# ==================== 4. PRÉ-CARREGAMENTO YOLO ====================
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Copiar handler
COPY handler.py .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)" || exit 1

# Comando de entrada
CMD ["python3", "-u", "handler.py"]
