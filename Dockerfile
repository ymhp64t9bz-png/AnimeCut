# ✂️ AnimeCut Serverless V2 TURBO - CORRIGIDO
# Base Image com PyTorch 2.2.1 + CUDA 12.1
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Diretório
WORKDIR /app

# Cache & Vars
ENV BUILD_DATE="2025-12-12_TURBO_V2_FIX"
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME="/runpod-volume/.cache/huggingface"

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
    && rm -rf /var/lib/apt/lists/*

# Atualiza pip
RUN pip install --upgrade pip

# ==================== 2. SEGURANÇA DE VERSÃO (CRÍTICO) ====================
# Instalamos isso PRIMEIRO para evitar conflito entre OpenCV e PyTorch
RUN pip install --no-cache-dir "numpy<2.0"

# ==================== 3. ARSENAL PYTHON ====================
RUN pip install --no-cache-dir \
    runpod>=1.6.0 \
    boto3>=1.34.0 \
    requests \
    tqdm \
    colorama \
    "moviepy>=2.0.0.dev2" \
    imageio-ffmpeg>=0.5.1 \
    "opencv-python-headless<=4.9.0.80" \
    Pillow \
    librosa \
    soundfile \
    ultralytics \
    proglog>=0.1.10 \
    deepfilternet

# ==================== 4. INSANELY FAST WHISPER (TURBO) ====================
RUN pip install --no-cache-dir \
    transformers \
    optimum \
    accelerate \
    scipy

# --- CORREÇÃO DO FLASH ATTENTION (O SEGREDO DA VELOCIDADE) ---
# Em vez de compilar, baixamos o binário pronto para Torch 2.2 + CUDA 12
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/flash_attn-2.5.6+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Instala o Wrapper do Insanely Fast Whisper
RUN pip install --no-cache-dir insanely-fast-whisper

# ==================== 5. TOOLS PRO (Upscale) ====================
RUN pip install --no-cache-dir \
    basicsr>=1.4.2 \
    facexlib>=0.2.5 \
    gfpgan>=1.3.8 \
    realesrgan>=0.3.0

# ==================== 6. CÓDIGO E INICIALIZAÇÃO ====================
# Pré-carrega YOLO
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

COPY handler.py .

CMD [ "python3", "-u", "handler.py" ]
