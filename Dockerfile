# ✂️ AnimeCut Serverless V4 FINAL - NUMPY CONSTRAINED & FLASH WHEEL
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Cache Busting & Vars
ENV BUILD_DATE="2025-12-12_V4_FINAL" 
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME="/runpod-volume/.cache/huggingface"

# ==================== 0. BLINDAGEM DE DEPENDÊNCIAS (CONSTRAINTS) ====================
# Cria arquivo de constraints para IMPEDIR que qualquer pacote instale numpy 2.0
RUN echo "numpy<2.0" > /app/constraints.txt
ENV PIP_CONSTRAINT="/app/constraints.txt"

# 1. Sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3-dev pkg-config ffmpeg libsndfile1 \
    libgl1 libglib2.0-0 git nano \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# ==================== 2. FLASH ATTENTION (VIA WHEEL - RÁPIDO) ====================
# Instala direto do binário para pular compilação de 1h
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/flash_attn-2.5.6+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# ==================== 3. ARSENAL PYTHON ====================
# Nota: PIP_CONSTRAINT garante que numpy fique < 2.0
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

# ==================== 4. WHISPER & IA ====================
RUN pip install --no-cache-dir \
    transformers \
    optimum \
    accelerate \
    scipy \
    insanely-fast-whisper

# ==================== 5. TOOLS PRO ====================
RUN pip install --no-cache-dir \
    basicsr>=1.4.2 \
    facexlib>=0.2.5 \
    gfpgan>=1.3.8 \
    realesrgan>=0.3.0

# ==================== 6. SETUP FINAL ====================
# Pré-carrega YOLO
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

COPY handler.py .

CMD [ "python3", "-u", "handler.py" ]
