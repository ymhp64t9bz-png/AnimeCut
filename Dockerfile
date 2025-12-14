# ✂️ AnimeCut Serverless V3 FINAL - NUMPY SHIELDED
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Mude isso para forçar o RunPod a ler o novo arquivo (Cache Bust manual)
ENV BUILD_DATE="V12.2_TORCHVISION_FIX" 
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
# Forçamos a versão 1.26.4 que é a última estável da série 1.x
RUN pip install --no-cache-dir "numpy==1.26.4"

# ==================== 3. FLASH ATTENTION & GPU OPTIMIZATION ====================
# Instala Flash Attention para Whisper Turbo
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/flash_attn-2.5.6+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# ==================== 4. ARSENAL PYTHON CORE ====================
RUN pip install --no-cache-dir \
    runpod>=1.6.0 \
    boto3>=1.34.0 \
    requests \
    tqdm \
    colorama \
    "moviepy==1.0.3" \
    imageio-ffmpeg>=0.5.1 \
    "opencv-python-headless<=4.9.0.80" \
    Pillow \
    librosa \
    soundfile \
    ultralytics \
    proglog>=0.1.10 \
    # ==================== 5. WHISPER & IA (GPU FOCUSED) ====================
    RUN pip install --no-cache-dir \
    transformers \
    optimum \
    accelerate \
    scipy \
    faster-whisper \
    insanely-fast-whisper \
    protobuf \
    sentencepiece

# ==================== 6. CORREÇÃO DE COMPATIBILIDADE (CRÍTICO) ====================
# Garante que torchvision e torchaudio estejam alinhados com o Torch 2.2.1 da imagem base
RUN pip install --no-cache-dir "torchvision==0.17.1" "torchaudio==2.2.1" --index-url https://download.pytorch.org/whl/cu121 --force-reinstall

# ==================== 7. BLINDAGEM FINAL NUMPY ====================
# Reinstalamos numpy 1.26.4 forçadamente no final para garantir integridade
RUN pip install "numpy==1.26.4" --force-reinstall

# Pré-carrega YOLO
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

COPY handler.py .

CMD [ "python3", "-u", "handler.py" ]
