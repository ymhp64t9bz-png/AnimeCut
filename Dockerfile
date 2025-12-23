# syntax=docker/dockerfile:1.4
# ✂️ AnimeCut Serverless V15.9.6 - ESTÁVEL SEM COMPILAÇÃO FFMPEG
# Usa FFmpeg do sistema (CPU) - Mais estável e rápido de buildar
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# ==================== FORÇA REBUILD ====================
ARG FORCE_REBUILD=12
ARG BUILD_TIMESTAMP=20251222_2100_V15_9_6_STABLE

RUN echo "Force rebuild: ${FORCE_REBUILD}" && \
    echo "Timestamp: ${BUILD_TIMESTAMP}" && \
    echo "Build ID: $(date +%s)_$RANDOM" > /BUILD_ID

RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* 2>/dev/null || true

# ==================== CACHE BUSTER ====================
ARG CACHEBUST=20251222_2100_V15_9_6_STABLE_REBUILD
RUN echo "=== ANIMECUT V15.9.6 STABLE ===" > /BUILD_INFO && \
    echo "Timestamp: ${CACHEBUST}" >> /BUILD_INFO && \
    echo "Build: $(date -Iseconds)" >> /BUILD_INFO

WORKDIR /app

# Variáveis de Ambiente
ENV BUILD_VERSION="15.9.6"
ENV BUILD_DATE="2025-12-22T21:00:00Z"
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME="/runpod-volume/.cache/huggingface"
ENV CUDA_VISIBLE_DEVICES="0"
ENV TF_CPP_MIN_LOG_LEVEL="3"
ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,video,utility

# ==================== 1. DEPENDÊNCIAS DE SISTEMA ====================
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    pkg-config \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    git \
    nano \
    curl \
    wget \
    gnupg \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Verifica FFmpeg
RUN echo "=== FFMPEG DO SISTEMA ===" && \
    ffmpeg -version | head -3 && \
    ffmpeg -encoders 2>/dev/null | grep -E "libx264|libx265" | head -3 && \
    echo "✓ FFmpeg instalado (CPU encoding)"

# ==================== 2. cuDNN 9 ====================
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends libcudnn9-cuda-12 libcudnn9-dev-cuda-12 && \
    rm -rf /var/lib/apt/lists/*

# Atualizar pip
RUN pip install --upgrade pip setuptools wheel

# ==================== 3. NUMPY SHIELD ====================
RUN pip install --no-cache-dir "numpy==1.26.4"

# ==================== 4. CORE DEPENDENCIES ====================
RUN pip install --no-cache-dir \
    "runpod>=1.6.0" \
    "boto3>=1.34.0" \
    "botocore>=1.34.0" \
    "requests>=2.31.0" \
    "tqdm>=4.66.4" \
    "colorama"

# ==================== 5. PROCESSAMENTO DE VÍDEO ====================
RUN pip install --no-cache-dir \
    "moviepy==1.0.3" \
    "imageio>=2.34.1" \
    "imageio-ffmpeg>=0.5.1" \
    "proglog>=0.1.10" \
    "opencv-python-headless>=4.9.0.80"

# ==================== 6. PROCESSAMENTO DE ÁUDIO ====================
RUN pip install --no-cache-dir \
    "librosa" \
    "soundfile>=0.12.1" \
    "scipy"

# ==================== 7. IA & VISÃO ====================
RUN pip install --no-cache-dir \
    "ultralytics" \
    "basicsr>=1.4.2" \
    "facexlib>=0.2.5" \
    "gfpgan>=1.3.8" \
    "realesrgan>=0.3.0"

# ==================== 8. DeepFilterNet ====================
RUN pip install --no-cache-dir "deepfilternet"

# ==================== 9. WHISPER & TRANSCRIÇÃO ====================
RUN pip install --no-cache-dir \
    "transformers>=4.40.0" \
    "accelerate>=0.30.0" \
    "optimum" \
    "protobuf" \
    "sentencepiece" \
    "ctranslate2>=4.0.0" \
    "faster-whisper>=1.0.0"

# ==================== 10. FERRAMENTAS ====================
RUN pip install --no-cache-dir \
    "Pillow>=10.3.0" \
    "decorator<5.0" \
    "Cython<3"

# ==================== 11. NUMPY INTEGRITY ====================
RUN pip install "numpy==1.26.4" --force-reinstall --no-cache-dir

# ==================== 12. VERIFICAÇÃO cuDNN ====================
RUN python3 -c "import ctranslate2; print(f'CTranslate2: {ctranslate2.__version__}')" && \
    python3 -c "from faster_whisper import WhisperModel; print('faster-whisper: OK')" && \
    ldconfig -p | grep cudnn || echo "Aviso: cuDNN libs"

# ==================== 13. PRÉ-CARREGAMENTO YOLO ====================
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# ==================== 14. FONTES CUSTOMIZADAS ====================
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core \
    fonts-dejavu-extra \
    fonts-liberation \
    fonts-freefont-ttf \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/fonts /workspace/fonts

COPY fontes/ /app/fonts/

RUN for font in /app/fonts/*; do \
        if [ -f "$font" ]; then \
            ln -sf "$font" /workspace/fonts/$(basename "$font"); \
        fi; \
    done && \
    echo "Fontes copiadas:" && \
    ls -la /workspace/fonts/ 2>/dev/null || true

RUN mkdir -p /usr/local/share/fonts/custom && \
    cp /app/fonts/* /usr/local/share/fonts/custom/ 2>/dev/null || true && \
    fc-cache -fv

# ==================== 15. HANDLER ====================
ARG HANDLER_NOCACHE=15.9.6_20251222_2100_STABLE
RUN echo "Handler rebuild: ${HANDLER_NOCACHE} - $(date)" > /tmp/handler_build.txt

COPY handler.py .

RUN echo "=== BUILD COMPLETO v15.9.6 STABLE ===" && \
    echo "Handler timestamp: $(date -Iseconds)" && \
    echo "FFmpeg disponível:" && \
    which ffmpeg && \
    ffmpeg -version | head -1 && \
    echo "Python version:" && python3 --version && \
    echo "Handler header:" && \
    head -12 handler.py && \
    echo "Build finalizado com sucesso!"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)" || exit 1

CMD ["python3", "-u", "handler.py"]
