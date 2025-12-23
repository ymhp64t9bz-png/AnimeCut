# syntax=docker/dockerfile:1.4
# ✂️ AnimeCut Serverless V15.9.7 - FFMPEG COM NVENC COMPILADO
# Compila FFmpeg 5.1 com NVENC (versões estáveis e compatíveis)
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# ==================== FORÇA REBUILD ====================
ARG FORCE_REBUILD=15
ARG BUILD_TIMESTAMP=20251223_0130_V15_9_7_NVENC_COMPILE

RUN echo "Force rebuild: ${FORCE_REBUILD}" && \
    echo "Timestamp: ${BUILD_TIMESTAMP}" && \
    echo "Build ID: $(date +%s)_$RANDOM" > /BUILD_ID

RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* 2>/dev/null || true

# ==================== CACHE BUSTER ====================
ARG CACHEBUST=20251223_0130_V15_9_7_NVENC_COMPILE
RUN echo "=== ANIMECUT V15.9.7 NVENC ===" > /BUILD_INFO && \
    echo "Timestamp: ${CACHEBUST}" >> /BUILD_INFO && \
    echo "Build: $(date -Iseconds)" >> /BUILD_INFO

WORKDIR /app

# Variáveis de Ambiente
ENV BUILD_VERSION="15.9.7"
ENV BUILD_DATE="2025-12-23T01:30:00Z"
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
    xz-utils \
    yasm \
    nasm \
    cmake \
    libx264-dev \
    libx265-dev \
    libvpx-dev \
    libfdk-aac-dev \
    libmp3lame-dev \
    libopus-dev \
    libass-dev \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

# ==================== 2. NV-CODEC-HEADERS (NVENC SDK) ====================
# Versão n12.1.14.0 compatível com CUDA 12.x e FFmpeg 5.x/6.x
RUN cd /tmp && \
    git clone --branch n12.1.14.0 --depth 1 https://git.videolan.org/git/ffmpeg/nv-codec-headers.git && \
    cd nv-codec-headers && \
    make install PREFIX=/usr/local && \
    cd / && rm -rf /tmp/nv-codec-headers && \
    echo "✓ nv-codec-headers n12.1.14.0 instalado"

# ==================== 3. FFMPEG 5.1.4 COM NVENC ====================
# FFmpeg 5.1.x é estável e bem testado com NVENC
RUN cd /tmp && \
    wget -q https://ffmpeg.org/releases/ffmpeg-5.1.4.tar.xz && \
    tar -xf ffmpeg-5.1.4.tar.xz && \
    cd ffmpeg-5.1.4 && \
    ./configure \
        --prefix=/usr/local \
        --enable-gpl \
        --enable-nonfree \
        --enable-cuda-nvcc \
        --enable-cuvid \
        --enable-nvenc \
        --enable-nvdec \
        --enable-libnpp \
        --enable-libx264 \
        --enable-libx265 \
        --enable-libvpx \
        --enable-libfdk-aac \
        --enable-libmp3lame \
        --enable-libopus \
        --enable-libass \
        --enable-libfreetype \
        --disable-debug \
        --disable-doc \
        --disable-static \
        --enable-shared \
        --extra-cflags="-I/usr/local/cuda/include" \
        --extra-ldflags="-L/usr/local/cuda/lib64" && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd / && rm -rf /tmp/ffmpeg* && \
    echo "✓ FFmpeg 5.1.4 com NVENC compilado"

# Verifica FFmpeg e NVENC
RUN echo "=== VERIFICANDO FFMPEG ===" && \
    ffmpeg -version | head -5 && \
    echo "" && \
    echo "=== ENCODERS DISPONÍVEIS ===" && \
    ffmpeg -hide_banner -encoders 2>/dev/null | grep -E "264|265|nvenc|cuvid" | head -10 && \
    echo "" && \
    if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q "h264_nvenc"; then \
        echo "✓ NVENC h264_nvenc DISPONÍVEL!"; \
    else \
        echo "⚠ NVENC não listado (será verificado em runtime com GPU)"; \
    fi

# ==================== 4. cuDNN 9 ====================
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends libcudnn9-cuda-12 libcudnn9-dev-cuda-12 && \
    rm -rf /var/lib/apt/lists/*

# Atualizar pip
RUN pip install --upgrade pip setuptools wheel

# ==================== 5. NUMPY SHIELD ====================
RUN pip install --no-cache-dir "numpy==1.26.4"

# ==================== 6. CORE DEPENDENCIES ====================
RUN pip install --no-cache-dir \
    "runpod>=1.6.0" \
    "boto3>=1.34.0" \
    "botocore>=1.34.0" \
    "requests>=2.31.0" \
    "tqdm>=4.66.4" \
    "colorama"

# ==================== 7. PROCESSAMENTO DE VÍDEO ====================
RUN pip install --no-cache-dir \
    "moviepy==1.0.3" \
    "imageio>=2.34.1" \
    "imageio-ffmpeg>=0.5.1" \
    "proglog>=0.1.10" \
    "opencv-python-headless>=4.9.0.80"

# ==================== 8. PROCESSAMENTO DE ÁUDIO ====================
RUN pip install --no-cache-dir \
    "librosa" \
    "soundfile>=0.12.1" \
    "scipy"

# ==================== 9. IA & VISÃO ====================
RUN pip install --no-cache-dir \
    "ultralytics" \
    "basicsr>=1.4.2" \
    "facexlib>=0.2.5" \
    "gfpgan>=1.3.8" \
    "realesrgan>=0.3.0"

# ==================== 10. DeepFilterNet ====================
RUN pip install --no-cache-dir "deepfilternet"

# ==================== 11. WHISPER & TRANSCRIÇÃO ====================
RUN pip install --no-cache-dir \
    "transformers>=4.40.0" \
    "accelerate>=0.30.0" \
    "optimum" \
    "protobuf" \
    "sentencepiece" \
    "ctranslate2>=4.0.0" \
    "faster-whisper>=1.0.0"

# ==================== 12. FERRAMENTAS ====================
RUN pip install --no-cache-dir \
    "Pillow>=10.3.0" \
    "decorator<5.0" \
    "Cython<3"

# ==================== 13. NUMPY INTEGRITY ====================
RUN pip install "numpy==1.26.4" --force-reinstall --no-cache-dir

# ==================== 14. VERIFICAÇÃO cuDNN ====================
RUN python3 -c "import ctranslate2; print(f'CTranslate2: {ctranslate2.__version__}')" && \
    python3 -c "from faster_whisper import WhisperModel; print('faster-whisper: OK')" && \
    ldconfig -p | grep cudnn || echo "Aviso: cuDNN libs"

# ==================== 15. PRÉ-CARREGAMENTO YOLO ====================
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# ==================== 16. FONTES CUSTOMIZADAS ====================
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

# ==================== 17. HANDLER ====================
ARG HANDLER_NOCACHE=15.9.7_20251223_0130_NVENC_COMPILE
RUN echo "Handler rebuild: ${HANDLER_NOCACHE} - $(date)" > /tmp/handler_build.txt

COPY handler.py .

RUN echo "=== BUILD COMPLETO v15.9.7 ===" && \
    echo "Handler timestamp: $(date -Iseconds)" && \
    echo "FFmpeg:" && \
    which ffmpeg && \
    ffmpeg -version | head -2 && \
    echo "Encoders NVENC:" && \
    ffmpeg -hide_banner -encoders 2>/dev/null | grep nvenc || echo "NVENC: verificar em runtime" && \
    echo "Python:" && python3 --version && \
    echo "Handler:" && \
    head -12 handler.py && \
    echo "Build finalizado!"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)" || exit 1

CMD ["python3", "-u", "handler.py"]
