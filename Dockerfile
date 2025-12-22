# syntax=docker/dockerfile:1.4
# ✂️ AnimeCut Serverless V15.9.7 - FFMPEG COM NVENC
# CORREÇÕES: FFmpeg compilado com suporte NVENC (GPU encoding)
# FORÇA REBUILD LIMPO - SEM CACHE
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# ==================== FORÇA REBUILD SEM CACHE ====================
ARG FORCE_REBUILD=10
ARG BUILD_TIMESTAMP=20251222_2130_V15_9_7_GNUTLS

RUN echo "Force rebuild: ${FORCE_REBUILD}" && \
    echo "Timestamp: ${BUILD_TIMESTAMP}" && \
    echo "Random: $(date +%s%N)" > /FORCE_REBUILD_$(date +%s) && \
    rm -f /FORCE_REBUILD_*

RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* 2>/dev/null || true

# ==================== CACHE BUSTER ====================
ARG CACHEBUST=20251222_2130_V15_9_7_GNUTLS
RUN echo "Build timestamp: ${CACHEBUST}" > /BUILD_INFO && \
    echo "V15.9.7 - FFmpeg com NVENC (gnutls)" >> /BUILD_INFO && \
    echo "Build ID: $(cat /proc/sys/kernel/random/uuid 2>/dev/null || echo $RANDOM)" >> /BUILD_INFO

WORKDIR /app

# Variáveis de Ambiente
ENV BUILD_VERSION="15.9.7"
ENV BUILD_DATE="2025-12-22T21:00:00Z"
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME="/runpod-volume/.cache/huggingface"
ENV CUDA_VISIBLE_DEVICES="0"
ENV TF_CPP_MIN_LOG_LEVEL="3"
ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,video,utility

# ==================== LIMPA CACHE APT ANTES DE INSTALAR ====================
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# ==================== 1. DEPENDÊNCIAS DE SISTEMA (SEM FFMPEG APT) ====================
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    && rm -rf /var/lib/apt/lists/*

# ==================== 2. FFMPEG COM NVENC - COMPILAÇÃO ====================
# Instala dependências para compilar FFmpeg com NVENC
RUN apt-get update && apt-get install -y --no-install-recommends \
    yasm \
    nasm \
    cmake \
    libtool \
    libc6 \
    libc6-dev \
    unzip \
    libx264-dev \
    libx265-dev \
    libnuma-dev \
    libvpx-dev \
    libfdk-aac-dev \
    libmp3lame-dev \
    libopus-dev \
    libass-dev \
    libfreetype6-dev \
    libgnutls28-dev \
    libssl-dev \
    libsdl2-dev \
    libva-dev \
    libvdpau-dev \
    libxcb1-dev \
    libxcb-shm0-dev \
    libxcb-xfixes0-dev \
    texinfo \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Instala NVIDIA Video Codec SDK headers (nv-codec-headers)
RUN git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git /tmp/nv-codec-headers && \
    cd /tmp/nv-codec-headers && \
    make install && \
    rm -rf /tmp/nv-codec-headers

# Compila FFmpeg com suporte NVENC (usando gnutls ao invés de openssl para evitar conflitos)
RUN git clone https://git.ffmpeg.org/ffmpeg.git /tmp/ffmpeg --depth 1 -b n6.1 && \
    cd /tmp/ffmpeg && \
    PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH" ./configure \
        --prefix=/usr/local \
        --enable-gpl \
        --enable-nonfree \
        --enable-cuda-nvcc \
        --enable-libnpp \
        --enable-cuvid \
        --enable-nvenc \
        --enable-nvdec \
        --enable-libx264 \
        --enable-libx265 \
        --enable-libvpx \
        --enable-libfdk-aac \
        --enable-libmp3lame \
        --enable-libopus \
        --enable-libass \
        --enable-libfreetype \
        --enable-gnutls \
        --enable-pic \
        --enable-shared \
        --disable-static \
        --disable-debug \
        --disable-doc \
        --extra-cflags="-I/usr/local/cuda/include" \
        --extra-ldflags="-L/usr/local/cuda/lib64" \
        --nvccflags="-gencode arch=compute_89,code=sm_89" && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    rm -rf /tmp/ffmpeg

# Verifica se NVENC foi compilado corretamente
RUN ffmpeg -encoders 2>/dev/null | grep -E "h264_nvenc|hevc_nvenc" && \
    echo "✓ FFmpeg com NVENC compilado com sucesso!"

# ==================== 3. INSTALA cuDNN 9 via NVIDIA Repository ====================
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends libcudnn9-cuda-12 libcudnn9-dev-cuda-12 && \
    rm -rf /var/lib/apt/lists/*

# Atualizar pip, setuptools e wheel
RUN pip install --upgrade pip setuptools wheel

# ==================== 4. NUMPY SHIELD (CRÍTICO - PRIMEIRO) ====================
RUN pip install --no-cache-dir "numpy==1.26.4"

# ==================== 5. CORE DEPENDENCIES ====================
RUN pip install --no-cache-dir \
    "runpod>=1.6.0" \
    "boto3>=1.34.0" \
    "botocore>=1.34.0" \
    "requests>=2.31.0" \
    "tqdm>=4.66.4" \
    "colorama"

# ==================== 6. PROCESSAMENTO DE VÍDEO ====================
RUN pip install --no-cache-dir \
    "moviepy==1.0.3" \
    "imageio>=2.34.1" \
    "imageio-ffmpeg>=0.5.1" \
    "proglog>=0.1.10" \
    "opencv-python-headless>=4.9.0.80"

# ==================== 7. PROCESSAMENTO DE ÁUDIO ====================
RUN pip install --no-cache-dir \
    "librosa" \
    "soundfile>=0.12.1" \
    "scipy"

# ==================== 8. IA & VISÃO (YOLO + TOOLS) ====================
RUN pip install --no-cache-dir \
    "ultralytics" \
    "basicsr>=1.4.2" \
    "facexlib>=0.2.5" \
    "gfpgan>=1.3.8" \
    "realesrgan>=0.3.0"

# ==================== 9. DeepFilterNet ====================
RUN pip install --no-cache-dir "deepfilternet"

# ==================== 10. WHISPER & TRANSCRIÇÃO ====================
RUN pip install --no-cache-dir \
    "transformers>=4.40.0" \
    "accelerate>=0.30.0" \
    "optimum" \
    "protobuf" \
    "sentencepiece" \
    "ctranslate2>=4.0.0" \
    "faster-whisper>=1.0.0"

# ==================== 11. FERRAMENTAS ====================
RUN pip install --no-cache-dir \
    "Pillow>=10.3.0" \
    "decorator<5.0" \
    "Cython<3"

# ==================== 12. FORÇA FINAL - NUMPY INTEGRITY ====================
RUN pip install "numpy==1.26.4" --force-reinstall --no-cache-dir

# ==================== 13. VERIFICAÇÃO cuDNN ====================
RUN python3 -c "import ctranslate2; print(f'CTranslate2: {ctranslate2.__version__}')" && \
    python3 -c "from faster_whisper import WhisperModel; print('faster-whisper: OK')" && \
    ldconfig -p | grep cudnn || echo "Aviso: cuDNN libs podem precisar de ldconfig"

# ==================== 14. PRÉ-CARREGAMENTO DE MODELOS ====================
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# ==================== 15. FONTES CUSTOMIZADAS ====================
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
    ls -la /workspace/fonts/

RUN mkdir -p /usr/local/share/fonts/custom && \
    cp /app/fonts/* /usr/local/share/fonts/custom/ 2>/dev/null || true && \
    fc-cache -fv

# ==================== 16. HANDLER - SEMPRE ATUALIZADO ====================
ARG HANDLER_NOCACHE=15.9.7_20251222_2130_GNUTLS
RUN echo "Handler rebuild: ${HANDLER_NOCACHE} - $(date)" > /tmp/handler_build.txt

COPY handler.py .

RUN echo "=== BUILD COMPLETO v15.9.7 ===" && \
    echo "Handler timestamp: $(date -Iseconds)" && \
    echo "Correções: FFmpeg compilado com NVENC" && \
    echo "Python version:" && python3 --version && \
    echo "FFmpeg version:" && ffmpeg -version | head -1 && \
    echo "NVENC encoders:" && ffmpeg -encoders 2>/dev/null | grep nvenc && \
    echo "Handler header:" && \
    head -15 handler.py && \
    echo "Build finalizado com sucesso!"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)" || exit 1

# Comando de entrada
CMD ["python3", "-u", "handler.py"]
