# ✂️ AnimeCut Serverless V15.8 - CORREÇÕES CRÍTICAS
# CORREÇÕES: Títulos únicos, PNG dtype, Image import, Fallback encoding 
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# ==================== CACHE BUSTER ====================
# IMPORTANTE: Mude este valor para forçar rebuild completo no RunPod
ARG CACHEBUST=20251218_2200_V15_8_CRITICAL_FIXES
RUN echo "Build timestamp: ${CACHEBUST}" > /BUILD_INFO && \
    echo "V15.8 - CORREÇÕES CRÍTICAS (Títulos, PNG, Encoding)" >> /BUILD_INFO

WORKDIR /app

# Variáveis de Ambiente
ENV BUILD_VERSION="15.8"
ENV BUILD_DATE="2025-12-18T22:00:00Z"
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME="/runpod-volume/.cache/huggingface"
ENV CUDA_VISIBLE_DEVICES="0"
ENV TF_CPP_MIN_LOG_LEVEL="3"
ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,video,utility

# ==================== 1. DEPENDÊNCIAS DE SISTEMA + cuDNN 9 ====================
# Instala cuDNN 9.x que é necessário para ctranslate2/faster-whisper recentes
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
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# ==================== 2. INSTALA cuDNN 9 via NVIDIA Repository ====================
# O faster-whisper/ctranslate2 recente PRECISA de cuDNN 9.x
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends libcudnn9-cuda-12 libcudnn9-dev-cuda-12 && \
    rm -rf /var/lib/apt/lists/*

# Atualizar pip, setuptools e wheel
RUN pip install --upgrade pip setuptools wheel

# ==================== 3. NUMPY SHIELD (CRÍTICO - PRIMEIRO) ====================
# Deve ser < 2.0 para compatibilidade com PyTorch e OpenCV
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

# ==================== 7. IA & VISÃO (YOLO + TOOLS) ====================
RUN pip install --no-cache-dir \
    "ultralytics" \
    "basicsr>=1.4.2" \
    "facexlib>=0.2.5" \
    "gfpgan>=1.3.8" \
    "realesrgan>=0.3.0"

# ==================== 8. DeepFilterNet (separado para controle de versão) ====================
RUN pip install --no-cache-dir "deepfilternet"

# ==================== 9. WHISPER & TRANSCRIÇÃO ====================
# CRÍTICO: ctranslate2 e faster-whisper com versões compatíveis com cuDNN 9
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

# ==================== 11. FORÇA FINAL - NUMPY INTEGRITY ====================
# Força numpy 1.26.4 no final para garantir integridade após todas as instalações
RUN pip install "numpy==1.26.4" --force-reinstall --no-cache-dir

# ==================== 12. VERIFICAÇÃO cuDNN ====================
# Verifica se cuDNN 9 está instalado corretamente
RUN python3 -c "import ctranslate2; print(f'CTranslate2: {ctranslate2.__version__}')" && \
    python3 -c "from faster_whisper import WhisperModel; print('faster-whisper: OK')" && \
    ldconfig -p | grep cudnn || echo "Aviso: cuDNN libs podem precisar de ldconfig"

# ==================== 13. PRÉ-CARREGAMENTO DE MODELOS ====================
# Pré-carrega YOLO para evitar delays na primeira requisição
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# ==================== 14. FONTES CUSTOMIZADAS ====================
# Instala pacotes de fontes do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core \
    fonts-dejavu-extra \
    fonts-liberation \
    fonts-freefont-ttf \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

# Cria diretórios de fontes
RUN mkdir -p /app/fonts /workspace/fonts

# Copia as fontes customizadas (pasta 'fontes' no repositório)
COPY fontes/ /app/fonts/

# Cria links simbólicos em /workspace/fonts (onde o handler procura)
RUN for font in /app/fonts/*; do \
        if [ -f "$font" ]; then \
            ln -sf "$font" /workspace/fonts/$(basename "$font"); \
        fi; \
    done && \
    echo "Fontes copiadas:" && \
    ls -la /workspace/fonts/

# Instala fontes no sistema
RUN mkdir -p /usr/local/share/fonts/custom && \
    cp /app/fonts/* /usr/local/share/fonts/custom/ 2>/dev/null || true && \
    fc-cache -fv

# ==================== 15. HANDLER - SEMPRE ATUALIZADO ====================
# Este ARG invalida o cache para SEMPRE copiar o handler mais recente
ARG HANDLER_VERSION=15.8_20251218_2200_CRITICAL_FIXES
RUN echo "Handler version: ${HANDLER_VERSION}"

# Copia handler (NUNCA usa cache devido ao ARG acima)
COPY handler.py .

# Mostra versão no build log
RUN echo "=== BUILD COMPLETO v15.8 ===" && \
    echo "Handler: ${HANDLER_VERSION}" && \
    echo "Correções: Títulos únicos, PNG dtype, Image import, Fallback encoding" && \
    echo "Fontes disponíveis:" && \
    ls -la /workspace/fonts/ && \
    head -20 handler.py

# Verifica se NVENC está disponível no build
RUN ffmpeg -encoders 2>/dev/null | grep nvenc || echo "NVENC será verificado em runtime"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)" || exit 1

# Comando de entrada
CMD ["python3", "-u", "handler.py"]
