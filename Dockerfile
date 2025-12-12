# ✂️ AnimeCut Serverless v11.0 - PRO EDITION (Zoom Tático + RealESRGAN)
# Base Image com PyTorch 2.2.1 (Já contém CUDA 12.1 e Python 3.10) - Economia de tempo.
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Define o diretório de trabalho
WORKDIR /app

# Cache Data
ENV BUILD_DATE="2025-12-12_PRO_V1"
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME="/runpod-volume/.cache/huggingface"

# ==================== DEPENDÊNCIAS DE SISTEMA ====================
# Instala libs para processamento de vídeo/áudio e suporte a build (caso precise compilar av)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    pkg-config \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    # Libs para OpenCV e PyAV
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    && rm -rf /var/lib/apt/lists/*

# Atualiza o pip
RUN pip install --upgrade pip

# ==================== PACOTES PYTHON (Agrupados) ====================

# 1. Infraestrutura e Fix do Cython (Vital para o 'av' antigo)
RUN pip install "Cython<3" wheel setuptools && \
    pip install --no-cache-dir \
    runpod>=1.6.0 \
    boto3>=1.34.0 \
    requests \
    tqdm

# 2. Processamento de Mídia (Edição de Vídeo)
# MoviePy 2.0.0.dev2 corrige o erro com Numpy novo
RUN pip install --no-cache-dir \
    "moviepy>=2.0.0.dev2" \
    imageio-ffmpeg>=0.5.1 \
    opencv-python-headless \
    Pillow \
    librosa \
    soundfile \
    decorator>=4.4.2 \
    proglog>=0.1.10

# 3. Inteligência Artificial (Transcrição e Legendas)
# Desativamos isolamento de build para que o pip use o Cython<3 instalado acima
RUN pip install --no-cache-dir --no-build-isolation \
    faster-whisper==0.10.1 \
    ctranslate2==3.24.0 \
    transformers>=4.41.2 \
    accelerate>=0.30.1 \
    scipy>=1.13.1

# 4. Ferramentas PRO (Upscale) - Opcionais, mas instaladas
# Instala basicsr e realesrgan para quando o plano PRO for ativado
RUN pip install --no-cache-dir \
    basicsr>=1.4.2 \
    facexlib>=0.2.5 \
    gfpgan>=1.3.8 \
    realesrgan>=0.3.0

# ==================== MODEL BAKING ====================
# Cria diretório de cache
RUN mkdir -p /runpod-volume/.cache/huggingface

# "Assa" o modelo Whisper na imagem para start instantâneo.
# Aponta para o diretório que será montado ou usado como cache fallback.
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', download_root='/runpod-volume/.cache/huggingface')"

# ==================== CÓDIGO ====================
COPY handler.py .

# Inicialização
CMD [ "python3", "-u", "handler.py" ]
