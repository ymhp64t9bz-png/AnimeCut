# Use a imagem base do RunPod com CUDA 12.1
FROM runpod/base:0.4.0-cuda12.1.1

# Cache Busting
ENV BUILD_DATE="V12.4_FIXED_SYNTAX"

# Configura variáveis de ambiente para GPU
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/usr/local/cuda/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
ENV CUDA_HOME="/usr/local/cuda"

# Atualiza sistema e instala dependências do sistema
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libsndfile1 \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    pkg-config \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Cria diretório de trabalho
WORKDIR /workspace

# Copia requirements primeiro para cache
COPY requirements.txt .

# Instala Python dependencies em etapas para melhor cache
RUN pip install --upgrade pip setuptools wheel

# 1. Dependências básicas do sistema
RUN pip install --no-cache-dir \
    runpod>=1.6.0 \
    boto3>=1.34.0 \
    requests>=2.31.0 \
    tqdm>=4.66.0 \
    colorama>=0.4.6 \
    protobuf>=4.25.0 \
    scipy>=1.11.0 \
    sentencepiece>=0.1.99

# 2. Dependências de processamento de vídeo e imagem
RUN pip install --no-cache-dir \
    "moviepy==1.0.3" \
    imageio-ffmpeg>=0.5.1 \
    "opencv-python-headless<=4.9.0.80" \
    Pillow>=10.0.0 \
    proglog>=0.1.10 \
    imageio>=2.31.0

# 3. Dependências de áudio
RUN pip install --no-cache-dir \
    librosa>=0.10.0 \
    soundfile>=0.12.0 \
    pydub>=0.25.1

# 4. PyTorch com CUDA 12.1 (versão específica para compatibilidade)
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 5. Dependências de IA e ML
RUN pip install --no-cache-dir \
    transformers>=4.36.0 \
    optimum>=1.15.0 \
    accelerate>=0.25.0 \
    einops>=0.7.0 \
    safetensors>=0.4.0 \
    peft>=0.7.0

# 6. Whisper e transcrição (instala separadamente para evitar conflitos)
RUN pip install --no-cache-dir \
    faster-whisper>=0.10.0 \
    openai-whisper>=20231117 \
    insanely-fast-whisper>=0.0.5

# 7. Visão computacional
RUN pip install --no-cache-dir \
    ultralytics>=8.0.0 \
    numpy>=1.24.0 \
    pandas>=2.0.0

# 8. Dependências opcionais e utilitários
RUN pip install --no-cache-dir \
    psutil>=5.9.0 \
    humanize>=4.8.0 \
    py-cpuinfo>=9.0.0 \
    nvidia-ml-py3>=7.352.0

# Cria diretórios necessários
RUN mkdir -p /workspace/output /workspace/models /workspace/fonts /workspace/cache /workspace/temp /tmp/animecut

# Copia fonte do projeto
COPY handler.py /workspace/handler.py
COPY *.txt *.py /workspace/

# Configura permissões
RUN chmod +x /workspace/handler.py

# Baixa modelo Whisper pré-treinado para cache (Linha única segura)
RUN python3 -c "from faster_whisper import WhisperModel; import os; os.makedirs('/workspace/models', exist_ok=True); print('Downloading model...'); model = WhisperModel('tiny', device='cpu', compute_type='float32', download_root='/workspace/models')"

# Baixa fontes padrão
RUN cd /workspace/fonts && \
    wget -q https://github.com/google/fonts/raw/main/ofl/oswald/Oswald-Bold.ttf -O oswald.ttf && \
    wget -q https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Bold.ttf -O roboto.ttf

# Verifica instalações (Linha única segura)
RUN python3 -c "import sys; import torch; import faster_whisper; import moviepy; import cv2; print('Check OK: All packages imported successfully')"

# Limpa cache do pip
RUN pip cache purge

# Define comando de execução
CMD ["python", "-u", "/workspace/handler.py"]
