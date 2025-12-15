# Dockerfile para AnimeCut v12.2
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Variáveis de ambiente críticas
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/usr/local/cuda/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsndfile1 \
    sox \
    libsox-fmt-mp3 \
    python3-pip \
    python3-dev \
    python3-venv \
    python3-tk \
    fonts-dejavu-core \
    fonts-liberation \
    ttf-mscorefonts-installer \
    && rm -rf /var/lib/apt/lists/*

# Instala fontes adicionais
RUN mkdir -p /usr/share/fonts/truetype/custom/ && \
    wget -q -O /usr/share/fonts/truetype/custom/impact.ttf \
    https://github.com/google/fonts/raw/main/apache/impact/Impact.ttf && \
    fc-cache -f -v

WORKDIR /workspace

# Cria estrutura de diretórios
RUN mkdir -p /workspace/{output,models,fonts,cache,temp} /tmp/animecut

# Copia requirements primeiro para cache de camadas
COPY requirements_v12.2.txt /tmp/requirements.txt

# Instala Python dependencies com versões específicas para CUDA 12.1
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir \
    torch==2.2.1+cu121 \
    torchvision==0.17.1+cu121 \
    torchaudio==2.2.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Instala outras dependências
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Instala pacotes opcionais separadamente para evitar conflitos
RUN pip3 install --no-cache-dir \
    opencv-python-headless==4.9.0.80 \
    ultralytics==8.1.22 \
    deepfilternet==0.6.1 \
    faster-whisper==0.10.0 \
    transformers==4.40.0 \
    moviepy==1.0.3 \
    imageio[ffmpeg]==2.34.1

# Copia código fonte
COPY handler.py /workspace/handler.py
COPY runpod_handler.py /workspace/runpod_handler.py

# Permissões
RUN chmod +x /workspace/handler.py && \
    chmod +x /workspace/runpod_handler.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import sys; sys.path.insert(0, '/workspace'); from handler import health_check; print(health_check())" || exit 1

# Comando padrão
CMD ["python3", "/workspace/runpod_handler.py"]
