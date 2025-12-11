# ✂️ AnimeCut Serverless v8.0 - ANTI-CACHE EDITION

# Usando imagem base mais recente com CUDA 12.1 para compatibilidade com bibliotecas modernas
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# ==================== FORCE NEW LAYER ====================
# Mudamos o nome da variavel para garantir que nao existe hash igual (CACHE BUSTER)
ENV BUILD_DATE="2025-12-11_FORCE_REBUILD_V9"

# ==================== DEBUG ====================
# Instala algo inutil so para mudar o hash da imagem
RUN apt-get update && apt-get install -y tree

# ==================== VARIÁVEIS ====================
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME="/runpod-volume/.cache/huggingface"

# ==================== DEPENDÊNCIAS ====================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    fonts-dejavu-core \
    imagemagick \
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ==================== PIP ====================
COPY requirements.txt .

# Força reinstalação ignorando cache local do pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --force-reinstall -r requirements.txt

# ==================== CODIGO ====================
COPY handler.py .

# FORÇAR VERSÕES COMPATÍVEIS NO FINAL DO BUILD PARA GARANTIR
# Isso corrige o erro 'register_pytree_node' garantindo que transformers use APIs antigas do PyTorch
RUN pip uninstall -y transformers accelerate && \
    pip install "transformers==4.38.2" "accelerate==0.27.2" --no-cache-dir

CMD ["python3", "-u", "handler.py"]
