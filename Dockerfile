# ✂️ AnimeCut Serverless v8.0 - ANTI-CACHE EDITION

# Usando imagem base mais recente com CUDA 12.1 para compatibilidade com bibliotecas modernas
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# ==================== FORCE NEW LAYER ====================
# Mudamos o nome da variavel para garantir que nao existe hash igual
ENV BUILD_DATE="2025-12-11_ANTI_CACHE_V8"

# ==================== DEBUG ====================
# Instala algo inutil so para mudar o hash da imagem
RUN apt-get update && apt-get install -y tree

# ==================== VARIÁVEIS ====================
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME="/runpod-volume/.cache/huggingface"

# ==================== DEPENDÊNCIAS ====================
# Adicionando dependências explicitas do CUDNN para evitar erros de loading
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    libcudnn9-cuda-12 \

    ffmpeg \
    libsm6 \
    libxext6 \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    fonts-dejavu-core \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ==================== PIP ====================
COPY requirements.txt .

# Força reinstalação ignorando cache local do pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --force-reinstall -r requirements.txt

# ==================== CODIGO ====================
COPY handler.py .

CMD ["python3", "-u", "handler.py"]
