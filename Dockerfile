# ✂️ AnimeCut Serverless v9.0 - MODERN STACK EDITION
# Base Image com CUDA 12.1.1 (Perfeita para PyTorch 2.3+)
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Cache Buster para garantir build limpo
ENV BUILD_DATE="2025-12-12_MODERN_STACK_V1"

# ==================== SISTEMA OPERACIONAL ====================
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME="/runpod-volume/.cache/huggingface"

# Instalação de dependências de sistema (apenas runtime, nada de compilação pesada)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    fonts-dejavu-core \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ==================== PYTHON DEPENDENCIES ====================
COPY requirements.txt .

# Instalação limpa e direta.
# --no-cache-dir garante que baixamos os wheels novos
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==================== CORREÇÃO CRÍTICA DE PATH ====================
# Isso adiciona as bibliotecas NVIDIA instaladas via pip (cuDNN 9, Cublas) ao PATH do sistema.
# Sem isso, o CTranslate2 não encontra o cuDNN 9 e falha.
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH}"

# ==================== CÓDIGO FONTE ====================
COPY handler.py .

# Comando de inicialização
CMD ["python3", "-u", "handler.py"]
