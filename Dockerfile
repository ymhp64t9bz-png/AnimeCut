# ✂️ AnimeCut Serverless v6.0 - Dockerfile
# Imagem otimizada para RunPod Serverless

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# ==================== METADADOS ====================
LABEL maintainer="AutoCortes Team"
LABEL description="AnimeCut Serverless v6.0"
LABEL version="6.0"

# ==================== VARIÁVEIS DE AMBIENTE ====================
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# ==================== INSTALAÇÃO DE DEPENDÊNCIAS DO SISTEMA ====================
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    imagemagick \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    wget \
    curl \
    git \
    fonts-dejavu-core \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Configura ImageMagick
RUN sed -i 's/rights="none" pattern="PDF"/rights="read|write" pattern="PDF"/' /etc/ImageMagick-6/policy.xml || true

# ==================== WORKDIR ====================
WORKDIR /app

# ==================== CRIAR DIRETÓRIOS ====================
RUN mkdir -p /tmp/animecut /tmp/animecut/output

# ==================== INSTALAR DEPENDÊNCIAS PYTHON ====================
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# ==================== COPIAR CÓDIGO ====================
COPY handler.py .

# ==================== HEALTHCHECK ====================
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import runpod; print('OK')" || exit 1

# ==================== COMANDO ====================
CMD ["python3", "-u", "handler.py"]
