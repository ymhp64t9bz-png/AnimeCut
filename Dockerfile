# ✂️ AnimeCut Serverless v6.0 - Dockerfile
# Imagem otimizada para RunPod Serverless com GPU

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# ==================== METADADOS ====================
LABEL maintainer="AutoCortes Team"
LABEL description="AnimeCut Serverless v6.0 - AI Video Processing with Qwen 2.5 + Whisper"
LABEL version="6.0"

# ==================== VARIÁVEIS DE AMBIENTE ====================
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Configurações de memória
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_LAUNCH_BLOCKING=0

# Diretórios de trabalho
ENV APP_DIR=/app
ENV MODELS_DIR=/app/models
ENV TEMP_DIR=/tmp/animecut
ENV OUTPUT_DIR=/tmp/animecut/output

# ==================== INSTALAÇÃO DE DEPENDÊNCIAS DO SISTEMA ====================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # FFmpeg (essencial para MoviePy)
    ffmpeg \
    # ImageMagick (para TextClip)
    imagemagick \
    # OpenCV dependencies
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    # Audio processing
    libsndfile1 \
    # Networking
    wget \
    curl \
    git \
    # Build tools para llama-cpp-python
    build-essential \
    cmake \
    # Fonts (para PIL)
    fonts-dejavu-core \
    fonts-liberation \
    fonts-freefont-ttf \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Configura política do ImageMagick
RUN sed -i 's/rights="none" pattern="PDF"/rights="read|write" pattern="PDF"/' /etc/ImageMagick-6/policy.xml || true

# ==================== CRIAR DIRETÓRIOS ====================
WORKDIR $APP_DIR

RUN mkdir -p \
    $MODELS_DIR \
    $TEMP_DIR \
    $OUTPUT_DIR \
    /app/src \
    /app/src/core \
    /app/src/core/ai_services

# ==================== COPIAR REQUIREMENTS ====================
COPY requirements.txt .

# ==================== INSTALAR DEPENDÊNCIAS PYTHON ====================
# Atualiza pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Instala llama-cpp-python com CUDA PRIMEIRO (separado)
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir --force-reinstall --no-deps llama-cpp-python>=0.2.27

# Instala dependências do llama-cpp-python
RUN pip install --no-cache-dir typing-extensions>=4.5.0 numpy>=1.20.0 diskcache>=5.6.1

# Instala demais dependências
RUN pip install --no-cache-dir -r requirements.txt

# Cleanup
RUN pip cache purge

# ==================== COPIAR CÓDIGO DA APLICAÇÃO ====================
# Cria estrutura de diretórios necessária
RUN mkdir -p /app/src/core/ai_services /app/models /app/utils

# Copia handler.py (deve estar no diretório raiz do repositório)
COPY handler.py /app/handler.py

# Garante que todos os __init__.py existam
RUN touch /app/__init__.py && \
    touch /app/src/__init__.py && \
    touch /app/src/core/__init__.py && \
    touch /app/src/core/ai_services/__init__.py

# ==================== HEALTHCHECK ====================
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import torch; print('GPU:', torch.cuda.is_available())" || exit 1

# ==================== EXPOR PORTA ====================
EXPOSE 8000

# ==================== COMANDO DE INICIALIZAÇÃO ====================
CMD ["python3", "-u", "handler.py"]
