# ✂️ AnimeCut Serverless V3 FINAL - NUMPY SHIELDED (CONSTRAINTS MODE)
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

ENV BUILD_DATE="V3_FINAL_CONSTRAINTS_FIX" 
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME="/runpod-volume/.cache/huggingface"

# 1. Sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3-dev pkg-config ffmpeg libsndfile1 \
    libgl1 libglib2.0-0 git nano \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# 2. CRIAR ARQUIVO DE RESTRIÇÕES (A Arma Secreta)
# Isso força o pip a NUNCA instalar numpy 2.0, não importa o que as outras libs peçam.
RUN echo "numpy<2.0" > constraints.txt

# 3. Flash Attention (Via Wheel)
# Instalamos antes para garantir que não tente compilar
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/flash_attn-2.5.6+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# 4. Arsenal Python (Com Constraints)
# Adicionamos -c constraints.txt em TODOS os comandos pip
RUN pip install --no-cache-dir -c constraints.txt \
    runpod>=1.6.0 \
    boto3>=1.34.0 \
    requests \
    tqdm \
    colorama \
    "moviepy>=2.0.0.dev2" \
    imageio-ffmpeg>=0.5.1 \
    "opencv-python-headless<=4.9.0.80" \
    Pillow \
    librosa \
    soundfile \
    ultralytics \
    proglog>=0.1.10 \
    deepfilternet

# 5. Whisper & IA (Com Constraints)
RUN pip install --no-cache-dir -c constraints.txt \
    transformers \
    optimum \
    accelerate \
    scipy \
    insanely-fast-whisper

# 6. Tools Pro (Com Constraints)
RUN pip install --no-cache-dir -c constraints.txt \
    basicsr>=1.4.2 \
    facexlib>=0.2.5 \
    gfpgan>=1.3.8 \
    realesrgan>=0.3.0

# 7. Setup Final
# Baixa YOLO e verifica Numpy uma última vez (redundância de segurança)
RUN pip install "numpy==1.26.4" --force-reinstall
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

COPY handler.py .

CMD [ "python3", "-u", "handler.py" ]
