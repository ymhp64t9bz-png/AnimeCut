# AnimeCut Serverless - Dockerfile Ultra Simples

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1

# Workdir
WORKDIR /app

# Instalar apenas runpod
RUN pip install --no-cache-dir runpod

# Copiar handler
COPY handler.py .

# Sem healthcheck (pode estar causando o crash)
# HEALTHCHECK DISABLED

# Comando
CMD ["python3", "-u", "handler.py"]
