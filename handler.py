# -*- coding: utf-8 -*-
"""
✂️ AnimeCut Serverless v6.0 - Handler Simplificado
Versão funcional sem dependências de módulos externos
"""

import runpod
import os
import sys
import logging
from pathlib import Path

# ==================== CONFIGURAÇÃO ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AnimeCut")

# Diretórios
TEMP_DIR = Path("/tmp/animecut")
OUTPUT_DIR = Path("/tmp/animecut/output")
TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== VERIFICAR DEPENDÊNCIAS ====================
def check_dependencies():
    """Verifica quais dependências estão disponíveis"""
    deps = {}
    
    try:
        import moviepy
        deps['moviepy'] = True
        logger.info("✅ MoviePy disponível")
    except ImportError as e:
        deps['moviepy'] = False
        logger.error(f"❌ MoviePy não disponível: {e}")
    
    try:
        import whisper
        deps['whisper'] = True
        logger.info("✅ Whisper disponível")
    except ImportError as e:
        deps['whisper'] = False
        logger.error(f"❌ Whisper não disponível: {e}")
    
    try:
        import boto3
        deps['boto3'] = True
        logger.info("✅ Boto3 disponível")
    except ImportError as e:
        deps['boto3'] = False
        logger.error(f"❌ Boto3 não disponível: {e}")
    
    try:
        from PIL import Image
        deps['pil'] = True
        logger.info("✅ PIL disponível")
    except ImportError as e:
        deps['pil'] = False
        logger.error(f"❌ PIL não disponível: {e}")
    
    return deps

# ==================== HANDLER ====================
def handler(event):
    """
    Handler principal do AnimeCut Serverless
    
    Payload esperado:
    {
        "input": {
            "video_url": "https://...",
            "mode": "test"  # ou "auto" ou "manual"
        }
    }
    """
    try:
        logger.info("🚀 AnimeCut Serverless v6.0 iniciado")
        logger.info(f"📦 Event recebido: {event}")
        
        # Verifica dependências
        deps = check_dependencies()
        
        # Extrai input
        input_data = event.get("input", {})
        mode = input_data.get("mode", "test")
        
        # Modo de teste
        if mode == "test":
            return {
                "status": "success",
                "message": "AnimeCut worker está funcionando!",
                "dependencies": deps,
                "python_version": sys.version,
                "temp_dir": str(TEMP_DIR),
                "output_dir": str(OUTPUT_DIR),
                "env_vars": {
                    "MODELS_PATH": os.getenv("MODELS_PATH", "not set"),
                    "B2_BUCKET_NAME": os.getenv("B2_BUCKET_NAME", "not set")
                }
            }
        
        # Modo automático (TODO: implementar)
        elif mode == "auto":
            return {
                "status": "error",
                "message": "Modo automático ainda não implementado nesta versão"
            }
        
        # Modo manual (TODO: implementar)
        elif mode == "manual":
            return {
                "status": "error",
                "message": "Modo manual ainda não implementado nesta versão"
            }
        
        else:
            return {
                "status": "error",
                "message": f"Modo '{mode}' não reconhecido. Use: test, auto ou manual"
            }
    
    except Exception as e:
        logger.error(f"❌ Erro no handler: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "type": type(e).__name__
        }

# ==================== INICIALIZAÇÃO ====================
if __name__ == "__main__":
    logger.info("🎬 Iniciando AnimeCut Serverless Worker...")
    
    # Verifica dependências na inicialização
    deps = check_dependencies()
    
    # Inicia o worker RunPod
    runpod.serverless.start({"handler": handler})
