#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
✂️ AnimeCut Serverless v7.0 - Handler Funcional
Versão estável sem HEALTHCHECK
"""

import runpod
import os
import sys
import logging
import tempfile
import requests
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

print("=" * 60)
print("✂️ AnimeCut Serverless v7.0")
print("=" * 60)

# ==================== VERIFICAR DEPENDÊNCIAS ====================
def check_dependencies():
    """Verifica dependências disponíveis"""
    deps = {}
    
    try:
        from moviepy.editor import VideoFileClip
        deps['moviepy'] = True
        logger.info("✅ MoviePy disponível")
    except ImportError as e:
        deps['moviepy'] = False
        logger.warning(f"⚠️ MoviePy não disponível: {e}")
    
    try:
        import boto3
        deps['boto3'] = True
        logger.info("✅ Boto3 disponível")
    except ImportError as e:
        deps['boto3'] = False
        logger.warning(f"⚠️ Boto3 não disponível: {e}")
    
    try:
        from PIL import Image
        deps['pil'] = True
        logger.info("✅ PIL disponível")
    except ImportError as e:
        deps['pil'] = False
        logger.warning(f"⚠️ PIL não disponível: {e}")
    
    return deps

# ==================== DOWNLOAD DE VÍDEO ====================
def download_video(url: str) -> str:
    """Baixa vídeo da URL"""
    try:
        logger.info(f"📥 Baixando vídeo: {url[:100]}...")
        
        temp_file = TEMP_DIR / f"input_{os.urandom(8).hex()}.mp4"
        
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        with open(temp_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"✅ Download completo: {temp_file}")
        return str(temp_file)
        
    except Exception as e:
        logger.error(f"❌ Erro no download: {e}")
        raise

# ==================== PROCESSAMENTO DE VÍDEO ====================
def process_video_simple(video_path: str, cut_type: str = "manual") -> list:
    """Processa vídeo de forma simples"""
    try:
        from moviepy.editor import VideoFileClip
        
        logger.info(f"🎬 Processando vídeo: {video_path}")
        
        video = VideoFileClip(video_path)
        duration = video.duration
        
        logger.info(f"📊 Duração: {duration}s")
        
        # Cortes simples de 60s
        cuts = []
        num_cuts = min(5, int(duration / 60))
        
        for i in range(num_cuts):
            start = i * 60
            end = min(start + 60, duration)
            
            output_file = OUTPUT_DIR / f"cut_{i}_{os.urandom(4).hex()}.mp4"
            
            logger.info(f"✂️ Corte {i+1}/{num_cuts}: {start}s - {end}s")
            
            subclip = video.subclip(start, end)
            subclip.write_videofile(
                str(output_file),
                codec='libx264',
                audio_codec='aac',
                verbose=False,
                logger=None
            )
            
            cuts.append(str(output_file))
            logger.info(f"✅ Corte {i+1} salvo: {output_file}")
        
        video.close()
        return cuts
        
    except Exception as e:
        logger.error(f"❌ Erro no processamento: {e}")
        raise

# ==================== HANDLER PRINCIPAL ====================
def handler(event):
    """Handler principal do AnimeCut"""
    try:
        logger.info("🚀 AnimeCut Handler iniciado")
        logger.info(f"📦 Event: {event}")
        
        input_data = event.get("input", {})
        
        # Modo de teste
        if input_data.get("mode") == "test":
            deps = check_dependencies()
            return {
                "status": "success",
                "message": "AnimeCut worker funcionando!",
                "dependencies": deps,
                "version": "7.0"
            }
        
        # Processamento de vídeo
        video_url = input_data.get("video_url")
        if not video_url:
            return {
                "status": "error",
                "error": "video_url não fornecido"
            }
        
        # Download
        video_path = download_video(video_url)
        
        # Processamento
        cut_type = input_data.get("cutType", "manual")
        cuts = process_video_simple(video_path, cut_type)
        
        # Resultado
        result = {
            "status": "success",
            "message": f"{len(cuts)} cortes gerados",
            "cuts": cuts,
            "video_processed": video_path
        }
        
        logger.info(f"✅ Processamento completo: {len(cuts)} cortes")
        return result
        
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
    
    # Verifica dependências
    deps = check_dependencies()
    logger.info(f"📊 Dependências: {deps}")
    
    # Inicia worker
    runpod.serverless.start({"handler": handler})
    logger.info("✅ Worker iniciado!")
