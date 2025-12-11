#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
✂️ AnimeCut Serverless v7.0 - Handler Completo
Todas as funcionalidades do AnimeCut local
"""

import runpod
import os
import sys
import logging
import tempfile
import requests
import gc
import time
import uuid
from pathlib import Path
from typing import List, Dict, Optional

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
print("✂️ AnimeCut Serverless v7.0 - Full Features")
print("=" * 60)

# ==================== IMPORTS CONDICIONAIS ====================
try:
    from moviepy.editor import (
        VideoFileClip, ImageClip, CompositeVideoClip,
        ColorClip, TextClip
    )
    from moviepy.video.fx.all import speedx
    import numpy as np
    MOVIEPY_AVAILABLE = True
    logger.info("✅ MoviePy disponível")
except ImportError as e:
    MOVIEPY_AVAILABLE = False
    logger.error(f"❌ MoviePy não disponível: {e}")

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
    logger.info("✅ PIL disponível")
except ImportError as e:
    PIL_AVAILABLE = False
    logger.error(f"❌ PIL não disponível: {e}")

try:
    import boto3
    from botocore.client import Config
    
    B2_KEY_ID = os.getenv("B2_KEY_ID", "")
    B2_APP_KEY = os.getenv("B2_APP_KEY", "")
    B2_ENDPOINT = os.getenv("B2_ENDPOINT", "https://s3.us-east-005.backblazeb2.com")
    B2_BUCKET = os.getenv("B2_BUCKET_NAME", "autocortes-storage")
    
    if B2_KEY_ID and B2_APP_KEY:
        s3_client = boto3.client(
            "s3",
            endpoint_url=B2_ENDPOINT,
            aws_access_key_id=B2_KEY_ID,
            aws_secret_access_key=B2_APP_KEY,
            config=Config(signature_version="s3v4")
        )
        B2_AVAILABLE = True
        logger.info("✅ Backblaze B2 configurado")
    else:
        B2_AVAILABLE = False
        logger.warning("⚠️ B2 credentials não configuradas")
except Exception as e:
    B2_AVAILABLE = False
    logger.error(f"❌ Erro ao configurar B2: {e}")

# ==================== DOWNLOAD DE VÍDEO ====================
def download_video(url: str) -> str:
    """Baixa vídeo da URL"""
    try:
        logger.info(f"📥 Baixando vídeo...")
        
        temp_file = TEMP_DIR / f"input_{uuid.uuid4().hex[:8]}.mp4"
        
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(temp_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0 and downloaded % (1024 * 1024) == 0:  # Log a cada 1MB
                    progress = (downloaded / total_size) * 100
                    logger.info(f"📥 Download: {progress:.1f}%")
        
        logger.info(f"✅ Download completo: {temp_file} ({downloaded / 1024 / 1024:.2f} MB)")
        return str(temp_file)
        
    except Exception as e:
        logger.error(f"❌ Erro no download: {e}")
        raise

# ==================== DOWNLOAD DE BACKGROUND ====================
def download_background(url: str) -> Optional[str]:
    """Baixa imagem de background"""
    try:
        if not url:
            return None
            
        logger.info(f"🖼️ Baixando background...")
        
        temp_file = TEMP_DIR / f"bg_{uuid.uuid4().hex[:8]}.png"
        
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        with open(temp_file, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"✅ Background baixado: {temp_file}")
        return str(temp_file)
        
    except Exception as e:
        logger.warning(f"⚠️ Erro ao baixar background: {e}")
        return None

# ==================== GERAÇÃO DE TÍTULO COM PIL ====================
def criar_titulo_pil(
    texto: str,
    largura: int,
    altura: int,
    duracao: float,
    font_size: int = 60,
    text_color: str = "#FFFFFF",
    stroke_color: str = "#000000",
    stroke_width: int = 3,
    pos_vertical: float = 0.15
) -> ImageClip:
    """Cria título usando PIL"""
    try:
        if not PIL_AVAILABLE:
            logger.warning("⚠️ PIL não disponível, título não será gerado")
            return None
        
        # Cria imagem transparente
        img = Image.new('RGBA', (largura, altura), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Fonte (usa DejaVu como fallback)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Quebra texto em linhas
        words = texto.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] < largura * 0.9:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Desenha texto
        y_pos = int(altura * pos_vertical)
        
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x_pos = (largura - text_width) // 2
            
            # Borda
            for adj_x in range(-stroke_width, stroke_width + 1):
                for adj_y in range(-stroke_width, stroke_width + 1):
                    draw.text((x_pos + adj_x, y_pos + adj_y), line, font=font, fill=stroke_color)
            
            # Texto
            draw.text((x_pos, y_pos), line, font=font, fill=text_color)
            y_pos += text_height + 10
        
        # Converte para clip
        img_array = np.array(img)
        return ImageClip(img_array).set_duration(duracao)
        
    except Exception as e:
        logger.error(f"❌ Erro ao criar título: {e}")
        return None

# ==================== PROCESSAMENTO DE CORTE ====================
def processar_corte(
    video_path: str,
    inicio: float,
    fim: float,
    numero_corte: int,
    config: dict
) -> str:
    """Processa um corte de vídeo com todas as funcionalidades"""
    try:
        logger.info(f"✂️ Processando corte {numero_corte}: {inicio}s - {fim}s")
        
        if not MOVIEPY_AVAILABLE:
            raise Exception("MoviePy não disponível")
        
        with VideoFileClip(video_path) as video:
            duracao_corte = min(fim - inicio, 300)
            clip = video.subclip(inicio, min(inicio + duracao_corte, video.duration))
            
            # Anti-Shadowban (Speed Ramp)
            if config.get("antiShadowban", False):
                logger.info("🎭 Aplicando Anti-Shadowban (Speed Ramp 1.05x)")
                clip = clip.fx(speedx, 1.05)
            
            # Composição
            target_w, target_h = 1080, 1920
            
            # Background
            background_path = config.get("background_path")
            if background_path and os.path.exists(background_path):
                logger.info("🖼️ Usando background customizado")
                from PIL import Image as PILImage
                fundo_img = PILImage.open(background_path).convert('RGB')
                fundo_img = fundo_img.resize((target_w, target_h), PILImage.Resampling.LANCZOS)
                fundo_array = np.array(fundo_img)
                fundo = ImageClip(fundo_array).set_duration(duracao_corte)
            else:
                logger.info("🎨 Usando background padrão")
                fundo = ColorClip(size=(target_w, target_h), color=(20, 10, 40)).set_duration(duracao_corte)
            
            # Redimensiona vídeo (fit)
            video_w, video_h = clip.size
            scale = min(target_w / video_w, target_h / video_h)
            if video_w * (target_h / video_h) < target_w:
                scale = target_w / video_w
            
            clip_resized = clip.resize(scale)
            pos_y = int((target_h - clip_resized.h) * 0.5)
            clip_resized = clip_resized.set_position(('center', pos_y))
            
            elementos = [fundo, clip_resized]
            
            # Título
            titulo = config.get("titulo")
            if titulo and config.get("generateTitles", False):
                logger.info(f"📝 Adicionando título: {titulo}")
                title_style = config.get("titleStyle", {})
                t_clip = criar_titulo_pil(
                    titulo,
                    target_w,
                    target_h,
                    duracao_corte,
                    font_size=title_style.get("fontSize", 60),
                    text_color=title_style.get("textColor", "#FFFFFF"),
                    stroke_color=title_style.get("borderColor", "#000000"),
                    stroke_width=title_style.get("borderWidth", 3),
                    pos_vertical=title_style.get("verticalPosition", 15) / 100
                )
                if t_clip:
                    elementos.append(t_clip)
            
            clip_final = CompositeVideoClip(elementos, size=(target_w, target_h))
            
            # Exportação
            filename = f"cut_{numero_corte:03d}_{uuid.uuid4().hex[:8]}.mp4"
            output_path = OUTPUT_DIR / filename
            
            logger.info(f"🎬 Renderizando corte {numero_corte}...")
            
            clip_final.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac',
                preset='fast',
                ffmpeg_params=[
                    '-profile:v', 'high',
                    '-level', '4.1',
                    '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart'
                ],
                verbose=False,
                logger=None
            )
            
            clip_final.close()
            gc.collect()
            
            logger.info(f"✅ Corte {numero_corte} concluído: {output_path}")
            return str(output_path)
            
    except Exception as e:
        logger.error(f"❌ Erro ao processar corte {numero_corte}: {e}")
        raise

# ==================== UPLOAD PARA B2 ====================
def upload_to_b2(file_path: str, object_name: str = None) -> Optional[str]:
    """Upload para Backblaze B2"""
    try:
        if not B2_AVAILABLE:
            logger.warning("⚠️ B2 não disponível, upload ignorado")
            return None
        
        if object_name is None:
            object_name = f"animecut/{os.path.basename(file_path)}"
        
        logger.info(f"📤 Uploading para B2: {object_name}")
        
        s3_client.upload_file(file_path, B2_BUCKET, object_name)
        
        # Gera URL assinada (válida por 1 hora)
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': B2_BUCKET, 'Key': object_name},
            ExpiresIn=3600
        )
        
        logger.info(f"✅ Upload completo: {object_name}")
        return url
        
    except Exception as e:
        logger.error(f"❌ Erro no upload B2: {e}")
        return None

# ==================== PROCESSAMENTO PRINCIPAL ====================
def process_video(video_path: str, config: dict) -> List[Dict]:
    """Processa vídeo completo"""
    try:
        logger.info("🎬 Iniciando processamento de vídeo")
        
        if not MOVIEPY_AVAILABLE:
            raise Exception("MoviePy não disponível")
        
        with VideoFileClip(video_path) as video:
            duration = video.duration
            logger.info(f"📊 Duração do vídeo: {duration}s")
        
        cut_type = config.get("cutType", "manual")
        cuts_data = []
        
        if cut_type == "manual":
            # Cortes manuais de 60s
            num_cuts = min(5, int(duration / 60))
            logger.info(f"✂️ Modo Manual: {num_cuts} cortes de 60s")
            
            for i in range(num_cuts):
                start = i * 60
                end = min(start + 60, duration)
                
                output_path = processar_corte(
                    video_path,
                    start,
                    end,
                    i + 1,
                    config
                )
                
                # Upload para B2
                b2_url = upload_to_b2(output_path)
                
                cuts_data.append({
                    "cut_number": i + 1,
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "local_path": output_path,
                    "b2_url": b2_url
                })
        
        else:  # auto mode
            logger.info("🤖 Modo Automático não implementado ainda")
            # TODO: Implementar detecção automática de cenas
            raise Exception("Modo automático ainda não implementado")
        
        logger.info(f"✅ Processamento completo: {len(cuts_data)} cortes gerados")
        return cuts_data
        
    except Exception as e:
        logger.error(f"❌ Erro no processamento: {e}")
        raise

# ==================== HANDLER PRINCIPAL ====================
def handler(event):
    """Handler principal do AnimeCut"""
    try:
        logger.info("🚀 AnimeCut Handler iniciado")
        logger.info(f"📦 Event: {event.get('id', 'N/A')}")
        
        input_data = event.get("input", {})
        
        # Modo de teste
        if input_data.get("mode") == "test":
            return {
                "status": "success",
                "message": "AnimeCut worker funcionando!",
                "version": "7.0",
                "features": {
                    "moviepy": MOVIEPY_AVAILABLE,
                    "pil": PIL_AVAILABLE,
                    "b2": B2_AVAILABLE
                }
            }
        
        # Validação
        video_url = input_data.get("video_url")
        if not video_url:
            return {
                "status": "error",
                "error": "video_url não fornecido"
            }
        
        # Download de vídeo
        video_path = download_video(video_url)
        
        # Download de background (opcional)
        background_url = input_data.get("background_url")
        background_path = download_background(background_url) if background_url else None
        
        # Configuração
        config = {
            "cutType": input_data.get("cutType", "manual"),
            "antiShadowban": input_data.get("antiShadowban", False),
            "generateTitles": input_data.get("generateTitles", False),
            "titulo": input_data.get("animeName", "Anime"),
            "titleStyle": input_data.get("titleStyle", {}),
            "background_path": background_path
        }
        
        # Processamento
        cuts = process_video(video_path, config)
        
        # Limpeza
        try:
            os.remove(video_path)
            if background_path:
                os.remove(background_path)
        except:
            pass
        
        # Resultado
        result = {
            "status": "success",
            "message": f"{len(cuts)} cortes gerados com sucesso",
            "cuts": cuts,
            "config": config
        }
        
        logger.info(f"✅ Job completo: {len(cuts)} cortes")
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
    logger.info(f"📊 MoviePy: {MOVIEPY_AVAILABLE}")
    logger.info(f"📊 PIL: {PIL_AVAILABLE}")
    logger.info(f"📊 B2: {B2_AVAILABLE}")
    
    runpod.serverless.start({"handler": handler})
    logger.info("✅ Worker iniciado!")
