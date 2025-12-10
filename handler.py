#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
✂️ AnimeCut Cloud - Handler Serverless COMPLETO v6.0
=====================================================
Sistema completo de cortes automáticos de anime para RunPod Serverless.

FUNCIONALIDADES IMPLEMENTADAS:
✅ Detecção inteligente de cenas com IA (Qwen 2.5)
✅ Geração automática de títulos virais
✅ Transcrição de áudio (Whisper)
✅ Processamento de vídeo com templates
✅ Anti-Shadowban (speed ramp + noise)
✅ Suporte a cortes manuais e automáticos
✅ Upload para Backblaze B2
✅ Renderização com GPU (NVENC) ou CPU (fallback)
✅ Geração de títulos sobrepostos no vídeo
"""

import runpod
import os
import sys
import gc
import torch
import tempfile
import json
import logging
import time
import uuid
import requests
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

# ==================== CONFIGURAÇÃO DE LOGGING ====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AnimeCut-Serverless")

# ==================== PATHS ====================

# Adiciona src ao path
SRC_PATH = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_PATH))

# Diretórios de trabalho
TEMP_DIR = Path("/tmp/animecut")
OUTPUT_DIR = Path("/tmp/animecut/output")
TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== IMPORTS DO SISTEMA ====================

try:
    from core.ai_services.local_ai_service import (
        transcribe_audio_batch,
        generate_viral_title_batch,
        analyze_viral_segments_deepseek,
        manually_unload_whisper,
        manually_unload_llama,
        load_whisper_model,
        load_llama_model
    )
    AI_AVAILABLE = True
    logger.info("✅ Módulos de IA carregados com sucesso")
except ImportError as e:
    AI_AVAILABLE = False
    logger.error(f"❌ Erro ao importar módulos de IA: {e}")

try:
    from moviepy.editor import (
        VideoFileClip, 
        ImageClip, 
        CompositeVideoClip, 
        ColorClip
    )
    from moviepy.video.fx.all import speedx
    MOVIEPY_AVAILABLE = True
    logger.info("✅ MoviePy carregado com sucesso")
except ImportError as e:
    MOVIEPY_AVAILABLE = False
    logger.error(f"❌ Erro ao importar MoviePy: {e}")

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
    logger.info("✅ PIL carregado com sucesso")
except ImportError as e:
    PIL_AVAILABLE = False
    logger.error(f"❌ Erro ao importar PIL: {e}")

# ==================== BACKBLAZE B2 STORAGE ====================

try:
    import boto3
    from botocore.client import Config
    
    B2_KEY_ID = os.getenv("B2_KEY_ID", "68702c2cbfc6")
    B2_APP_KEY = os.getenv("B2_APP_KEY", "00506496bc1450b6722b672d9a43d00605f17eadd7")
    B2_ENDPOINT = os.getenv("B2_ENDPOINT", "https://s3.us-east-005.backblazeb2.com")
    B2_BUCKET = os.getenv("B2_BUCKET_NAME", "autocortes-storage")
    
    s3_client = boto3.client(
        "s3",
        endpoint_url=B2_ENDPOINT,
        aws_access_key_id=B2_KEY_ID,
        aws_secret_access_key=B2_APP_KEY,
        config=Config(signature_version="s3v4")
    )
    
    B2_AVAILABLE = True
    logger.info("✅ Backblaze B2 configurado")
except Exception as e:
    B2_AVAILABLE = False
    logger.error(f"❌ Erro ao configurar B2: {e}")

# ==================== VARIÁVEIS GLOBAIS ====================

MODELS_LOADED = False

# ==================== FUNÇÕES AUXILIARES ====================

def clean_memory():
    """Limpa memória RAM e VRAM"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.info("🧹 Memória limpa")

def download_file(url: str, destination: Path) -> bool:
    """Download de arquivo com retry"""
    try:
        logger.info(f"⬇️  Baixando: {url}")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"✅ Download completo: {destination}")
        return True
    except Exception as e:
        logger.error(f"❌ Erro no download: {e}")
        return False

def upload_to_b2(file_path: Path, remote_path: str) -> Optional[str]:
    """Upload para Backblaze B2"""
    if not B2_AVAILABLE:
        logger.error("❌ B2 não disponível")
        return None
    
    try:
        logger.info(f"📤 Uploading para B2: {remote_path}")
        
        with open(file_path, 'rb') as f:
            s3_client.upload_fileobj(f, B2_BUCKET, remote_path)
        
        # Gera URL assinada (válida por 7 dias)
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': B2_BUCKET, 'Key': remote_path},
            ExpiresIn=604800  # 7 dias
        )
        
        logger.info(f"✅ Upload completo: {url}")
        return url
    except Exception as e:
        logger.error(f"❌ Erro no upload B2: {e}")
        return None

# ==================== GERAÇÃO DE TÍTULOS (PIL) ====================

def criar_titulo_pil(
    texto: str,
    largura: int,
    altura: int,
    duracao: float,
    font_size: int = 70,
    text_color: str = "#FFFFFF",
    stroke_color: str = "#000000",
    stroke_width: int = 6,
    pos_vertical: float = 0.15
) -> Optional[ImageClip]:
    """Cria título sobreposto usando PIL"""
    
    if not PIL_AVAILABLE:
        logger.error("❌ PIL não disponível")
        return None
    
    try:
        # Converte cores hex para RGB
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        text_rgb = hex_to_rgb(text_color)
        stroke_rgb = hex_to_rgb(stroke_color)
        
        # Cria canvas transparente
        canvas_h = int(altura * 0.3)
        img = Image.new('RGBA', (largura, canvas_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Carrega fonte (fallback para padrão)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
        
        # Quebra texto em linhas (máx 2)
        palavras = texto.split()
        linhas = []
        linha_atual = []
        
        for palavra in palavras:
            linha_atual.append(palavra)
            bbox = draw.textbbox((0, 0), " ".join(linha_atual), font=font)
            w = bbox[2] - bbox[0]
            
            if w > largura * 0.9:
                linha_atual.pop()
                if linha_atual:
                    linhas.append(" ".join(linha_atual))
                linha_atual = [palavra]
                
                if len(linhas) >= 2:
                    break
        
        if linha_atual and len(linhas) < 2:
            linhas.append(" ".join(linha_atual))
        
        linhas = linhas[:2]
        
        # Desenha texto com contorno
        y = 20
        for linha in linhas:
            # Contorno
            for off_x in range(-stroke_width, stroke_width + 1):
                for off_y in range(-stroke_width, stroke_width + 1):
                    if off_x != 0 or off_y != 0:
                        draw.text(
                            (largura / 2 + off_x, y + off_y),
                            linha,
                            font=font,
                            fill=stroke_rgb,
                            anchor="mt"
                        )
            
            # Texto principal
            draw.text(
                (largura / 2, y),
                linha,
                font=font,
                fill=text_rgb,
                anchor="mt"
            )
            
            bbox = draw.textbbox((0, 0), linha, font=font)
            h = bbox[3] - bbox[1]
            y += h + 15
        
        # Converte para MoviePy
        numpy_img = np.array(img)
        clip = ImageClip(numpy_img).set_duration(duracao)
        
        # Posiciona
        pos_y = int(altura * pos_vertical)
        clip = clip.set_position(('center', pos_y))
        
        return clip
        
    except Exception as e:
        logger.error(f"❌ Erro ao criar título: {e}")
        return None

# ==================== PROCESSAMENTO DE VÍDEO ====================

def processar_corte(
    video_path: Path,
    inicio: float,
    fim: float,
    numero_corte: int,
    config: Dict,
    anime_name: str
) -> Optional[Path]:
    """
    Processa um único corte de vídeo
    
    Args:
        video_path: Caminho do vídeo original
        inicio: Tempo de início (segundos)
        fim: Tempo de fim (segundos)
        numero_corte: Número do corte
        config: Configurações (template, cores, etc)
        anime_name: Nome do anime
    
    Returns:
        Path do arquivo gerado ou None
    """
    
    if not MOVIEPY_AVAILABLE:
        logger.error("❌ MoviePy não disponível")
        return None
    
    try:
        logger.info(f"🎬 Processando corte {numero_corte}: {inicio:.1f}s - {fim:.1f}s")
        
        # Carrega vídeo
        with VideoFileClip(str(video_path)) as video:
            # Extrai segmento
            duracao_corte = min(fim - inicio, 300)
            clip = video.subclip(inicio, min(inicio + duracao_corte, video.duration))
            
            # Anti-Shadowban
            if config.get("anti_shadowban", True):
                logger.info("🛡️ Aplicando Anti-Shadowban")
                clip = clip.fx(speedx, 1.05)
            
            # Gera título com IA
            titulo_viral = None
            if config.get("usar_ia", True) and AI_AVAILABLE:
                logger.info("🤖 Gerando título viral...")
                try:
                    # Extrai áudio
                    fd, temp_audio = tempfile.mkstemp(suffix=".wav")
                    os.close(fd)
                    clip.audio.write_audiofile(temp_audio, logger=None)
                    
                    # Transcreve
                    dialogo_res = transcribe_audio_batch(temp_audio)
                    dialogo_text = dialogo_res['text'] if isinstance(dialogo_res, dict) else dialogo_res
                    
                    # Gera título
                    if dialogo_text:
                        titulo_viral = generate_viral_title_batch(anime_name, dialogo_text)
                        logger.info(f"✅ Título gerado: {titulo_viral}")
                    
                    # Remove áudio temporário
                    if os.path.exists(temp_audio):
                        os.remove(temp_audio)
                        
                except Exception as e:
                    logger.error(f"❌ Erro ao gerar título: {e}")
            
            # Dimensões alvo (9:16 vertical)
            target_w, target_h = 1080, 1920
            
            # Cria fundo
            template_path = config.get("template_path")
            if template_path and os.path.exists(template_path):
                logger.info("🖼️ Usando template personalizado")
                fundo_img = Image.open(template_path).convert('RGB')
                fundo_img = fundo_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                fundo_array = np.array(fundo_img)
                fundo = ImageClip(fundo_array).set_duration(duracao_corte)
            else:
                logger.info("🎨 Usando fundo padrão")
                fundo = ColorClip(size=(target_w, target_h), color=(20, 10, 40)).set_duration(duracao_corte)
            
            # Redimensiona vídeo
            video_w, video_h = clip.size
            scale = min(target_w / video_w, target_h / video_h)
            if video_w * (target_h / video_h) < target_w:
                scale = target_w / video_w
            
            clip_resized = clip.resize(scale)
            pos_y = int((target_h - clip_resized.h) * 0.5)
            clip_resized = clip_resized.set_position(('center', pos_y))
            
            # Compõe elementos
            elementos = [fundo, clip_resized]
            
            # Adiciona título
            if titulo_viral:
                logger.info("📝 Adicionando título ao vídeo")
                t_clip = criar_titulo_pil(
                    titulo_viral,
                    target_w,
                    target_h,
                    duracao_corte,
                    font_size=config.get("font_size", 70),
                    text_color=config.get("text_color", "#FFFFFF"),
                    stroke_color=config.get("stroke_color", "#000000"),
                    stroke_width=config.get("stroke_width", 6),
                    pos_vertical=config.get("pos_vertical", 0.15)
                )
                if t_clip:
                    elementos.append(t_clip)
            
            # Composição final
            clip_final = CompositeVideoClip(elementos, size=(target_w, target_h))
            
            # Exporta
            filename = f"animecut_{numero_corte:03d}_{uuid.uuid4().hex[:8]}.mp4"
            output_path = OUTPUT_DIR / filename
            
            logger.info("🎥 Renderizando vídeo...")
            
            # Parâmetros de renderização
            ffmpeg_params = [
                '-preset', 'fast',
                '-profile:v', 'high',
                '-level', '4.1',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-rc:v', 'vbr',
                '-cq:v', '23',
                '-b:v', '5M',
                '-maxrate:v', '8M',
                '-bufsize:v', '10M',
                '-b:a', '192k'
            ]
            
            if config.get("anti_shadowban", True):
                ffmpeg_params.extend(['-vf', 'noise=alls=1:allf=t,eq=contrast=1.02'])
            
            # Tenta NVENC (GPU)
            try:
                if torch.cuda.is_available():
                    logger.info("🚀 Renderizando com GPU (NVENC)")
                    clip_final.write_videofile(
                        str(output_path),
                        codec='h264_nvenc',
                        audio_codec='aac',
                        audio_bitrate='192k',
                        ffmpeg_params=ffmpeg_params,
                        threads=4,
                        logger=None
                    )
                else:
                    raise Exception("GPU não disponível")
            except Exception as nvenc_error:
                # Fallback para CPU
                logger.warning(f"⚠️ NVENC falhou: {nvenc_error}")
                logger.info("🔄 Renderizando com CPU (libx264)")
                
                ffmpeg_params_cpu = [
                    '-preset', 'ultrafast',
                    '-profile:v', 'high',
                    '-level', '4.1',
                    '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart',
                    '-crf', '23',
                    '-b:a', '192k'
                ]
                
                if config.get("anti_shadowban", True):
                    ffmpeg_params_cpu.extend(['-vf', 'noise=alls=1:allf=t,eq=contrast=1.02'])
                
                clip_final.write_videofile(
                    str(output_path),
                    codec='libx264',
                    audio_codec='aac',
                    audio_bitrate='192k',
                    ffmpeg_params=ffmpeg_params_cpu,
                    threads=4,
                    logger=None
                )
            
            logger.info(f"✅ Corte {numero_corte} renderizado: {output_path}")
            
            # Limpa memória
            clean_memory()
            
            return output_path
            
    except Exception as e:
        logger.error(f"❌ Erro ao processar corte {numero_corte}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        clean_memory()
        return None

# ==================== MODO AUTOMÁTICO (IA) ====================

def process_auto_mode(video_path: Path, anime_name: str, config: Dict) -> Dict:
    """
    Modo automático com detecção de cenas por IA
    """
    logger.info("🤖 Iniciando modo AUTOMÁTICO (IA)")
    
    try:
        # Extrai áudio completo
        logger.info("🎵 Extraindo áudio do vídeo...")
        with VideoFileClip(str(video_path)) as video:
            duration_total = video.duration
            fd, temp_audio = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            video.audio.write_audiofile(temp_audio, logger=None)
        
        logger.info(f"✅ Áudio extraído: {duration_total:.1f}s")
        
        # Transcreve áudio
        logger.info("🎤 Transcrevendo áudio...")
        transcript_res = transcribe_audio_batch(temp_audio)
        transcript_text = transcript_res['text'] if isinstance(transcript_res, dict) else transcript_res
        
        logger.info(f"✅ Transcrição completa: {len(transcript_text)} caracteres")
        
        # Analisa segmentos virais
        logger.info("🔍 Analisando segmentos virais com Qwen 2.5...")
        segments = analyze_viral_segments_deepseek(transcript_text, duration_total)
        
        logger.info(f"✅ {len(segments)} segmentos identificados")
        
        # Remove áudio temporário
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        
        # Processa cada segmento
        results = []
        for i, seg in enumerate(segments, 1):
            logger.info(f"📹 Processando segmento {i}/{len(segments)}")
            
            output_path = processar_corte(
                video_path,
                seg['start'],
                seg['end'],
                i,
                config,
                anime_name
            )
            
            if output_path:
                # Upload para B2
                remote_path = f"animecut/{output_path.name}"
                url = upload_to_b2(output_path, remote_path)
                
                if url:
                    results.append({
                        "name": output_path.name,
                        "url": url,
                        "start": seg['start'],
                        "end": seg['end'],
                        "duration": seg['end'] - seg['start']
                    })
                
                # Remove arquivo local
                output_path.unlink()
            
            # Limpa memória entre cortes
            clean_memory()
        
        logger.info(f"🎉 Modo automático concluído: {len(results)} vídeos gerados")
        
        return {
            "status": "completed",
            "mode": "auto",
            "files": results,
            "total_files": len(results),
            "transcript_length": len(transcript_text)
        }
        
    except Exception as e:
        logger.error(f"❌ Erro no modo automático: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# ==================== MODO MANUAL ====================

def process_manual_mode(video_path: Path, anime_name: str, config: Dict) -> Dict:
    """
    Modo manual com cortes distribuídos uniformemente
    """
    logger.info("✂️ Iniciando modo MANUAL")
    
    try:
        num_cuts = config.get("num_cuts", 5)
        cut_duration = config.get("cut_duration", 60)
        
        # Obtém duração do vídeo
        with VideoFileClip(str(video_path)) as video:
            duration_total = video.duration
        
        logger.info(f"📊 Vídeo: {duration_total:.1f}s | Cortes: {num_cuts} x {cut_duration}s")
        
        # Calcula intervalos
        if num_cuts == 1:
            interval = 0
        else:
            available_space = duration_total - cut_duration
            interval = available_space / (num_cuts - 1) if num_cuts > 1 else 0
        
        # Gera cortes
        segments = []
        for i in range(num_cuts):
            start = i * interval
            end = min(start + cut_duration, duration_total)
            
            if end - start >= 5:  # Mínimo 5 segundos
                segments.append({'start': start, 'end': end})
        
        logger.info(f"✅ {len(segments)} cortes planejados")
        
        # Processa cada corte
        results = []
        for i, seg in enumerate(segments, 1):
            logger.info(f"📹 Processando corte {i}/{len(segments)}")
            
            output_path = processar_corte(
                video_path,
                seg['start'],
                seg['end'],
                i,
                config,
                anime_name
            )
            
            if output_path:
                # Upload para B2
                remote_path = f"animecut/{output_path.name}"
                url = upload_to_b2(output_path, remote_path)
                
                if url:
                    results.append({
                        "name": output_path.name,
                        "url": url,
                        "start": seg['start'],
                        "end": seg['end'],
                        "duration": seg['end'] - seg['start']
                    })
                
                # Remove arquivo local
                output_path.unlink()
            
            # Limpa memória entre cortes
            clean_memory()
        
        logger.info(f"🎉 Modo manual concluído: {len(results)} vídeos gerados")
        
        return {
            "status": "completed",
            "mode": "manual",
            "files": results,
            "total_files": len(results)
        }
        
    except Exception as e:
        logger.error(f"❌ Erro no modo manual: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# ==================== HANDLER PRINCIPAL ====================

async def handler(job):
    """
    Handler principal do RunPod Serverless
    
    Input esperado:
    {
        "input": {
            "video_url": "https://..." ou "path/to/file.mp4" (B2),
            "anime_name": "Nome do Anime",
            "mode": "auto" | "manual",
            "config": {
                "num_cuts": 5,           # Apenas para modo manual
                "cut_duration": 60,      # Apenas para modo manual
                "font_size": 70,
                "text_color": "#FFFFFF",
                "stroke_color": "#000000",
                "stroke_width": 6,
                "pos_vertical": 0.15,
                "anti_shadowban": true,
                "usar_ia": true,
                "template_url": "https://..." (opcional)
            }
        }
    }
    """
    
    try:
        input_data = job.get("input", {})
        
        # Parâmetros
        video_input = input_data.get("video_url") or input_data.get("video_path")
        anime_name = input_data.get("anime_name", "Anime")
        mode = input_data.get("mode", "auto")
        config = input_data.get("config", {})
        
        logger.info("=" * 60)
        logger.info("🚀 AnimeCut Serverless v6.0 - Iniciando")
        logger.info(f"📺 Anime: {anime_name}")
        logger.info(f"🎯 Modo: {mode.upper()}")
        logger.info("=" * 60)
        
        if not video_input:
            return {"status": "error", "error": "No video_url or video_path provided"}
        
        # Download do vídeo
        video_path = TEMP_DIR / f"source_{uuid.uuid4().hex}.mp4"
        
        if video_input.startswith("http"):
            # URL direta
            if not download_file(video_input, video_path):
                return {"status": "error", "error": "Failed to download video"}
        else:
            # Path do B2 - gera signed URL
            if not B2_AVAILABLE:
                return {"status": "error", "error": "B2 not configured"}
            
            try:
                signed_url = s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': B2_BUCKET, 'Key': video_input},
                    ExpiresIn=300
                )
                if not download_file(signed_url, video_path):
                    return {"status": "error", "error": "Failed to download video from B2"}
            except Exception as e:
                return {"status": "error", "error": f"Failed to generate B2 signed URL: {e}"}
        
        # Download template (se fornecido)
        template_path = None
        template_url = config.get("template_url")
        if template_url:
            template_path = TEMP_DIR / f"template_{uuid.uuid4().hex}.png"
            if download_file(template_url, template_path):
                config["template_path"] = str(template_path)
            else:
                logger.warning("⚠️ Falha ao baixar template, usando padrão")
        
        # Processa baseado no modo
        if mode == "auto":
            result = process_auto_mode(video_path, anime_name, config)
        else:
            result = process_manual_mode(video_path, anime_name, config)
        
        # Limpa arquivos temporários
        if video_path.exists():
            video_path.unlink()
        if template_path and template_path.exists():
            template_path.unlink()
        
        # Limpa memória final
        clean_memory()
        
        logger.info("=" * 60)
        logger.info("✅ AnimeCut Serverless - Concluído")
        logger.info("=" * 60)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Erro fatal no handler: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# ==================== INICIALIZAÇÃO ====================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("✂️ AnimeCut Cloud Serverless v6.0")
    logger.info("=" * 60)
    logger.info(f"✅ AI Available: {AI_AVAILABLE}")
    logger.info(f"✅ MoviePy Available: {MOVIEPY_AVAILABLE}")
    logger.info(f"✅ PIL Available: {PIL_AVAILABLE}")
    logger.info(f"✅ B2 Available: {B2_AVAILABLE}")
    logger.info("=" * 60)
    logger.info("🚀 Iniciando RunPod Serverless Handler...")
    logger.info("=" * 60)
    
    runpod.serverless.start({"handler": handler})
