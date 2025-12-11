#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
✂️ AnimeCut Serverless v7.0 PRO (GPU + IA)
Identificação de cenas virais com Qwen 2.5 e Whisper
"""

import runpod
import os
import sys
import logging
import tempfile
import requests
import gc
import json
import time
import uuid
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ==================== CONFIGURAÇÃO ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AnimeCutPro")

# Diretórios
TEMP_DIR = Path("/tmp/animecut")
OUTPUT_DIR = Path("/tmp/animecut/output")
TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Debug Env Vars
print("--- ENV VARS DEBUG ---")
for k, v in os.environ.items():
    if "KEY" in k or "SECRET" in k or "TOKEN" in k:
        print(f"{k}: {'*' * 8}")
    else:
        print(f"{k}: {v}")
print("----------------------")
sys.stdout.flush()

# Caminho do Volume de Rede (Ajustar conforme necessário)
VOLUME_PATH = Path("/runpod-volume")
MODELS_PATH = VOLUME_PATH / "models"
QWEN_MODEL_PATH = MODELS_PATH / "Qwen2.5-7B-Instruct" # Caminho provável, ajustável via ENV

print("=" * 60)
print("✂️ AnimeCut Serverless v7.0 PRO - GPU/AI Activated")
print(f"📂 Volume Path: {VOLUME_PATH}")
print("=" * 60)

# ==================== IMPORTS CONDICIONAIS ====================
try:
    from moviepy.editor import (
        VideoFileClip, ImageClip, CompositeVideoClip,
        ColorClip, TextClip, AudioFileClip
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
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from faster_whisper import WhisperModel
    
    GPU_AVAILABLE = torch.cuda.is_available()
    DEVICE = "cuda" if GPU_AVAILABLE else "cpu"
    
    if GPU_AVAILABLE:
        logger.info(f"✅ GPU Detectada: {torch.cuda.get_device_name(0)}")
        logger.info(f"💾 VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("⚠️ GPU NÃO detectada, processamento será lento!")
        
    AI_AVAILABLE = True
except ImportError as e:
    AI_AVAILABLE = False
    GPU_AVAILABLE = False
    logger.error(f"❌ Bibliotecas de IA faltando: {e}")

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
        logger.warning("⚠️ B2 credentials não configuradas nas Env Vars!")
        # Debug das keys (apenas tamanho para segurança)
        logger.info(f"🔑 Key ID Len: {len(B2_KEY_ID)}")
        logger.info(f"🔑 App Key Len: {len(B2_APP_KEY)}")
except Exception as e:
    B2_AVAILABLE = False
    logger.error(f"❌ Erro ao configurar B2: {e}")

# ==================== CARREGAMENTO DE MODELOS ====================
whisper_model = None
qwen_model = None
qwen_tokenizer = None

def load_whisper():
    """Carrega Whisper na GPU"""
    global whisper_model
    if whisper_model is None and AI_AVAILABLE:
        try:
            logger.info("🎧 Carregando Whisper (Large-v3)...")
            whisper_model = WhisperModel("large-v3", device=DEVICE, compute_type="float16")
            logger.info("✅ Whisper carregado")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar Whisper: {e}")

def load_qwen():
    """Carrega Qwen 2.5 da GPU/Volume"""
    global qwen_model, qwen_tokenizer
    if qwen_model is None and AI_AVAILABLE:
        try:
            model_path = str(QWEN_MODEL_PATH)
            
            # Fallback para Hub se não achar no volume
            if not os.path.exists(model_path):
                logger.warning(f"⚠️ Modelo não encontrado no volume: {model_path}")
                logger.info("🌐 Tentando baixar do HuggingFace (Qwen/Qwen2.5-7B-Instruct)...")
                model_path = "Qwen/Qwen2.5-7B-Instruct"
            else:
                logger.info(f"📂 Carregando Qwen do volume: {model_path}")

            logger.info("🧠 Carregando Qwen 2.5...")
            qwen_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            qwen_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            logger.info("✅ Qwen 2.5 carregado")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar Qwen: {e}")

# ==================== DOWNLOAD ====================
def download_video(url: str) -> str:
    try:
        logger.info(f"📥 Baixando vídeo...")
        temp_file = TEMP_DIR / f"input_{uuid.uuid4().hex[:8]}.mp4"
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        downloaded = 0
        with open(temp_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0 and downloaded % (10*1024*1024) == 0:
                    logger.info(f"📥 Download: {downloaded/total*100:.1f}%")
        logger.info(f"✅ Download completo: {temp_file}")
        return str(temp_file)
    except Exception as e:
        logger.error(f"❌ Erro download: {e}")
        raise

def download_background(url: str) -> Optional[str]:
    try:
        if not url: return None
        logger.info(f"🖼️ Baixando background...")
        temp_file = TEMP_DIR / f"bg_{uuid.uuid4().hex[:8]}.png"
        response = requests.get(url, timeout=60)
        with open(temp_file, 'wb') as f: f.write(response.content)
        return str(temp_file)
    except: return None

# ==================== IA: QUICK TRANSCRIPTION & ANALYSIS ====================
def analyze_video_content(video_path: str, anime_name: str) -> List[Dict]:
    """Analisa vídeo para encontrar cenas virais"""
    try:
        load_whisper()
        load_qwen()
        
        if not whisper_model or not qwen_model:
            raise Exception("Modelos de IA não carregados")
            
        # 1. Extração de áudio
        logger.info("🔊 Extraindo áudio para transcrição...")
        audio_path = TEMP_DIR / f"temp_audio_{uuid.uuid4().hex[:8]}.wav"
        
        # Usar ffmpeg diretamente é mais rápido que moviepy para extrair audio
        import subprocess
        subprocess.run([
            'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', 
            '-ar', '16000', '-ac', '1', str(audio_path), '-y', '-hide_banner', '-loglevel', 'error'
        ])
        
        # 2. Transcrição
        logger.info("🎤 Transcrevendo com Whisper...")
        segments, _ = whisper_model.transcribe(str(audio_path), language="pt")
        
        transcript = []
        full_text = ""
        for seg in segments:
            transcript.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text
            })
            full_text += f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}\n"
            
        logger.info(f"📝 Transcrição completa ({len(transcript)} segmentos)")
        
        # 3. Análise com Qwen
        logger.info("🧠 Analisando roteiro com Qwen 2.5...")
        
        prompt = f"""
        Você é um editor de vídeo especialista em Animes e TikTok.
        Analise o seguinte roteiro transcrito do anime '{anime_name}'.
        Identifique as 3 MELHORES cenas para clipes virais (entre 40s e 90s).
        Procure por momentos de: Ação Intensa, Plot Twist, Comédia, Emoção Fore ou Frases Impactantes.
        
        ROTEIRO:
        {full_text[:12000]} # Limite de contexto
        
        Retorne APENAS um JSON neste formato, sem explicações:
        [
            {{
                "start": 10.5,
                "end": 65.0,
                "reason": "Explicação curta do motivo",
                "title": "TÍTULO VIRAL e CURTO",
                "score": 95
            }}
        ]
        """
        
        inputs = qwen_tokenizer([prompt], return_tensors="pt").to(DEVICE)
        outputs = qwen_model.generate(
            **inputs, 
            max_new_tokens=1000,
            temperature=0.7
        )
        response_text = qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extrair JSON da resposta (pode ter texto antes/depois)
        json_str = response_text
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0]
        elif "[" in response_text and "]" in response_text:
            start = response_text.find("[")
            end = response_text.rfind("]") + 1
            json_str = response_text[start:end]
            
        viral_cuts = json.loads(json_str)
        logger.info(f"🔥 {len(viral_cuts)} cenas virais identificadas!")
        
        # Limpeza
        os.remove(audio_path)
        
        return viral_cuts
        
    except Exception as e:
        logger.error(f"❌ Erro na análise de IA: {e}")
        # Fallback para cortes manuais se IA falhar
        return []

# ==================== RENDERIZAÇÃO OTIMIZADA ====================
def criar_titulo_pil(texto: str, w: int, h: int, duracao: float, style: dict) -> ImageClip:
    """Cria título com PIL"""
    if not PIL_AVAILABLE: return None
    try:
        img = Image.new('RGBA', (w, h), (0,0,0,0))
        draw = ImageDraw.Draw(img)
        
        font_size = style.get("fontSize", 60)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
            
        color = style.get("textColor", "#FFFFFF")
        stroke = style.get("borderColor", "#000000")
        width = style.get("borderWidth", 3)
        
        # Centraliza texto
        bbox = draw.textbbox((0, 0), texto, font=font)
        text_w = bbox[2] - bbox[0]
        x = (w - text_w) // 2
        y = int(h * (style.get("verticalPosition", 15) / 100))
        
        # Borda
        for dx in range(-width, width+1):
            for dy in range(-width, width+1):
                draw.text((x+dx, y+dy), texto, font=font, fill=stroke)
        
        draw.text((x, y), texto, font=font, fill=color)
        
        return ImageClip(np.array(img)).set_duration(duracao)
    except: return None

def processar_corte(video_path: str, cut_data: Dict, num: int, config: Dict) -> str:
    try:
        start = cut_data['start']
        end = cut_data['end']
        duration = end - start
        
        logger.info(f"🎬 Renderizando Corte {num}: {start:.1f}-{end:.1f} ({config.get('animeName')})")
        
        with VideoFileClip(video_path) as video:
            clip = video.subclip(start, end)
            
            # Anti-Shadowban (Speed)
            if config.get("antiShadowban"):
                clip = clip.fx(speedx, 1.05)
            
            target_w, target_h = 1080, 1920
            
            # Verticalização Inteligente (Crop Central Simples por enquanto)
            # Para otimizar, não vou usar mediapipe aqui se estivermos com pouco tempo
            # Mas vamos manter o resize básico
            
            # Background
            bg_path = config.get("background_path")
            if bg_path:
                from PIL import Image as PILImage
                bg_img = PILImage.open(bg_path).convert('RGB').resize((target_w, target_h))
                bg_clip = ImageClip(np.array(bg_img)).set_duration(clip.duration)
            else:
                bg_clip = ColorClip(size=(target_w, target_h), color=(10,10,20)).set_duration(clip.duration)
            
            # Ajuste do Clip
            scale = target_w / clip.w
            clip_resized = clip.resize(scale)
            clip_pos = clip_resized.set_position(('center', 'center'))
            
            # Título Inteligente (do Qwen) ou Fallback
            titulo_texto = cut_data.get("title", config.get("titulo", ""))
            
            layers = [bg_clip, clip_pos]
            
            if config.get("generateTitles"):
                t_clip = criar_titulo_pil(titulo_texto.upper(), target_w, target_h, clip.duration, config.get("titleStyle", {}))
                if t_clip: layers.append(t_clip)
            
            final = CompositeVideoClip(layers, size=(target_w, target_h))
            
            output_path = OUTPUT_DIR / f"cut_{num}_{uuid.uuid4().hex[:6]}.mp4"
            
            # OTIMIZAÇÃO DE RENDER: usar preset ultrafast e threads
            final.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac',
                preset='ultrafast',  # MUITO MAIS RÁPIDO
                threads=4,          # Usa múltiplos núcleos
                ffmpeg_params=['-crf', '23'], # Qualidade balanceada
                verbose=False,
                logger=None
            )
            
            final.close()
            return str(output_path)
            
    except Exception as e:
        logger.error(f"❌ Erro render corte {num}: {e}")
        raise

# ==================== UPLOAD B2 ====================
def upload_to_b2(file_path: str) -> str:
    if not B2_AVAILABLE: return None
    try:
        name = f"animecut/v7/{os.path.basename(file_path)}"
        logger.info(f"📤 Upload B2: {name}")
        s3_client.upload_file(file_path, B2_BUCKET, name)
        return s3_client.generate_presigned_url('get_object', Params={'Bucket': B2_BUCKET, 'Key': name}, ExpiresIn=3600)
    except Exception as e:
        logger.error(f"❌ Erro upload B2: {e}")
        return None

# ==================== HANDLER ====================
def handler(event):
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    input_data = event.get("input", {})
    if input_data.get("mode") == "test":
        return {"status": "success", "gpu": GPU_AVAILABLE, "qwen": AI_AVAILABLE}
    
    try:
        video_url = input_data.get("video_url")
        if not video_url: raise Exception("No video_url")
        
        # Config
        config = {
            "animeName": input_data.get("animeName", "Anime"),
            "cutType": input_data.get("cutType", "auto"), # Auto por padrão agora
            "antiShadowban": input_data.get("antiShadowban", True),
            "generateTitles": input_data.get("generateTitles", True),
            "titleStyle": input_data.get("titleStyle", {"fontSize": 70}),
            "background_path": download_background(input_data.get("background_url"))
        }
        
        # 1. Download
        video_path = download_video(video_url)
        
        # 2. Definição de Cortes (Auto ou Manual)
        cuts_to_process = []
        
        if config["cutType"] == "auto" and AI_AVAILABLE:
            logger.info("🤖 Iniciando Modo Automático (IA)...")
            viral_cuts = analyze_video_content(video_path, config["animeName"])
            if viral_cuts:
                cuts_to_process = viral_cuts
            else:
                logger.warning("⚠️ IA não encontrou cortes, fallback para manual")
                config["cutType"] = "manual"
        
        if config["cutType"] != "auto" or not cuts_to_process:
            # Manual fallback
            duration = VideoFileClip(video_path).duration
            for i in range(min(5, int(duration/60))):
                cuts_to_process.append({
                    "start": i*60, 
                    "end": min((i+1)*60, duration),
                    "title": config["animeName"]
                })
        
        # 3. Renderização
        results = []
        for i, cut in enumerate(cuts_to_process):
            out_path = processar_corte(video_path, cut, i+1, config)
            b2_url = upload_to_b2(out_path)
            
            results.append({
                "path": out_path,
                "url": b2_url,
                "title": cut.get("title"),
                "score": cut.get("score", 0)
            })
            
            # Limpa temp file do corte para economizar espaço
            # Mas mantemos para retorno local se precisar
            
        # Cleanup video original
        try: os.remove(video_path)
        except: pass
        
        return {"status": "success", "cuts": results}
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    print("--- INICIANDO WORKER ---")
    sys.stdout.flush()
    
    try:
        runpod.serverless.start({"handler": handler})
    except Exception as e:
        print(f"CRITICAL ERROR IN RUNPOD START: {e}")
        sys.stdout.flush()
    
    print("--- WORKER TERMINOU INESPERADAMENTE ---")
    sys.stdout.flush()
    
    # Previne exit code 0 imediato se o runpod falhar
    while True:
        time.sleep(10)
