#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
✂️ AnimeCut Serverless v12.0 ULTIMATE HYBRID
Stack: Qwen 2.5, Whisper V3 Turbo, YOLOv8, DeepFilterNet, NVENC + MoviePy V2
"""

import runpod
import os
import sys
import logging
import tempfile
import requests
import gc
import json
import uuid
import math
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random

# --- CONFIGURAÇÃO DE AMBIENTE ---
sys.path.append("/runpod-volume/site-packages") 
sys.path.append("/runpod-volume")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AnimeCutUltimate")

# Diretórios
TEMP_DIR = Path("/tmp/animecut")
OUTPUT_DIR = Path("/runpod-volume/output")
VOLUME_PATH = Path("/runpod-volume")
MODELS_PATH = VOLUME_PATH / "models"
QWEN_MODEL_PATH = MODELS_PATH / "Qwen2.5-7B-Instruct"
FONT_PATH = VOLUME_PATH / "fonts" / "impact.ttf"

# Garante diretórios
for p in [TEMP_DIR, OUTPUT_DIR, FONT_PATH.parent]:
    p.mkdir(parents=True, exist_ok=True)

# ==================== IMPORTS ROBUSTOS ====================

# 1. Visão Computacional (OpenCV + YOLO)
CV2_AVAILABLE = False
try:
    import cv2
    import numpy as np
    from ultralytics import YOLO
    CV2_AVAILABLE = True
    logger.info("✅ OpenCV + YOLO disponível")
except ImportError as e:
    logger.warning(f"⚠️ Visão limitada: {e}")

# 2. MoviePy V2 (Sintaxe Nova) + Compatibilidade
MOVIEPY_AVAILABLE = False
MOVIEPY_V2 = False
try:
    # Tenta MoviePy v2 primeiro
    from moviepy import *
    from moviepy.video.io.VideoFileClip import VideoFileClip
    from moviepy.video.VideoClip import ImageClip, ColorClip, TextClip
    from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
    from moviepy.audio.io.AudioFileClip import AudioFileClip
    from moviepy.video.fx import MirrorX, GammaCorr, MultiplyColor
    MOVIEPY_AVAILABLE = True
    MOVIEPY_V2 = True
    logger.info("✅ MoviePy v2 carregado")
except ImportError:
    try:
        # Fallback para MoviePy v1
        from moviepy.editor import (
            VideoFileClip, ImageClip, CompositeVideoClip,
            ColorClip, TextClip, AudioFileClip
        )
        from moviepy.video.fx.all import mirror_x, gamma_corr, colorx
        MOVIEPY_AVAILABLE = True
        MOVIEPY_V2 = False
        logger.info("✅ MoviePy v1 carregado (compatibilidade)")
    except ImportError as e:
        logger.error(f"❌ MoviePy não disponível: {e}")

# 3. IA (Transformers/Torch)
AI_AVAILABLE = False
GPU_AVAILABLE = False
try:
    import torch
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
    from faster_whisper import WhisperModel
    
    GPU_AVAILABLE = torch.cuda.is_available()
    DEVICE = "cuda" if GPU_AVAILABLE else "cpu"
    AI_AVAILABLE = True
    
    if GPU_AVAILABLE:
        logger.info(f"✅ GPU Detectada: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("⚠️ GPU NÃO detectada")
except ImportError:
    logger.warning("⚠️ IA libs ausentes.")

# 4. Pillow (Imagens)
try:
    from PIL import Image, ImageDraw, ImageFont, ImageColor
    PIL_AVAILABLE = True
except ImportError as e:
    PIL_AVAILABLE = False
    logger.warning(f"⚠️ PIL não disponível: {e}")

# 5. B2 Backblaze (Configuração Robusta)
B2_AVAILABLE = False
try:
    import boto3
    from botocore.client import Config
    
    B2_KEY_ID = os.environ.get("B2_KEY_ID", "00568702c2cbfc60000000002")
    B2_APP_KEY = os.environ.get("B2_APPLICATION_KEY", "K005W2f9Ske24aextx8LwxMRxsoYnNE")
    B2_ENDPOINT = os.environ.get("B2_ENDPOINT", "https://s3.us-east-005.backblazeb2.com")
    B2_BUCKET = os.environ.get("B2_BUCKET_NAME", "KortexAI2")
    
    if B2_KEY_ID and B2_APP_KEY:
        s3_client = boto3.client(
            "s3",
            endpoint_url=B2_ENDPOINT,
            aws_access_key_id=B2_KEY_ID,
            aws_secret_access_key=B2_APP_KEY,
            config=Config(signature_version="s3v4")
        )
        B2_AVAILABLE = True
        logger.info(f"✅ Backblaze B2 configurado no bucket: {B2_BUCKET}")
    else:
        B2_AVAILABLE = False
        logger.warning("⚠️ B2 credentials não configuradas!")
except Exception as e:
    B2_AVAILABLE = False
    logger.error(f"❌ Erro ao configurar B2: {e}")

# ==================== UTILITÁRIOS DE MÍDIA ====================

def download_font():
    """Baixa fonte Impact se não existir"""
    if not FONT_PATH.exists():
        try:
            logger.info("📥 Baixando fonte Impact...")
            url = "https://github.com/google/fonts/raw/main/ofl/oswald/Oswald-Bold.ttf"
            r = requests.get(url, timeout=30)
            with open(FONT_PATH, "wb") as f:
                f.write(r.content)
            logger.info(f"✅ Fonte salva em: {FONT_PATH}")
        except Exception as e:
            logger.error(f"❌ Erro ao baixar fonte: {e}")

download_font()

def clean_audio_deepfilter(input_path: Path) -> Path:
    """Usa DeepFilterNet para remover música de fundo e isolar voz"""
    logger.info("🧹 Limpando áudio com DeepFilterNet...")
    try:
        output_dir = input_path.parent
        cmd = ["deepFilter", str(input_path), "-o", str(output_dir)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        cleaned = output_dir / f"{input_path.stem}_DeepFilterNet3.wav"
        if cleaned.exists(): 
            return cleaned
        return input_path
    except Exception as e:
        logger.warning(f"⚠️ DeepFilter falhou: {e}. Usando original.")
        return input_path

def download_video(url: str) -> str:
    """Download robusto de vídeo"""
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
                if total > 0 and downloaded % (20*1024*1024) == 0:
                    logger.info(f"📥 Download: {downloaded/total*100:.1f}%")
        
        logger.info(f"✅ Download completo: {temp_file}")
        return str(temp_file)
    except Exception as e:
        logger.error(f"❌ Erro download: {e}")
        raise

def download_background(url: str) -> Optional[str]:
    """Download de imagem de background"""
    if not url: return None
    try:
        logger.info(f"🖼️ Baixando background...")
        temp_file = TEMP_DIR / f"bg_{uuid.uuid4().hex[:8]}.png"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with open(temp_file, 'wb') as f: 
            f.write(response.content)
        return str(temp_file)
    except Exception as e:
        logger.warning(f"⚠️ Erro background: {e}")
        return None

# ==================== SENSOR DE ADRENALINA (ActionDetector) ====================
class ActionDetector:
    """Detecta cenas de luta analisando movimento brusco nos pixels"""
    def __init__(self, video_path: str):
        self.video_path = video_path
    
    def calculate_visual_energy(self, sample_rate=1.0) -> List[Dict]:
        """Calcula energia visual (movimento) usando Frame Differencing"""
        if not CV2_AVAILABLE: return []
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        prev_frame = None
        energy_scores = []
        
        step = int(fps * sample_rate) if fps > 0 else 30
        if step < 1: step = 1
        
        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if prev_frame is not None:
                delta = cv2.absdiff(prev_frame, gray)
                thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
                score = np.sum(thresh) / thresh.size
                
                timestamp = i / fps
                energy_scores.append({"time": timestamp, "score": score})
            
            prev_frame = gray
            
        cap.release()
        
        # Normaliza scores (0 a 100)
        if not energy_scores: return []
        max_score = max(s["score"] for s in energy_scores) or 1
        for item in energy_scores:
            item["score"] = (item["score"] / max_score) * 100
            
        return energy_scores

    def detect_high_energy_segments(self, threshold=70) -> List[Dict]:
        """Retorna segmentos de alta energia (Ação/Luta)"""
        logger.info("⚡ Iniciando Sensor de Adrenalina...")
        visual_data = self.calculate_visual_energy()
        
        action_segments = []
        current_segment = None
        
        for data in visual_data:
            is_action = data["score"] > threshold
            
            if is_action:
                if current_segment is None:
                    current_segment = {"start": data["time"], "end": data["time"], "score": data["score"]}
                else:
                    current_segment["end"] = data["time"]
                    current_segment["score"] = max(current_segment["score"], data["score"])
            else:
                if current_segment:
                    if (current_segment["end"] - current_segment["start"]) > 2.0:
                        action_segments.append(current_segment)
                    current_segment = None
                    
        logger.info(f"⚡ {len(action_segments)} cenas de ação intensa detectadas!")
        return action_segments

# ==================== ANTI-SHADOWBAN (Compatível com ambas versões MoviePy) ====================
def apply_antishadowban(clip):
    """Aplica filtros matemáticos para tornar o vídeo único"""
    logger.info("🛡️ Aplicando Anti-Shadowban 2.0...")
    
    # 1. Espelhamento Inteligente (50% de chance)
    if random.choice([True, False]): 
        try:
            if MOVIEPY_V2:
                from moviepy.video.fx import MirrorX
                clip = clip.with_effect(MirrorX())
            else:
                clip = clip.fx(mirror_x)
            logger.info("   -> Vídeo Espelhado")
        except Exception as e:
            logger.warning(f"   -> Falha espelhamento: {e}")

    # 2. Color Grading Aleatório (Sutil)
    gamma_val = random.uniform(0.95, 1.05)
    contrast_val = random.uniform(0.95, 1.05)
    
    try:
        if MOVIEPY_V2:
            from moviepy.video.fx import GammaCorr, MultiplyColor
            clip = clip.with_effect(GammaCorr(gamma_val)).with_effect(MultiplyColor(contrast_val))
        else:
            clip = clip.fx(gamma_corr, gamma_val).fx(colorx, contrast_val)
        logger.info(f"   -> Color Grading: Gamma={gamma_val:.2f}, Contrast={contrast_val:.2f}")
    except Exception as e:
        logger.warning(f"   -> Falha Color Grading: {e}")

    # 3. Micro-Zoom "Breath" (Respiração) - Usando CV2 se disponível
    if CV2_AVAILABLE:
        def zoom_effect(get_frame, t):
            img = get_frame(t)
            h, w = img.shape[:2]
            scale = 1 + 0.03 * (0.5 + 0.5 * math.sin(2 * math.pi * t / 5.0))
            
            new_w, new_h = int(w / scale), int(h / scale)
            x1 = (w - new_w) // 2
            y1 = (h - new_h) // 2
            
            cropped = img[y1:y1+new_h, x1:x1+new_w]
            return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        
        try:
            if MOVIEPY_V2:
                clip = clip.with_transform(zoom_effect)
            else:
                clip = clip.fl(zoom_effect)
            logger.info("   -> Micro-Zoom Dinâmico Aplicado")
        except:
            pass

    return clip

# ==================== CARREGADORES IA ====================
whisper_pipeline = None
qwen_model = None
qwen_tokenizer = None
yolo_model = None

def load_turbo_whisper():
    """Carrega Whisper com Flash Attention 2"""
    global whisper_pipeline
    if whisper_pipeline is None and AI_AVAILABLE:
        try:
            logger.info("🚀 Carregando Whisper V3 com Flash Attention 2...")
            whisper_pipeline = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3",
                torch_dtype=torch.float16,
                device="cuda:0" if GPU_AVAILABLE else "cpu",
                model_kwargs={"attn_implementation": "flash_attention_2"}
            )
            logger.info("✅ Whisper Turbo Carregado!")
        except Exception as e:
            logger.error(f"❌ Erro Whisper FA2: {e}")
            try:
                whisper_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-large-v3",
                    torch_dtype=torch.float16 if GPU_AVAILABLE else torch.float32,
                    device="cuda:0" if GPU_AVAILABLE else "cpu"
                )
            except Exception as e2:
                logger.error(f"❌ Falha crítica no Whisper: {e2}")

def load_qwen():
    """Carrega Qwen 2.5 da GPU/Volume"""
    global qwen_model, qwen_tokenizer
    if qwen_model is None and AI_AVAILABLE:
        try:
            model_path = str(QWEN_MODEL_PATH)
            
            if not os.path.exists(model_path):
                logger.warning(f"⚠️ Modelo não encontrado no volume: {model_path}")
                logger.info("🌐 Tentando baixar do HuggingFace...")
                model_path = "Qwen/Qwen2.5-7B-Instruct"
            else:
                logger.info(f"📂 Carregando Qwen do volume: {model_path}")

            logger.info("🧠 Carregando Qwen 2.5...")
            qwen_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            qwen_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16 if GPU_AVAILABLE else torch.float32,
                trust_remote_code=True
            )
            logger.info("✅ Qwen 2.5 carregado")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar Qwen: {e}")

def get_yolo():
    """Carrega YOLO para detecção de rostos"""
    global yolo_model
    if yolo_model is None and CV2_AVAILABLE:
        try:
            yolo_model = YOLO("yolov8n.pt") 
            logger.info("✅ YOLO carregado")
        except Exception as e:
            logger.warning(f"⚠️ Erro YOLO: {e}")
            yolo_model = None
    return yolo_model

# ==================== GERADOR DE TÍTULOS (Do v8.0) ====================
def hex_to_rgb(hex_color):
    """Converte hex (#RRGGBB) para tupla RGB."""
    if hex_color.startswith('#'):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    else:
        try: 
            return ImageColor.getrgb(hex_color)
        except: 
            return (255, 255, 255)

def criar_titulo_pil(texto, largura_video, altura_video, duracao, 
                     font_filename=None, 
                     font_size=None,
                     text_color="#FFFFFF", 
                     stroke_color="#000000", 
                     stroke_width=6,
                     pos_vertical=0.15):
    """
    Renderiza título PRO com quebra de linha e contorno
    """
    if not PIL_AVAILABLE: 
        return None
    
    # Cores
    text_color_rgb = hex_to_rgb(text_color)
    stroke_color_rgb = hex_to_rgb(stroke_color)
    
    # Fonte
    if font_size is None:
        font_size = int(largura_video * 0.08)
    
    # Tenta usar a fonte baixada
    font_to_use = str(FONT_PATH)
    if not os.path.exists(font_to_use):
        font_to_use = "arial.ttf"
        
    try:
        font = ImageFont.truetype(font_to_use, font_size)
    except:
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()

    # Canvas
    canvas_h = int(altura_video * 0.4)
    img = Image.new('RGBA', (largura_video, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Quebra de Linha (Word Wrap)
    palavras = texto.split()
    linhas = []
    linha_atual = []
    
    for palavra in palavras:
        linha_atual.append(palavra)
        try:
            bbox = draw.textbbox((0, 0), " ".join(linha_atual), font=font)
            w = bbox[2] - bbox[0]
        except:
            w = draw.textlength(" ".join(linha_atual), font=font)
            
        if w > largura_video * 0.9:
            linha_atual.pop()
            if linha_atual:
                linhas.append(" ".join(linha_atual))
            linha_atual = [palavra]
            if len(linhas) >= 2: break
            
    if linha_atual and len(linhas) < 2:
        linhas.append(" ".join(linha_atual))
    
    linhas = linhas[:2]
    
    # Desenhar
    y = 20
    for linha in linhas:
        # Contorno
        for off_x in range(-stroke_width, stroke_width + 1):
            for off_y in range(-stroke_width, stroke_width + 1):
                if off_x != 0 or off_y != 0:
                    try:
                        draw.text((largura_video / 2 + off_x, y + off_y), 
                                 linha, font=font, fill=stroke_color_rgb, 
                                 anchor="mt")
                    except:
                        w_txt = draw.textlength(linha, font=font)
                        draw.text(((largura_video - w_txt) / 2 + off_x, y + off_y), 
                                 linha, font=font, fill=stroke_color_rgb)

        # Texto Principal
        try:
            draw.text((largura_video / 2, y), linha, font=font, 
                     fill=text_color_rgb, anchor="mt")
        except:
            w_txt = draw.textlength(linha, font=font)
            draw.text(((largura_video - w_txt) / 2, y), 
                     linha, font=font, fill=text_color_rgb)
        
        # Avança Y
        try:
            bbox = draw.textbbox((0, 0), linha, font=font)
            h = bbox[3] - bbox[1]
        except:
            h = font_size
        y += h + 15

    # Converter para MoviePy Clip
    numpy_img = np.array(img)
    
    if MOVIEPY_V2:
        from moviepy.video.VideoClip import ImageClip
        clip = ImageClip(numpy_img).with_duration(duracao)
        pos_y = int(altura_video * pos_vertical)
        clip = clip.with_position(('center', pos_y))
    else:
        clip = ImageClip(numpy_img).set_duration(duracao)
        pos_y = int(altura_video * pos_vertical)
        clip = clip.set_position(('center', pos_y))
    
    return clip

# ==================== ANÁLISE DE VÍDEO COM IA ====================
def analyze_video_content(video_path: str, anime_name: str) -> List[Dict]:
    """Analisa vídeo para encontrar cenas virais (Pipeline Turbo)"""
    try:
        load_turbo_whisper()
        load_qwen()
        
        if not whisper_pipeline or not qwen_model:
            raise Exception("Modelos de IA não carregados")
            
        # 1. Extração de áudio
        logger.info("🔊 Extraindo áudio para transcrição...")
        raw_audio_path = TEMP_DIR / f"temp_audio_{uuid.uuid4().hex[:8]}.wav"
        
        subprocess.run([
            'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', 
            '-ar', '16000', '-ac', '1', str(raw_audio_path), '-y', 
            '-hide_banner', '-loglevel', 'error'
        ])
        
        # 2. Limpeza de Áudio
        logger.info("🧹 Limpando áudio (DeepFilterNet)...")
        clean_audio_path = clean_audio_deepfilter(raw_audio_path)
        
        # 3. Transcrição
        logger.info("🎤 Transcrevendo...")
        
        result = whisper_pipeline(
            str(clean_audio_path),
            chunk_length_s=30,
            batch_size=24 if GPU_AVAILABLE else 4,
            return_timestamps=True,
            generate_kwargs={"language": "portuguese"} 
        )
        
        segments = result.get("chunks", [])
        
        transcript_objs = []
        for seg in segments:
            start_t, end_t = seg["timestamp"]
            transcript_objs.append({
                "start": start_t,
                "end": end_t,
                "text": seg["text"],
                "type": "dialogue"
            })
            
        # 4. RODAR SENSOR DE ADRENALINA ⚡
        detector = ActionDetector(video_path)
        action_scenes = detector.detect_high_energy_segments()
        
        # Injetar cenas de ação no transcript
        for action in action_scenes:
            transcript_objs.append({
                "start": action["start"],
                "end": action["end"],
                "text": f"[⚡ CENA DE AÇÃO INTENSA / LUTA - {int(action['score'])}% ENERGIA]",
                "type": "action"
            })
            
        # Ordenar tudo por tempo
        transcript_objs.sort(key=lambda x: x["start"])
        
        # Gerar texto final
        full_text = ""
        for item in transcript_objs:
            if item.get("type") == "action":
                full_text += f"\n[{item['start']:.1f}s - {item['end']:.1f}s] {item['text']}\n"
            else:
                full_text += f"[{item['start']:.1f}s - {item['end']:.1f}s] {item['text']}\n"
            
        logger.info(f"📝 Roteiro Híbrido Gerado: {len(transcript_objs)} eventos")
        
        # 5. Análise com Qwen
        logger.info("🧠 Analisando roteiro com Qwen 2.5...")
        
        prompt = f"""
        Você é um editor de vídeo especialista em Animes e TikTok.
        O roteiro abaixo contém DIÁLOGOS e MARCAÇÕES DE AÇÃO [⚡] detectadas por sensores.
        
        Analise o anime '{anime_name}'.
        Identifique as 3 MELHORES cenas para clipes virais (entre 40s e 90s).
        
        PRIORIDADE MÁXIMA:
        1. Cenas marcadas com [⚡ CENA DE AÇÃO INTENSA] (Lutas épicas mudo ou não).
        2. Plot Twists e revelações.
        3. Momentos engraçados.
        
        ROTEIRO HÍBRIDO:
        {full_text[:25000]} 
        
        Retorne APENAS um JSON:
        [
            {{
                "start": 10.5,
                "end": 65.0,
                "title": "TITULO CURTO E VIRAL",
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
        
        json_str = response_text
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0]
        elif "[" in response_text and "]" in response_text:
            start = response_text.find("[")
            end = response_text.rfind("]") + 1
            json_str = response_text[start:end]
            
        viral_cuts = json.loads(json_str)
        logger.info(f"🔥 {len(viral_cuts)} cenas virais identificadas!")
        
        try: 
            os.remove(raw_audio_path)
            if clean_audio_path != raw_audio_path:
                os.remove(clean_audio_path)
        except: 
            pass
        
        return viral_cuts
        
    except Exception as e:
        logger.error(f"❌ Erro na análise de IA: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

# ==================== PROCESSAMENTO DE CORTES (Compatível com MoviePy v1/v2) ====================
def processar_corte(video_path: str, cut_data: Dict, num: int, config: Dict) -> str:
    try:
        start = cut_data['start']
        end = cut_data['end']
        
        logger.info(f"🎬 Renderizando Corte {num}: {start:.1f}-{end:.1f} ({config.get('animeName')})")
        
        # Carrega vídeo com sintaxe compatível
        if MOVIEPY_V2:
            video = VideoFileClip(video_path)
            clip = video.subclipped(start, end)
        else:
            video = VideoFileClip(video_path)
            clip = video.subclip(start, end)
        
        # Anti-Shadowban
        if config.get("antiShadowban", True):
            clip = apply_antishadowban(clip)
        
        target_w, target_h = 1080, 1920
        
        # Background
        bg_path = config.get("background_path")
        if bg_path and os.path.exists(bg_path):
            from PIL import Image as PILImage
            bg_img = PILImage.open(bg_path).convert('RGB').resize((target_w, target_h))
            
            if MOVIEPY_V2:
                bg_clip = ImageClip(np.array(bg_img)).with_duration(clip.duration)
            else:
                bg_clip = ImageClip(np.array(bg_img)).set_duration(clip.duration)
        else:
            if MOVIEPY_V2:
                bg_clip = ColorClip(size=(target_w, target_h), color=(15,15,30)).with_duration(clip.duration)
            else:
                bg_clip = ColorClip(size=(target_w, target_h), color=(15,15,30)).set_duration(clip.duration)
        
        # ZOOM TÁTICO & SMART CROP
        zoom_factor = 1.15
        w, h = clip.w, clip.h
        new_w = w / zoom_factor
        new_h = h / zoom_factor
        
        # Tenta detectar rosto com YOLO para centralizar crop
        x1, y1 = w/2 - new_w/2, h/2 - new_h/2
        
        try:
            yolo = get_yolo()
            if yolo:
                # Analisa frame do meio
                frame = clip.get_frame(clip.duration / 2)
                results = yolo(frame, verbose=False)
                
                max_area = 0
                best_box = None
                
                for result in results:
                    for box in result.boxes:
                        if int(box.cls) == 0: # Person
                            xyxy = box.xyxy[0].cpu().numpy()
                            width = xyxy[2] - xyxy[0]
                            height = xyxy[3] - xyxy[1]
                            area = width * height
                            if area > max_area:
                                max_area = area
                                best_box = xyxy
                
                if best_box is not None:
                    face_cx = (best_box[0] + best_box[2]) / 2
                    face_cy = (best_box[1] + best_box[3]) / 2
                    
                    x1 = face_cx - (new_w / 2)
                    y1 = face_cy - (new_h / 2)
                    
                    x1 = max(0, min(x1, w - new_w))
                    y1 = max(0, min(y1, h - new_h))
                    logger.info("🎯 YOLO: Rosto detectado! Ajustando crop.")
        except Exception as e:
            logger.warning(f"⚠️ Erro no Smart Crop: {e}")
        
        # Aplica crop
        if MOVIEPY_V2:
            clip_cropped = clip.cropped(x1=x1, y1=y1, width=new_w, height=new_h)
            scale = target_w / clip_cropped.w
            clip_resized = clip_cropped.resized(width=target_w)
            clip_pos = clip_resized.with_position('center')
        else:
            clip_cropped = clip.crop(x1=x1, y1=y1, width=new_w, height=new_h)
            scale = target_w / clip_cropped.w
            clip_resized = clip_cropped.resize(scale)
            clip_pos = clip_resized.set_position(('center', 'center'))
        
        layers = [bg_clip, clip_pos]
        
        # Título
        titulo_texto = cut_data.get("title", config.get("animeName", "")).upper()
        
        if config.get("generateTitles", True) and titulo_texto:
            logger.info(f"🏷️ Gerando Título: {titulo_texto}")
            t_clip = criar_titulo_pil(
                titulo_texto, 
                target_w, target_h, clip.duration,
                font_filename="impact.ttf",
                font_size=config.get("titleStyle", {}).get("fontSize", 80),
                text_color=config.get("titleStyle", {}).get("textColor", "#FFD700"),
                stroke_color="#000000",
                stroke_width=6,
                pos_vertical=0.15
            )
            if t_clip:
                layers.append(t_clip)
        
        # Composição final
        if MOVIEPY_V2:
            final = CompositeVideoClip(layers, size=(target_w, target_h))
        else:
            final = CompositeVideoClip(layers, size=(target_w, target_h))
        
        output_path = OUTPUT_DIR / f"cut_{num}_{uuid.uuid4().hex[:6]}.mp4"
        
        # Render NVENC
        logger.info("⚙️ Iniciando Encode...")
        
        ffmpeg_params = [
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart'
        ]
        
        if GPU_AVAILABLE and torch.cuda.is_available():
            logger.info("🚀 Usando Aceleração NVENC (GPU)")
            codec = 'h264_nvenc'
            preset = 'fast'
            ffmpeg_params.extend([
               '-rc:v', 'vbr',
               '-cq:v', '23',
               '-b:v', '5M',
               '-maxrate:v', '8M',
               '-bufsize:v', '10M'
            ])
        else:
            logger.warning("🐢 Usando CPU Encoding")
            codec = 'libx264'
            preset = 'ultrafast'
            ffmpeg_params.extend(['-crf', '23'])
        
        # Escreve arquivo
        if MOVIEPY_V2:
            final.write_videofile(
                str(output_path),
                codec=codec,
                audio_codec='aac',
                preset=preset,
                threads=4,
                ffmpeg_params=ffmpeg_params,
                logger=None,
                verbose=False
            )
        else:
            final.write_videofile(
                str(output_path),
                codec=codec,
                audio_codec='aac',
                preset=preset,
                threads=4,
                ffmpeg_params=ffmpeg_params,
                logger=None,
                verbose=False
            )
        
        # Limpeza
        if MOVIEPY_V2:
            final.close()
            video.close()
        else:
            final.close()
            video.close()
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"❌ Erro render corte {num}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

# ==================== UPLOAD B2 ====================
def upload_to_b2(file_path: str) -> str:
    if not B2_AVAILABLE: 
        logger.warning("⚠️ Upload B2 cancelado (indisponível)")
        return None
    try:
        filename = os.path.basename(file_path)
        key = f"animecut/v12/{filename}"
        
        logger.info(f"📤 Uploading to B2: {B2_BUCKET}/{key}")
        
        s3_client.upload_file(file_path, B2_BUCKET, key)
        
        url = s3_client.generate_presigned_url(
            'get_object', 
            Params={'Bucket': B2_BUCKET, 'Key': key}, 
            ExpiresIn=86400
        )
        logger.info(f"✅ Upload Sucesso: {url}")
        return url
    except Exception as e:
        logger.error(f"❌ Erro upload B2: {e}")
        return None

# ==================== HANDLER PRINCIPAL ====================
def handler(event):
    gc.collect()
    if torch.cuda.is_available(): 
        torch.cuda.empty_cache()
    
    input_data = event.get("input", {})
    
    # Test mode
    if input_data.get("mode") == "test":
        return {
            "status": "success", 
            "gpu": GPU_AVAILABLE, 
            "moviepy": MOVIEPY_AVAILABLE,
            "moviepy_v2": MOVIEPY_V2,
            "ai": AI_AVAILABLE,
            "b2": B2_AVAILABLE
        }
    
    try:
        video_url = input_data.get("video_url")
        if not video_url: 
            raise Exception("No video_url provided")
        
        # Config params
        anime_name = input_data.get("animeName", "Anime")
        
        # 1. Download
        video_path = download_video(video_url)
        bg_path = download_background(input_data.get("background_url"))
        
        # Config Objeto
        config = {
            "animeName": anime_name,
            "antiShadowban": input_data.get("antiShadowban", True),
            "generateTitles": input_data.get("generateTitles", True),
            "titleStyle": input_data.get("titleStyle", {}),
            "background_path": bg_path
        }
        
        # 2. Definição de Cortes
        cuts = []
        if input_data.get("cutType", "auto") == "auto" and AI_AVAILABLE:
            logger.info("🤖 Modo Automático (IA)")
            cuts = analyze_video_content(video_path, anime_name)
        
        # Fallback manual se IA falhar ou não disponível
        if not cuts:
            logger.info("⚠️ Fallback para cortes manuais")
            try:
                if MOVIEPY_V2:
                    video = VideoFileClip(video_path)
                else:
                    from moviepy.editor import VideoFileClip
                    video = VideoFileClip(video_path)
                
                duration = video.duration
                video.close()
                
                num_cuts = min(5, int(duration/60))
                for i in range(num_cuts):
                    cuts.append({
                        "start": i*60, 
                        "end": min((i+1)*60, duration),
                        "title": f"{anime_name} - Parte {i+1}"
                    })
            except:
                # Fallback extremo
                cuts = [{"start": 30, "end": 90, "title": anime_name}]
        
        # 3. Processamento
        results = []
        for i, cut in enumerate(cuts):
            try:
                out_path = processar_corte(video_path, cut, i+1, config)
                b2_url = upload_to_b2(out_path)
                
                results.append({
                    "path": str(out_path),
                    "url": b2_url,
                    "title": cut.get("title"),
                    "score": cut.get("score", 0),
                    "start": cut.get("start"),
                    "end": cut.get("end")
                })
                
                # Limpa VRAM
                gc.collect()
                if torch.cuda.is_available(): 
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"❌ Erro no corte {i}: {e}")
                continue
        
        # Limpeza
        try: 
            os.remove(video_path)
        except: 
            pass
        
        if bg_path and os.path.exists(bg_path):
            try: 
                os.remove(bg_path)
            except: 
                pass
        
        return {
            "status": "success", 
            "cuts": results,
            "metadata": {
                "total_cuts": len(results),
                "moviepy_version": "v2" if MOVIEPY_V2 else "v1",
                "gpu_used": GPU_AVAILABLE
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Erro Handler: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    print("--- ANIMECUT ULTIMATE HYBRID v12.0 STARTING ---")
    sys.stdout.flush()
    runpod.serverless.start({"handler": handler})
