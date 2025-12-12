#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
✂️ AnimeCut Serverless v8.0 PRO (NVENC + Smart Titles + B2 Fix)
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
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random
try:
    import cv2
    import numpy as np
    import torch
    from transformers import pipeline
    from ultralytics import YOLO
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ==================== CONFIGURAÇÃO ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AnimeCutPro")

# Diretórios
TEMP_DIR = Path("/tmp/animecut")
OUTPUT_DIR = Path("/runpod-volume/output")  # Persistir no volume se possível
TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Caminho do Volume de Rede
VOLUME_PATH = Path("/runpod-volume")
MODELS_PATH = VOLUME_PATH / "models"
QWEN_MODEL_PATH = MODELS_PATH / "Qwen2.5-7B-Instruct"

# Fonte
FONT_PATH = VOLUME_PATH / "fonts" / "impact.ttf"
FONT_PATH.parent.mkdir(parents=True, exist_ok=True)

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
    from PIL import Image, ImageDraw, ImageFont, ImageColor
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
    
    # Credenciais com Fallback (valores atualizados conforme RunPod)
    B2_KEY_ID = os.environ.get("B2_KEY_ID", "00568702c2cbfc60000000001")
    B2_APP_KEY = os.environ.get("B2_APPLICATION_KEY", "K005aP6cXPuBIw6IakBaMHYtXx4VGq")
    B2_ENDPOINT = os.environ.get("B2_ENDPOINT", "https://s3.us-east-005.backblazeb2.com")
    B2_BUCKET = os.environ.get("B2_BUCKET_NAME", "KortexAI")  # Corrigido para KortexAI
    
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

# ==================== UTILITÁRIOS ====================
def download_font():
    """Baixa fonte Impact se não existir"""
    if not FONT_PATH.exists():
        try:
            logger.info("📥 Baixando fonte Impact...")
            # Usando GitHub Raw do Google Fonts (Oswald é similar a Impact e open source)
            url = "https://github.com/google/fonts/raw/main/ofl/oswald/Oswald-Bold.ttf"
            
            r = requests.get(url, timeout=30)
            with open(FONT_PATH, "wb") as f:
                f.write(r.content)
            logger.info(f"✅ Fonte salva em: {FONT_PATH}")
        except Exception as e:
            logger.error(f"❌ Erro ao baixar fonte: {e}")

download_font()

# ==================== CARREGAMENTO DE MODELOS ====================
whisper_model = None
qwen_model = None
qwen_tokenizer = None

# ==================== FUNÇÕES TURBO (YOLO + DEEPFILTER) ====================
def clean_audio_deepfilter(input_path: Path) -> Path:
    """Limpa áudio usando DeepFilterNet (Remove BGM/Ruído)"""
    try:
        output_dir = input_path.parent
        # Executa DeepFilterNet via CLI (mais estável)
        # deepFilter retorna arquivo com sufixo _DeepFilterNet3.wav
        cmd = ["deepFilter", str(input_path), "-o", str(output_dir)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Encontra o arquivo gerado
        cleaned_path = output_dir / f"{input_path.stem}_DeepFilterNet3.wav"
        if cleaned_path.exists():
            return cleaned_path
        return input_path
    except Exception as e:
        logger.error(f"⚠️ Erro no DeepFilter: {e}. Usando áudio original.")
        return input_path

whisper_pipeline = None
def load_turbo_whisper():
    """Carrega Insanely-Fast-Whisper (HF Pipeline + Flash Attention 2)"""
    global whisper_pipeline
    if whisper_pipeline is None:
        try:
            logger.info("🚀 Carregando Whisper V3 com Flash Attention 2...")
            # Força carregamento se não existir
            whisper_pipeline = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3",
                torch_dtype=torch.float16,
                device="cuda:0",
                model_kwargs={"attn_implementation": "flash_attention_2"}
            )
            logger.info("✅ Whisper Turbo Carregado!")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar Whisper Turbo (FA2): {e}")
            logger.warning("⚠️ Tentando fallback sem Flash Attention...")
            try:
                whisper_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-large-v3",
                    torch_dtype=torch.float16,
                    device="cuda:0"
                )
            except Exception as e2:
                logger.error(f"❌ Falha crítica no Whisper: {e2}")

# Cache YOLO
yolo_model = None
def get_yolo():
    global yolo_model
    if yolo_model is None:
        try:
            # Carrega YOLOv8 Nano para detecção de rostos de anime
            yolo_model = YOLO("yolov8n.pt") 
        except:
            yolo_model = None
    return yolo_model

def load_qwen():
    """Carrega Qwen 2.5 da GPU/Volume"""
    global qwen_model, qwen_tokenizer
    if qwen_model is None and AI_AVAILABLE:
        try:
            model_path = str(QWEN_MODEL_PATH)
            
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
                torch_dtype=torch.float16, # Importante para RTX 4090
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
                if total > 0 and downloaded % (20*1024*1024) == 0:
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

# ==================== SENSOR DE ADRENALINA (ActionDetector) ====================
class ActionDetector:
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
        
        # Pula frames para performance (analisa 1 frame a cada 'sample_rate' segundos)
        try:
            step = int(fps * sample_rate)
        except:
            step = 30 # Fallback
            
        if step < 1: step = 1
        
        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret: break
            
            # Converte para escala de cinza e blur para reduzir ruído
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if prev_frame is not None:
                # Diferença absoluta entre frames
                delta = cv2.absdiff(prev_frame, gray)
                thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
                score = np.sum(thresh) / thresh.size # % de pixels que mudaram
                
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
                    # Só guarda se durar mais de 2 segundos
                    if (current_segment["end"] - current_segment["start"]) > 2.0:
                        action_segments.append(current_segment)
                    current_segment = None
                    
        logger.info(f"⚡ {len(action_segments)} cenas de ação intensa detectadas!")
        return action_segments

# ==================== ANTI-SHADOWBAN 2.0 ====================
def apply_antishadowban(clip):
    """Aplica filtros matemáticos para tornar o vídeo único"""
    logger.info("🛡️ Aplicando Anti-Shadowban 2.0...")
    
    # 1. Espelhamento Inteligente (50% de chance)
    if random.choice([True, False]): 
        clip = clip.fx(vfx.mirror_x)
        logger.info("   -> Vídeo Espelhado")

    # 2. Color Grading Aleatório (Sutil)
    gamma_val = random.uniform(0.95, 1.05)
    contrast_val = random.uniform(0.95, 1.05)
    
    # Nota: MoviePy v1 usa fx, v2 usa effects. Vamos usar sintaxe v1 compatível.
    try:
        clip = clip.fx(vfx.gamma_corr, gamma_val).fx(vfx.colorx, contrast_val)
        logger.info(f"   -> Color Grading: Gamma={gamma_val:.2f}, Contrast={contrast_val:.2f}")
    except:
        logger.warning("   -> Falha ao aplicar Color Grading (MoviePy filter error)")

    # 3. Micro-Zoom "Breath" (Respiração)
    # Zoom lento de 1.0 -> 1.03 -> 1.0
    def zoom_effect(get_frame, t):
        img = get_frame(t)
        h, w = img.shape[:2]
        # Ciclo de 5 segundos
        scale = 1 + 0.03 * (0.5 + 0.5 * math.sin(2 * math.pi * t / 5.0))
        
        new_w, new_h = int(w / scale), int(h / scale)
        x1 = (w - new_w) // 2
        y1 = (h - new_h) // 2
        
        cropped = img[y1:y1+new_h, x1:x1+new_w]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    if CV2_AVAILABLE:
        # fl_image aplica efeito frame a frame. Usamos fl para ter o tempo 't'
        clip = clip.fl(zoom_effect)
        logger.info("   -> Micro-Zoom Dinâmico Aplicado")

    return clip

# ==================== IA: ANALISE (TURBO V2) ====================
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
        
        # 1.5 Limpeza de Áudio (DeepFilterNet)
        logger.info("🧹 Limpando áudio (DeepFilterNet)...")
        clean_audio_path = clean_audio_deepfilter(raw_audio_path)
        
        # 2. Transcrição (Insanely Fast Whisper)
        logger.info("🎤 Transcrevendo (Flash Attention 2)...")
        
        # Parâmetros otimizados para FA2
        result = whisper_pipeline(
            str(clean_audio_path),
            chunk_length_s=30,
            batch_size=24, # Aumentar batch size graças ao FA2
            return_timestamps=True,
            generate_kwargs={"language": "portuguese"} 
        )
        
        segments = result.get("chunks", [])
        
        transcript_objs = []
        for seg in segments:
            # O pipeline retorna estrutura (timestamp=(start, end), text="...")
            start_t, end_t = seg["timestamp"]
            transcript_objs.append({
                "start": start_t,
                "end": end_t,
                "text": seg["text"],
                "type": "dialogue"
            })
            
        # 3. RODAR SENSOR DE ADRENALINA ⚡
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
        
        # 3. Análise com Qwen
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
        
        try: os.remove(audio_path)
        except: pass
        
        return viral_cuts
        
    except Exception as e:
        logger.error(f"❌ Erro na análise de IA: {e}")
        return []

# ==================== GERADOR DE TÍTULOS ROBUSTO (ORIGINAL LOCAL) ====================
def hex_to_rgb(hex_color):
    """Converte hex (#RRGGBB) para tupla RGB."""
    if hex_color.startswith('#'):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    else:
        try: return ImageColor.getrgb(hex_color)
        except: return (255, 255, 255)

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
    if not PIL_AVAILABLE: return None
    
    # Cores
    text_color_rgb = hex_to_rgb(text_color)
    stroke_color_rgb = hex_to_rgb(stroke_color)
    
    # Fonte
    if font_size is None:
        font_size = int(largura_video * 0.08)
    
    # Tenta usar a fonte baixada (Impact/Oswald)
    font_to_use = str(FONT_PATH)
    if not os.path.exists(font_to_use):
        font_to_use = "arial.ttf" # Fallback sistema
        
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
            if len(linhas) >= 2: break # Max 2 linhas
            
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
                        draw.text((largura_video / 2 + off_x, y + off_y), linha, font=font, fill=stroke_color_rgb, anchor="mt")
                    except:
                        # Fallback anchor para versoes antigas PIL
                        w_txt = draw.textlength(linha, font=font)
                        draw.text(((largura_video - w_txt) / 2 + off_x, y + off_y), linha, font=font, fill=stroke_color_rgb)

        # Texto Principal
        try:
            draw.text((largura_video / 2, y), linha, font=font, fill=text_color_rgb, anchor="mt")
        except:
             w_txt = draw.textlength(linha, font=font)
             draw.text(((largura_video - w_txt) / 2, y), linha, font=font, fill=text_color_rgb)
        
        # Avança Y
        try:
            bbox = draw.textbbox((0, 0), linha, font=font)
            h = bbox[3] - bbox[1]
        except:
             h = font_size
        y += h + 15

    # MoviePy Clip
    numpy_img = np.array(img)
    clip = ImageClip(numpy_img).set_duration(duracao)
    
    # Posição Relativa
    pos_y = int(altura_video * pos_vertical)
    clip = clip.set_position(('center', pos_y))
    
    return clip

# ==================== RENDERIZAÇÃO NVENC ====================
def processar_corte(video_path: str, cut_data: Dict, num: int, config: Dict) -> str:
    try:
        start = cut_data['start']
        end = cut_data['end']
        
        logger.info(f"🎬 Renderizando Corte {num}: {start:.1f}-{end:.1f} ({config.get('animeName')})")
        
        with VideoFileClip(video_path) as video:
            clip = video.subclip(start, end)
            
            # Anti-Shadowban 2.0 (Upgrade)
            if config.get("antiShadowban"):
                clip = apply_antishadowban(clip)
            
            target_w, target_h = 1080, 1920
            
            # Background
            bg_path = config.get("background_path")
            if bg_path:
                from PIL import Image as PILImage
                bg_img = PILImage.open(bg_path).convert('RGB').resize((target_w, target_h))
                bg_clip = ImageClip(np.array(bg_img)).set_duration(clip.duration)
            else:
                bg_clip = ColorClip(size=(target_w, target_h), color=(15,15,30)).set_duration(clip.duration)
            
            # ZOOM TÁTICO & YOLO SMART CROP (V2 Turbo)
            zoom_factor = 1.15 # 15% Zoom (Ideal)
            w, h = clip.w, clip.h
            new_w = w / zoom_factor
            new_h = h / zoom_factor
            
            # Tenta detectar rosto com YOLO para centralizar crop
            x1, y1 = w/2 - new_w/2, h/2 - new_h/2 # Default (Centro)
            
            try:
                yolo = get_yolo()
                if yolo:
                    # Analisa frame do meio para decidir enquadramento
                    frame = clip.get_frame(clip.duration / 2)
                    results = yolo(frame, verbose=False)
                    
                    # Procura 'person' (class 0) com maior área
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
                        # Centro do rosto
                        face_cx = (best_box[0] + best_box[2]) / 2
                        face_cy = (best_box[1] + best_box[3]) / 2
                        
                        # Calcula x1, y1 para que face_cx, face_cy fiquem no centro do crop
                        x1 = face_cx - (new_w / 2)
                        y1 = face_cy - (new_h / 2)
                        
                        # Limites (Clamping) para não sair do vídeo
                        x1 = max(0, min(x1, w - new_w))
                        y1 = max(0, min(y1, h - new_h))
                        logger.info("🎯 YOLO: Rosto detectado! Ajustando crop.")
            except Exception as e:
                logger.warning(f"⚠️ Erro no Smart Crop: {e}")

            clip_cropped = clip.crop(x1=x1, y1=y1, width=new_w, height=new_h)
            
            # Agora redimensiona para caber na largura (Cover logic)
            scale = target_w / clip_cropped.w 
            
            clip_resized = clip_cropped.resize(scale)
            clip_pos = clip_resized.set_position(('center', 'center'))
            
            layers = [bg_clip, clip_pos]
            
            # Título (Recuperar das configurações ou da IA)
            titulo_texto = cut_data.get("title", config.get("animeName", "")).upper()
            
            if config.get("generateTitles") and titulo_texto:
                logger.info(f"🏷️ Gerando Título: {titulo_texto}")
                t_clip = criar_titulo_pil(
                    titulo_texto, 
                    target_w, target_h, clip.duration,
                    font_filename="impact.ttf",
                    font_size=config.get("titleStyle", {}).get("fontSize", 80),
                    text_color=config.get("titleStyle", {}).get("textColor", "#FFD700"), # Dourado padrão
                    stroke_color="#000000",
                    stroke_width=6,
                    pos_vertical=0.15
                )
                if t_clip: layers.append(t_clip)
            
            final = CompositeVideoClip(layers, size=(target_w, target_h))
            
            output_path = OUTPUT_DIR / f"cut_{num}_{uuid.uuid4().hex[:6]}.mp4"
            
            # --- RENDERIZAÇÃO OTIMIZADA NVENC ---
            logger.info("⚙️ Iniciando Encode (Tentando NVENC)...")
            
            ffmpeg_params = [
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart'
            ]
            
            codec = 'libx264'
            preset = 'ultrafast'
            
            if torch.cuda.is_available():
                logger.info("🚀 Usando Aceleração NVENC (GPU)")
                codec = 'h264_nvenc'
                preset = 'fast' # NVENC fast é muito rapido
                ffmpeg_params.extend([
                   '-rc:v', 'vbr',
                   '-cq:v', '23',
                   '-b:v', '5M',
                   '-maxrate:v', '8M',
                   '-bufsize:v', '10M'
                ])
            else:
                logger.warning("🐢 Usando CPU Encoding (Lento)")
                ffmpeg_params.extend(['-crf', '23'])
            
            final.write_videofile(
                str(output_path),
                codec=codec,
                audio_codec='aac',
                preset=preset,
                threads=4,
                ffmpeg_params=ffmpeg_params,
                logger=None, # Reduz output log
                verbose=False
            )
            
            final.close()
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
        key = f"animecut/v7/{filename}"
        
        logger.info(f"📤 Uploading to B2: {B2_BUCKET}/{key}")
        
        s3_client.upload_file(file_path, B2_BUCKET, key)
        
        url = s3_client.generate_presigned_url(
            'get_object', 
            Params={'Bucket': B2_BUCKET, 'Key': key}, 
            ExpiresIn=86400 # 24 horas
        )
        logger.info(f"✅ Upload Sucesso: {url}")
        return url
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
            
        if not cuts:
            logger.info("⚠️ Fallback para cortes manuais")
            duration = VideoFileClip(video_path).duration
            for i in range(min(5, int(duration/60))):
                cuts.append({
                    "start": i*60, 
                    "end": min((i+1)*60, duration),
                    "title": anime_name # Titulo generico se manual
                })
        
        # 3. Processamento
        results = []
        for i, cut in enumerate(cuts):
            out_path = processar_corte(video_path, cut, i+1, config)
            b2_url = upload_to_b2(out_path)
            
            results.append({
                "path": str(out_path),
                "url": b2_url,
                "title": cut.get("title"),
                "score": cut.get("score", 0)
            })
            
            # Limpa VRAM
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
        try: os.remove(video_path)
        except: pass
        
        return {"status": "success", "cuts": results}
        
    except Exception as e:
        logger.error(f"❌ Erro Handler: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    print("--- INICIANDO WORKER ---")
    sys.stdout.flush()
    runpod.serverless.start({"handler": handler})
