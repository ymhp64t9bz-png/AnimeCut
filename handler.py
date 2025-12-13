#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
✂️ AnimeCut Serverless v12.0 ULTIMATE HYBRID
Stack: Qwen 2.5, Whisper V3 Turbo, YOLOv8, DeepFilterNet, NVENC + MoviePy V2
VOLUME: /workspace (RunPod Persistent Storage)
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

# ==================== CONFIGURAÇÃO DO VOLUME ====================
# PONTO CRÍTICO: Configuração correta do volume RunPod
VOLUME_BASE = "/workspace"  # Volume persistente do RunPod
VOLUME_PATH = Path(VOLUME_BASE)

# Diretórios dentro do volume
TEMP_DIR = Path("/tmp/animecut")  # Temporário na memória RAM (rápido)
OUTPUT_DIR = VOLUME_PATH / "output"  # Saídas no volume (persistente)
MODELS_DIR = VOLUME_PATH / "models"  # Modelos grandes no volume
FONTS_DIR = VOLUME_PATH / "fonts"    # Fontes no volume

# Caminhos específicos de modelos
QWEN_MODEL_PATH = MODELS_DIR / "Qwen2.5-7B-Instruct"
FONT_PATH = FONTS_DIR / "impact.ttf"

# Garante que todos os diretórios existam
for directory in [TEMP_DIR, OUTPUT_DIR, MODELS_DIR, FONTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(VOLUME_PATH / "animecut.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AnimeCutUltimate")
logger.info(f"📁 Volume base configurado: {VOLUME_BASE}")
logger.info(f"📂 Modelos: {MODELS_DIR}")
logger.info(f"📂 Fontes: {FONTS_DIR}")
logger.info(f"📂 Saída: {OUTPUT_DIR}")

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
    logger.warning(f"⚠️ Visão computacional limitada: {e}")

# 2. MoviePy (Detecção de Versão)
MOVIEPY_AVAILABLE = False
MOVIEPY_V2 = False
try:
    import moviepy
    logger.info(f"🎞️ MoviePy versão: {moviepy.__version__}")
    
    if moviepy.__version__.startswith('2'):
        MOVIEPY_V2 = True
        from moviepy import *
        from moviepy.video.io.VideoFileClip import VideoFileClip
        from moviepy.video.VideoClip import ImageClip, ColorClip, TextClip
        from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
        from moviepy.audio.io.AudioFileClip import AudioFileClip
        from moviepy.video.fx import MirrorX, GammaCorr, MultiplyColor
        logger.info("✅ MoviePy v2 configurado")
    else:
        MOVIEPY_V2 = False
        from moviepy.editor import (
            VideoFileClip, ImageClip, CompositeVideoClip,
            ColorClip, TextClip, AudioFileClip
        )
        from moviepy.video.fx.all import mirror_x, gamma_corr, colorx
        logger.info("✅ MoviePy v1 configurado")
        
    MOVIEPY_AVAILABLE = True
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
        gpu_name = torch.cuda.get_device_name(0)
        # gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"✅ GPU: {gpu_name}")
    else:
        logger.warning("⚠️ Executando em CPU (GPU não detectada)")
except ImportError as e:
    logger.warning(f"⚠️ Bibliotecas de IA não disponíveis: {e}")

# 4. Pillow (Imagens)
PIL_AVAILABLE = False
try:
    from PIL import Image, ImageDraw, ImageFont, ImageColor
    PIL_AVAILABLE = True
    logger.info("✅ Pillow disponível")
except ImportError as e:
    logger.warning(f"⚠️ Pillow não disponível: {e}")

# 5. DeepFilterNet (Áudio)
DF_AVAILABLE = False
try:
    import df  # DeepFilterNet instalado como 'df'
    DF_AVAILABLE = True
    logger.info(f"✅ DeepFilterNet disponível (v{df.__version__})")
except ImportError:
    try:
        import deepfilternet
        DF_AVAILABLE = True
        logger.info("✅ DeepFilterNet disponível")
    except ImportError as e:
        logger.warning(f"⚠️ DeepFilterNet não disponível: {e}")

# 6. Backblaze B2 (Upload)
B2_AVAILABLE = False
try:
    import boto3
    from botocore.client import Config
    
    # Credenciais injetadas (Fallback Hardcoded)
    B2_KEY_ID = os.environ.get("B2_KEY_ID", "00568702c2cbfc60000000001")
    B2_APP_KEY = os.environ.get("B2_APPLICATION_KEY", "K005aP6cXPuBIw6IakBaMHYtXx4VGq")
    B2_ENDPOINT = os.environ.get("B2_ENDPOINT", "https://s3.us-east-005.backblazeb2.com")
    B2_BUCKET = os.environ.get("B2_BUCKET_NAME", "KortexAI")
    
    if B2_KEY_ID and B2_APP_KEY:
        s3_client = boto3.client(
            "s3",
            endpoint_url=B2_ENDPOINT,
            aws_access_key_id=B2_KEY_ID,
            aws_secret_access_key=B2_APP_KEY,
            config=Config(signature_version="s3v4")
        )
        B2_AVAILABLE = True
        logger.info(f"✅ Backblaze B2 configurado: {B2_BUCKET}")
    else:
        logger.warning("⚠️ Credenciais B2 não configuradas")
except Exception as e:
    logger.warning(f"⚠️ Backblaze B2 não configurado: {e}")

# ==================== UTILITÁRIOS DE MÍDIA ====================

def download_font():
    """Baixa fonte Impact se não existir no volume"""
    if not FONT_PATH.exists():
        try:
            logger.info(f"📥 Baixando fonte para: {FONT_PATH}")
            url = "https://github.com/google/fonts/raw/main/ofl/oswald/Oswald-Bold.ttf"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            FONT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(FONT_PATH, "wb") as f:
                f.write(response.content)
            logger.info(f"✅ Fonte salva: {FONT_PATH}")
        except Exception as e:
            logger.error(f"❌ Erro ao baixar fonte: {e}")
            # Cria arquivo vazio para evitar erros
            FONT_PATH.touch()
    else:
        logger.info(f"✅ Fonte já existe: {FONT_PATH}")

# Inicializa fonte
download_font()

def clean_audio_deepfilter(input_path: Path) -> Path:
    """
    Limpeza de áudio usando DeepFilterNet (df) com fallback robusto
    """
    logger.info(f"🧹 Processando áudio: {input_path.name}")
    
    original_path = Path(input_path)
    output_dir = original_path.parent
    
    # Método 1: DeepFilterNet CLI (se disponível)
    try:
        deepfilter_cmd = shutil.which("deepFilter") or shutil.which("df")
        if deepfilter_cmd:
            logger.info(f"🔧 Usando DeepFilterNet CLI: {deepfilter_cmd}")
            
            # Executa DeepFilterNet
            cmd = [deepfilter_cmd, str(original_path), "-o", str(output_dir)]
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True,
                timeout=300
            )
            
            # Procura arquivo de saída
            possible_outputs = [
                output_dir / f"{original_path.stem}_DeepFilterNet3.wav",
                output_dir / f"{original_path.stem}_enhanced.wav",
                output_dir / f"{original_path.stem}.wav_enhanced.wav",
                output_dir / f"enhanced_{original_path.name}",
                output_dir / f"{original_path.stem}_df.wav"
            ]
            
            for output_file in possible_outputs:
                if output_file.exists() and output_file.stat().st_size > 0:
                    logger.info(f"✅ Áudio processado: {output_file.name}")
                    return output_file
    except Exception as e:
        logger.warning(f"⚠️ DeepFilterNet CLI falhou: {e}")
    
    # Método 2: FFmpeg fallback (sempre disponível)
    try:
        output_file = output_dir / f"{original_path.stem}_cleaned.wav"
        logger.info(f"🔄 Usando FFmpeg fallback: {output_file.name}")
        
        cmd = [
            'ffmpeg', '-i', str(original_path),
            '-af', 'highpass=f=100,lowpass=f=8000,afftdn=nf=-25',
            '-ar', '16000', '-ac', '1',
            '-acodec', 'pcm_s16le',
            str(output_file), '-y',
            '-hide_banner', '-loglevel', 'error'
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        if output_file.exists() and output_file.stat().st_size > 0:
            logger.info(f"✅ Áudio limpo com FFmpeg")
            return output_file
    except Exception as e:
        logger.error(f"❌ FFmpeg também falhou: {e}")
    
    # Retorna original se tudo falhar
    logger.warning(f"🚨 Retornando áudio original")
    return original_path

def download_video(url: str) -> str:
    """Download robusto de vídeo para diretório temporário"""
    try:
        logger.info(f"📥 Baixando vídeo: {url[:80]}...")
        temp_file = TEMP_DIR / f"input_{uuid.uuid4().hex[:8]}.mp4"
        
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(temp_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0 and downloaded % (50*1024*1024) == 0:
                        percent = (downloaded / total_size) * 100
                        logger.info(f"📥 Download: {percent:.1f}% ({downloaded/1e6:.1f} MB)")
        
        file_size = temp_file.stat().st_size / 1e6
        logger.info(f"✅ Download completo: {temp_file.name} ({file_size:.1f} MB)")
        return str(temp_file)
    except Exception as e:
        logger.error(f"❌ Erro no download: {e}")
        raise

def download_background(url: str) -> Optional[str]:
    """Download de imagem de background"""
    if not url or url.lower() == "none":
        return None
    
    try:
        logger.info(f"🖼️ Baixando background: {url[:80]}...")
        temp_file = TEMP_DIR / f"bg_{uuid.uuid4().hex[:8]}.png"
        
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        with open(temp_file, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"✅ Background salvo: {temp_file.name}")
        return str(temp_file)
    except Exception as e:
        logger.warning(f"⚠️ Erro ao baixar background: {e}")
        return None

# ==================== SENSOR DE ADRENALINA ====================

class ActionDetector:
    """Detecta cenas de ação analisando movimento"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
    
    def calculate_visual_energy(self, sample_rate: float = 1.0) -> List[Dict]:
        """Calcula energia visual baseada em diferença de frames"""
        if not CV2_AVAILABLE:
            return []
        
        try:
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps <= 0 or total_frames <= 0:
                logger.warning("⚠️ Não foi possível ler propriedades do vídeo")
                return []
            
            prev_frame = None
            energy_scores = []
            step = max(1, int(fps * sample_rate))
            
            logger.info(f"⚡ Analisando {total_frames} frames (step={step})")
            
            for frame_idx in range(0, total_frames, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Pré-processamento
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                
                if prev_frame is not None:
                    # Diferença entre frames
                    delta = cv2.absdiff(prev_frame, gray)
                    thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
                    
                    # Calcula score de movimento
                    movement_score = np.sum(thresh) / thresh.size
                    timestamp = frame_idx / fps
                    
                    energy_scores.append({
                        "time": timestamp,
                        "score": float(movement_score)
                    })
                
                prev_frame = gray
            
            cap.release()
            
            # Normaliza scores (0-100)
            if energy_scores:
                max_score = max(s["score"] for s in energy_scores)
                if max_score > 0:
                    for item in energy_scores:
                        item["score"] = (item["score"] / max_score) * 100
            
            logger.info(f"✅ Análise concluída: {len(energy_scores)} pontos")
            return energy_scores
            
        except Exception as e:
            logger.error(f"❌ Erro no sensor de adrenalina: {e}")
            return []
    
    def detect_high_energy_segments(self, threshold: float = 70.0) -> List[Dict]:
        """Identifica segmentos de alta energia (ação)"""
        logger.info(f"⚡ Buscando cenas de ação (threshold={threshold})")
        
        visual_data = self.calculate_visual_energy()
        action_segments = []
        current_segment = None
        
        for data in visual_data:
            is_action = data["score"] > threshold
            
            if is_action:
                if current_segment is None:
                    current_segment = {
                        "start": data["time"],
                        "end": data["time"],
                        "score": data["score"],
                        "peak_score": data["score"]
                    }
                else:
                    current_segment["end"] = data["time"]
                    current_segment["peak_score"] = max(
                        current_segment["peak_score"], 
                        data["score"]
                    )
            else:
                if current_segment:
                    # Só adiciona se durar mais de 2 segundos
                    if (current_segment["end"] - current_segment["start"]) >= 2.0:
                        current_segment["score"] = current_segment["peak_score"]
                        action_segments.append(current_segment)
                    current_segment = None
        
        # Adiciona último segmento se existir
        if current_segment and (current_segment["end"] - current_segment["start"]) >= 2.0:
            current_segment["score"] = current_segment["peak_score"]
            action_segments.append(current_segment)
        
        logger.info(f"✅ {len(action_segments)} cenas de ação detectadas")
        return action_segments

# ==================== ANTI-SHADOWBAN ====================

def apply_antishadowban(clip):
    """Aplica transformações para tornar vídeo único"""
    logger.info("🛡️ Aplicando Anti-Shadowban...")
    
    if not MOVIEPY_AVAILABLE:
        logger.warning("⚠️ MoviePy não disponível, pulando Anti-Shadowban")
        return clip
    
    # 1. Espelhamento aleatório (50% chance)
    if random.choice([True, False]):
        try:
            if MOVIEPY_V2:
                from moviepy.video.fx import MirrorX
                clip = clip.with_effect(MirrorX())
            else:
                clip = clip.fx(mirror_x)
            logger.info("   → Vídeo espelhado")
        except Exception as e:
            logger.warning(f"   → Falha no espelhamento: {e}")
    
    # 2. Ajustes de cor sutis
    gamma_val = random.uniform(0.97, 1.03)
    contrast_val = random.uniform(0.97, 1.03)
    
    try:
        if MOVIEPY_V2:
            from moviepy.video.fx import GammaCorr, MultiplyColor
            clip = clip.with_effect(GammaCorr(gamma_val))
            clip = clip.with_effect(MultiplyColor(contrast_val))
        else:
            clip = clip.fx(gamma_corr, gamma_val)
            clip = clip.fx(colorx, contrast_val)
        logger.info(f"   → Cor: gamma={gamma_val:.2f}, contraste={contrast_val:.2f}")
    except Exception as e:
        logger.warning(f"   → Falha nos ajustes de cor: {e}")
    
    # 3. Zoom dinâmico sutil (se OpenCV disponível)
    if CV2_AVAILABLE and random.choice([True, False]):
        try:
            def zoom_effect(get_frame, t):
                frame = get_frame(t)
                h, w = frame.shape[:2]
                
                # Zoom pulsante sutil
                scale = 1.0 + 0.02 * math.sin(2 * math.pi * t / 4.0)
                
                new_w = int(w / scale)
                new_h = int(h / scale)
                x1 = (w - new_w) // 2
                y1 = (h - new_h) // 2
                
                cropped = frame[y1:y1+new_h, x1:x1+new_w]
                return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            
            if MOVIEPY_V2:
                clip = clip.with_transform(zoom_effect)
            else:
                clip = clip.fl(zoom_effect)
            logger.info("   → Zoom dinâmico aplicado")
        except Exception as e:
            logger.debug(f"   → Zoom dinâmico não aplicado: {e}")
    
    return clip

# ==================== CARREGADORES DE IA ====================

whisper_pipeline = None
qwen_model = None
qwen_tokenizer = None
yolo_model = None

def load_turbo_whisper():
    """Carrega Whisper com Flash Attention 2 se disponível"""
    global whisper_pipeline
    
    if whisper_pipeline is not None or not AI_AVAILABLE:
        return
    
    try:
        logger.info("🚀 Carregando Whisper...")
        
        # Tenta Flash Attention 2 primeiro
        try:
            whisper_pipeline = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3",
                torch_dtype=torch.float16 if GPU_AVAILABLE else torch.float32,
                device=DEVICE,
                model_kwargs={"attn_implementation": "flash_attention_2"}
            )
            logger.info("✅ Whisper com Flash Attention 2")
        except:
            # Fallback para versão normal
            whisper_pipeline = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3",
                torch_dtype=torch.float16 if GPU_AVAILABLE else torch.float32,
                device=DEVICE
            )
            logger.info("✅ Whisper (versão padrão)")
            
    except Exception as e:
        logger.error(f"❌ Erro ao carregar Whisper: {e}")
        whisper_pipeline = None

def load_qwen():
    """Carrega Qwen 2.5 do volume ou HuggingFace"""
    global qwen_model, qwen_tokenizer
    
    if qwen_model is not None or not AI_AVAILABLE:
        return
    
    try:
        model_path = str(QWEN_MODEL_PATH)
        
        # Verifica se o modelo está no volume
        if not os.path.exists(model_path):
            logger.warning(f"⚠️ Modelo Qwen não encontrado no volume: {model_path}")
            logger.info("🌐 Usando modelo leve do HuggingFace...")
            model_path = "Qwen/Qwen2.5-1.5B-Instruct"  # Versão leve para testes
        else:
            logger.info(f"📂 Carregando Qwen do volume: {model_path}")
        
        logger.info("🧠 Carregando Qwen 2.5...")
        
        qwen_tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        qwen_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto" if GPU_AVAILABLE else None,
            torch_dtype=torch.float16 if GPU_AVAILABLE else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        logger.info("✅ Qwen 2.5 carregado")
        
    except Exception as e:
        logger.error(f"❌ Erro ao carregar Qwen: {e}")
        qwen_model = None
        qwen_tokenizer = None

def get_yolo():
    """Carrega YOLO para detecção de rostos"""
    global yolo_model
    
    if yolo_model is not None or not CV2_AVAILABLE:
        return yolo_model
    
    try:
        # Tenta carregar do volume primeiro
        yolo_path = MODELS_DIR / "yolov8n.pt"
        
        if yolo_path.exists():
            logger.info(f"📂 Carregando YOLO do volume: {yolo_path}")
            yolo_model = YOLO(str(yolo_path))
        else:
            logger.info("🌐 Baixando YOLO...")
            yolo_model = YOLO("yolov8n.pt")
            # Salva no volume para uso futuro
            yolo_model.save(yolo_path)
        
        logger.info("✅ YOLO carregado")
        return yolo_model
        
    except Exception as e:
        logger.warning(f"⚠️ Erro ao carregar YOLO: {e}")
        yolo_model = None
        return None

# ==================== GERADOR DE TÍTULOS ====================

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Converte cor hexadecimal para RGB"""
    if hex_color.startswith('#'):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        elif len(hex_color) == 3:
            return tuple(int(hex_color[i]*2, 16) for i in (0, 1, 2))
    
    # Fallback para cores nomeadas
    try:
        return ImageColor.getrgb(hex_color) if PIL_AVAILABLE else (255, 255, 255)
    except:
        return (255, 255, 255)

def criar_titulo_pil(
    texto: str,
    largura_video: int,
    altura_video: int,
    duracao: float,
    font_filename: str = None,
    font_size: int = None,
    text_color: str = "#FFFFFF",
    stroke_color: str = "#000000",
    stroke_width: int = 6,
    pos_vertical: float = 0.15
):
    """
    Renderiza título com quebra de linha e contorno
    """
    if not PIL_AVAILABLE:
        logger.warning("⚠️ Pillow não disponível, pulando título")
        return None
    
    # Configurações padrão
    text_color_rgb = hex_to_rgb(text_color)
    stroke_color_rgb = hex_to_rgb(stroke_color)
    
    if font_size is None:
        font_size = int(largura_video * 0.07)
    
    # Procura fonte no volume
    font_paths = [
        FONT_PATH,
        FONTS_DIR / "Roboto-Bold.ttf",
        FONTS_DIR / "Montserrat-Bold.ttf"
    ]
    
    font_to_use = None
    for path in font_paths:
        if path.exists():
            font_to_use = str(path)
            break
    
    if not font_to_use:
        logger.warning("⚠️ Nenhuma fonte encontrada, usando padrão do sistema")
        font_to_use = "arial.ttf"
    
    try:
        font = ImageFont.truetype(font_to_use, font_size)
    except Exception as e:
        logger.warning(f"⚠️ Erro ao carregar fonte {font_to_use}: {e}")
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Cria canvas para texto
    canvas_h = int(altura_video * 0.3)
    img = Image.new('RGBA', (largura_video, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Quebra de linha inteligente
    palavras = texto.split()
    linhas = []
    linha_atual = []
    
    for palavra in palavras:
        linha_atual.append(palavra)
        texto_teste = " ".join(linha_atual)
        
        try:
            bbox = draw.textbbox((0, 0), texto_teste, font=font)
            largura_texto = bbox[2] - bbox[0]
        except:
            largura_texto = draw.textlength(texto_teste, font=font)
        
        if largura_texto > largura_video * 0.85:
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
    y_pos = 20
    for linha in linhas:
        # Contorno
        for dx in range(-stroke_width, stroke_width + 1):
            for dy in range(-stroke_width, stroke_width + 1):
                if dx == 0 and dy == 0:
                    continue
                try:
                    draw.text(
                        (largura_video // 2 + dx, y_pos + dy),
                        linha,
                        font=font,
                        fill=stroke_color_rgb,
                        anchor="mm"
                    )
                except:
                    largura_linha = draw.textlength(linha, font=font)
                    draw.text(
                        ((largura_video - largura_linha) // 2 + dx, y_pos + dy),
                        linha,
                        font=font,
                        fill=stroke_color_rgb
                    )
        
        # Texto principal
        try:
            draw.text(
                (largura_video // 2, y_pos),
                linha,
                font=font,
                fill=text_color_rgb,
                anchor="mm"
            )
        except:
            largura_linha = draw.textlength(linha, font=font)
            draw.text(
                ((largura_video - largura_linha) // 2, y_pos),
                linha,
                font=font,
                fill=text_color_rgb
            )
        
        # Calcula altura para próxima linha
        try:
            bbox = draw.textbbox((0, 0), linha, font=font)
            altura_linha = bbox[3] - bbox[1]
        except:
            altura_linha = font_size
        
        y_pos += altura_linha + 10
    
    # Converte para clip MoviePy
    numpy_img = np.array(img)
    
    try:
        if MOVIEPY_V2:
            clip = ImageClip(numpy_img).with_duration(duracao)
            pos_y = int(altura_video * pos_vertical)
            clip = clip.with_position(('center', pos_y))
        else:
            clip = ImageClip(numpy_img).set_duration(duracao)
            pos_y = int(altura_video * pos_vertical)
            clip = clip.set_position(('center', pos_y))
        
        return clip
    except Exception as e:
        logger.warning(f"⚠️ Erro ao criar clip de título: {e}")
        return None

# ==================== ANÁLISE DE VÍDEO COM IA ====================

def analyze_video_content(video_path: str, anime_name: str) -> List[Dict]:
    """Analisa vídeo para encontrar cenas virais"""
    
    if not AI_AVAILABLE:
        logger.warning("⚠️ IA não disponível, pulando análise automática")
        return []
    
    try:
        load_turbo_whisper()
        load_qwen()
        
        if not whisper_pipeline:
            raise Exception("Whisper não carregado")
        
        # 1. Extrai áudio do vídeo
        logger.info("🔊 Extraindo áudio...")
        raw_audio_path = TEMP_DIR / f"audio_raw_{uuid.uuid4().hex[:8]}.wav"
        
        ffmpeg_cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            str(raw_audio_path), '-y',
            '-hide_banner', '-loglevel', 'error'
        ]
        
        subprocess.run(ffmpeg_cmd, check=True)
        
        # 2. Limpa áudio
        logger.info("🧹 Limpando áudio...")
        clean_audio_path = clean_audio_deepfilter(raw_audio_path)
        
        # 3. Transcrição com Whisper
        logger.info("🎤 Transcrevendo...")
        result = whisper_pipeline(
            str(clean_audio_path),
            chunk_length_s=30,
            batch_size=16 if GPU_AVAILABLE else 4,
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
                "text": seg["text"].strip(),
                "type": "dialogue"
            })
        
        # 4. Detecção de ação
        logger.info("⚡ Analisando cenas de ação...")
        detector = ActionDetector(video_path)
        action_scenes = detector.detect_high_energy_segments()
        
        for action in action_scenes:
            transcript_objs.append({
                "start": action["start"],
                "end": action["end"],
                "text": f"[⚡ CENA DE AÇÃO - {int(action['score'])}% ENERGIA]",
                "type": "action"
            })
        
        # Ordena por tempo
        transcript_objs.sort(key=lambda x: x["start"])
        
        # Gera texto para Qwen
        full_text = ""
        for item in transcript_objs:
            if item["type"] == "action":
                full_text += f"\n[{item['start']:.1f}s - {item['end']:.1f}s] {item['text']}\n"
            else:
                full_text += f"[{item['start']:.1f}s - {item['end']:.1f}s] {item['text']}\n"
        
        logger.info(f"📝 Transcrição: {len(transcript_objs)} eventos")
        
        # 5. Análise com Qwen (se disponível)
        if not qwen_model or not qwen_tokenizer:
            logger.warning("⚠️ Qwen não disponível, usando heurística simples")
            return generate_fallback_cuts(video_path, transcript_objs, anime_name)
        
        logger.info("🧠 Analisando com Qwen 2.5...")
        
        prompt = f"""Você é um editor especialista em Animes e TikTok.
Analise este conteúdo do anime '{anime_name}' e identifique as 3 melhores cenas para clipes virais (40-90 segundos).

PRIORIDADE:
1. Cenas marcadas com [⚡ CENA DE AÇÃO]
2. Revelações importantes
3. Momentos emocionantes ou engraçados

CONTEÚDO:
{full_text[:20000]}

Retorne APENAS JSON no formato:
[
  {{
    "start": 10.5,
    "end": 65.0,
    "title": "TÍTULO VIRAL CURTO",
    "score": 95
  }}
]"""
        
        inputs = qwen_tokenizer([prompt], return_tensors="pt").to(DEVICE)
        outputs = qwen_model.generate(
            **inputs,
            max_new_tokens=1000,
            temperature=0.7,
            do_sample=True
        )
        
        response_text = qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extrai JSON da resposta
        json_str = response_text
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "[" in response_text and "]" in response_text:
            start_idx = response_text.find("[")
            end_idx = response_text.rfind("]") + 1
            json_str = response_text[start_idx:end_idx]
        
        viral_cuts = json.loads(json_str)
        logger.info(f"🔥 {len(viral_cuts)} cenas virais identificadas")
        
        # Limpeza
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
        return generate_fallback_cuts(video_path, [], anime_name)

def generate_fallback_cuts(video_path: str, transcript: List[Dict], anime_name: str) -> List[Dict]:
    """Gera cortes fallback quando IA falha"""
    
    cuts = []
    
    try:
        if MOVIEPY_V2:
            video = VideoFileClip(video_path)
        else:
            from moviepy.editor import VideoFileClip
            video = VideoFileClip(video_path)
        
        duration = video.duration
        video.close()
        
        # Tenta usar timestamps da transcrição
        if transcript:
            action_scenes = [t for t in transcript if t.get("type") == "action"]
            
            if action_scenes:
                # Usa cenas de ação
                for i, scene in enumerate(action_scenes[:3]):
                    start = max(0, scene["start"] - 5)
                    end = min(duration, scene["end"] + 5)
                    
                    if (end - start) >= 30:  # Mínimo 30 segundos
                        cuts.append({
                            "start": start,
                            "end": end,
                            "title": f"{anime_name} - CENA DE AÇÃO {i+1}",
                            "score": 85
                        })
        
        # Fallback: cortes regulares
        if not cuts:
            num_cuts = min(3, int(duration / 60))
            for i in range(num_cuts):
                start = i * 60
                end = min((i + 1) * 60, duration)
                
                if (end - start) >= 40:  # Mínimo 40 segundos
                    cuts.append({
                        "start": start,
                        "end": end,
                        "title": f"{anime_name} - Parte {i+1}",
                        "score": 70
                    })
        
        # Último fallback: cena do meio
        if not cuts:
            mid_point = duration / 2
            cuts.append({
                "start": max(0, mid_point - 30),
                "end": min(duration, mid_point + 30),
                "title": anime_name,
                "score": 50
            })
        
    except:
        # Fallback extremo
        cuts = [{
            "start": 30,
            "end": 90,
            "title": anime_name,
            "score": 50
        }]
    
    logger.info(f"🔄 Gerados {len(cuts)} cortes fallback")
    return cuts

# ==================== PROCESSAMENTO DE CORTES ====================

def processar_corte(video_path: str, cut_data: Dict, num: int, config: Dict) -> str:
    """Processa um corte individual do vídeo"""
    
    try:
        start = cut_data.get('start', 0)
        end = cut_data.get('end', start + 60)
        title = cut_data.get('title', config.get('animeName', 'Anime'))
        
        logger.info(f"🎬 Processando corte {num}: {title}")
        logger.info(f"   ⏱️  {start:.1f}s - {end:.1f}s ({end-start:.1f}s)")
        
        # Carrega vídeo
        if MOVIEPY_V2:
            video = VideoFileClip(video_path)
        else:
            from moviepy.editor import VideoFileClip
            video = VideoFileClip(video_path)
        
        # Corta segmento
        if hasattr(video, 'subclipped'):
            clip = video.subclipped(start, end)
        else:
            clip = video.subclip(start, end)
        
        # Aplica anti-shadowban
        if config.get("antiShadowban", True):
            clip = apply_antishadowban(clip)
        
        # Configurações de saída
        target_w, target_h = 1080, 1920  # Vertical/TikTok
        
        # Background
        bg_path = config.get("background_path")
        bg_clip = None
        
        if bg_path and os.path.exists(bg_path) and PIL_AVAILABLE:
            try:
                from PIL import Image as PILImage
                bg_img = PILImage.open(bg_path).convert('RGB')
                bg_img = bg_img.resize((target_w, target_h))
                
                if MOVIEPY_V2:
                    bg_clip = ImageClip(np.array(bg_img)).with_duration(clip.duration)
                else:
                    bg_clip = ImageClip(np.array(bg_img)).set_duration(clip.duration)
                    
                logger.info(f"🖼️ Background aplicado: {os.path.basename(bg_path)}")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao carregar background: {e}")
        
        if bg_clip is None:
            # Background gradiente escuro
            bg_color = (15, 15, 30)  # Azul escuro
            if MOVIEPY_V2:
                bg_clip = ColorClip(size=(target_w, target_h), color=bg_color)
                bg_clip = bg_clip.with_duration(clip.duration)
            else:
                bg_clip = ColorClip(size=(target_w, target_h), color=bg_color)
                bg_clip = bg_clip.set_duration(clip.duration)
        
        # Smart Crop com detecção de rosto
        zoom_factor = 1.15
        w, h = clip.w, clip.h
        new_w = w / zoom_factor
        new_h = h / zoom_factor
        
        # Posição inicial (centro)
        x1 = w/2 - new_w/2
        y1 = h/2 - new_h/2
        
        # Tenta detectar rosto para melhor crop
        try:
            yolo = get_yolo()
            if yolo and CV2_AVAILABLE:
                # Analisa frame do meio
                frame_time = clip.duration / 2
                frame = clip.get_frame(frame_time) if hasattr(clip, 'get_frame') else None
                
                if frame is not None:
                    results = yolo(frame, verbose=False)
                    
                    max_area = 0
                    best_box = None
                    
                    for result in results:
                        for box in result.boxes:
                            if int(box.cls) == 0:  # Pessoa
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
                        
                        # Limites
                        x1 = max(0, min(x1, w - new_w))
                        y1 = max(0, min(y1, h - new_h))
                        
                        logger.info("🎯 Crop ajustado para rosto detectado")
        except Exception as e:
            logger.debug(f"Crop inteligente falhou: {e}")
        
        # Aplica crop e resize
        if MOVIEPY_V2:
            clip_cropped = clip.cropped(x1=x1, y1=y1, width=new_w, height=new_h)
            clip_resized = clip_cropped.resized(width=target_w)
            clip_pos = clip_resized.with_position('center')
        else:
            clip_cropped = clip.crop(x1=x1, y1=y1, width=new_w, height=new_h)
            clip_resized = clip_cropped.resize(target_w / clip_cropped.w)
            clip_pos = clip_resized.set_position(('center', 'center'))
        
        # Camadas do vídeo
        layers = [bg_clip, clip_pos]
        
        # Título (se habilitado)
        if config.get("generateTitles", True) and title and PIL_AVAILABLE:
            logger.info(f"🏷️ Adicionando título: {title}")
            
            title_style = config.get("titleStyle", {})
            t_clip = criar_titulo_pil(
                texto=title.upper(),
                largura_video=target_w,
                altura_video=target_h,
                duracao=clip.duration,
                font_size=title_style.get("fontSize", 80),
                text_color=title_style.get("textColor", "#FFD700"),
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
        
        # Gera nome de arquivo único
        output_filename = f"cut_{num}_{uuid.uuid4().hex[:8]}.mp4"
        output_path = OUTPUT_DIR / output_filename
        
        # Configuração de encoding
        ffmpeg_params = [
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart'
        ]
        
        if GPU_AVAILABLE:
            logger.info("🚀 Usando NVENC (GPU)")
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
            logger.info("💻 Usando CPU encoding")
            codec = 'libx264'
            preset = 'ultrafast'
            ffmpeg_params.extend(['-crf', '23'])
        
        # Renderiza vídeo
        logger.info(f"⚙️ Renderizando: {output_filename}")
        
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
        
        file_size = output_path.stat().st_size / 1e6
        logger.info(f"✅ Corte {num} finalizado: {output_filename} ({file_size:.1f} MB)")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"❌ Erro no corte {num}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

# ==================== UPLOAD PARA B2 ====================

def upload_to_b2(file_path: str) -> Optional[str]:
    """Faz upload do arquivo para Backblaze B2"""
    
    if not B2_AVAILABLE:
        logger.warning("⚠️ B2 não configurado, pulando upload")
        return None
    
    try:
        filename = os.path.basename(file_path)
        key = f"animecut/v12/{filename}"
        
        logger.info(f"📤 Upload para B2: {B2_BUCKET}/{key}")
        
        # Upload
        s3_client.upload_file(
            file_path,
            B2_BUCKET,
            key,
            ExtraArgs={'ContentType': 'video/mp4'}
        )
        
        # Gera URL assinada
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': B2_BUCKET, 'Key': key},
            ExpiresIn=86400  # 24 horas
        )
        
        logger.info(f"✅ Upload concluído: {url[:80]}...")
        return url
        
    except Exception as e:
        logger.error(f"❌ Erro no upload B2: {e}")
        return None

# ==================== HANDLER PRINCIPAL ====================

def handler(event):
    """Handler principal do RunPod"""
    
    # Limpeza de memória
    gc.collect()
    if torch and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Log de entrada
    logger.info("=" * 60)
    logger.info("🚀 ANIMECUT ULTIMATE - NOVA REQUISIÇÃO")
    logger.info("=" * 60)
    
    input_data = event.get("input", {})
    
    # Modo teste
    if input_data.get("mode") == "test":
        return {
            "status": "success",
            "system": {
                "gpu": GPU_AVAILABLE,
                "gpu_name": torch.cuda.get_device_name(0) if GPU_AVAILABLE else None,
                "moviepy": MOVIEPY_AVAILABLE,
                "moviepy_version": "v2" if MOVIEPY_V2 else "v1",
                "ai": AI_AVAILABLE,
                "deepfilter": DF_AVAILABLE,
                "b2": B2_AVAILABLE,
                "volume": VOLUME_BASE,
                "python": sys.version.split()[0]
            }
        }
    
    try:
        # Valida entrada
        video_url = input_data.get("video_url")
        if not video_url:
            raise ValueError("video_url é obrigatório")
        
        anime_name = input_data.get("animeName", "Anime")
        
        logger.info(f"🎬 Iniciando processamento: {anime_name}")
        logger.info(f"📹 URL: {video_url[:100]}...")
        
        # 1. Download de recursos
        logger.info("📥 Baixando recursos...")
        video_path = download_video(video_url)
        bg_path = download_background(input_data.get("background_url"))
        
        # Configuração
        config = {
            "animeName": anime_name,
            "antiShadowban": input_data.get("antiShadowban", True),
            "generateTitles": input_data.get("generateTitles", True),
            "titleStyle": input_data.get("titleStyle", {}),
            "background_path": bg_path
        }
        
        # 2. Definição de cortes
        cuts = []
        cut_type = input_data.get("cutType", "auto")
        
        if cut_type == "auto" and AI_AVAILABLE:
            logger.info("🤖 Modo automático (IA)")
            cuts = analyze_video_content(video_path, anime_name)
        elif cut_type == "manual":
            # Cortes manuais fornecidos
            manual_cuts = input_data.get("cuts", [])
            if manual_cuts:
                cuts = manual_cuts
                logger.info(f"✂️ {len(cuts)} cortes manuais fornecidos")
            else:
                logger.warning("⚠️ Modo manual sem cortes, usando automático")
                cuts = analyze_video_content(video_path, anime_name)
        else:
            logger.warning("⚠️ Modo não reconhecido, usando automático")
            cuts = analyze_video_content(video_path, anime_name)
        
        # Fallback se nenhum corte definido
        if not cuts:
            logger.warning("⚠️ Nenhum corte definido, gerando fallback")
            cuts = [{
                "start": 30,
                "end": 90,
                "title": anime_name,
                "score": 50
            }]
        
        logger.info(f"✂️ {len(cuts)} cortes para processar")
        
        # 3. Processamento dos cortes
        results = []
        for i, cut in enumerate(cuts):
            try:
                logger.info(f"🔄 Processando corte {i+1}/{len(cuts)}")
                
                # Processa corte
                out_path = processar_corte(video_path, cut, i+1, config)
                
                # Upload para B2
                b2_url = upload_to_b2(out_path)
                
                # Adiciona resultado
                results.append({
                    "id": i+1,
                    "path": out_path,
                    "url": b2_url,
                    "title": cut.get("title", anime_name),
                    "score": cut.get("score", 0),
                    "start": cut.get("start"),
                    "end": cut.get("end"),
                    "duration": cut.get("end", 0) - cut.get("start", 0)
                })
                
                # Limpeza de memória
                gc.collect()
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info(f"✅ Corte {i+1} concluído")
                
            except Exception as e:
                logger.error(f"❌ Erro no corte {i+1}: {e}")
                continue
        
        # 4. Limpeza de arquivos temporários
        logger.info("🧹 Limpando arquivos temporários...")
        
        try:
            os.remove(video_path)
            logger.info(f"🗑️ Vídeo removido: {os.path.basename(video_path)}")
        except:
            pass
        
        if bg_path and os.path.exists(bg_path):
            try:
                os.remove(bg_path)
                logger.info(f"🗑️ Background removido: {os.path.basename(bg_path)}")
            except:
                pass
        
        # Limpa diretório temporário
        for temp_file in TEMP_DIR.glob("*"):
            try:
                if temp_file.is_file():
                    temp_file.unlink()
            except:
                pass
        
        # 5. Retorna resultados
        logger.info(f"🎉 Processamento concluído: {len(results)} cortes gerados")
        
        return {
            "status": "success",
            "cuts": results,
            "metadata": {
                "anime_name": anime_name,
                "total_cuts": len(results),
                "successful_cuts": len([r for r in results if r.get("url")]),
                "moviepy_version": "v2" if MOVIEPY_V2 else "v1",
                "gpu_used": GPU_AVAILABLE,
                "processing_time": "N/A"  # Poderia adicionar timestamp
            },
            "volume_info": {
                "base_path": VOLUME_BASE,
                "output_dir": str(OUTPUT_DIR),
                "models_dir": str(MODELS_DIR)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Erro no handler: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc() if input_data.get("debug", False) else None
        }

# ==================== INICIALIZAÇÃO ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎬 ANIMECUT ULTIMATE HYBRID v12.0")
    print("📁 Volume: /workspace")
    print("="*60 + "\n")
    
    sys.stdout.flush()
    
    # Inicia servidor RunPod
    runpod.serverless.start({"handler": handler})
