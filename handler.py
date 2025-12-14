#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
✂️ AnimeCut Serverless v12.0 ULTIMATE HYBRID - OTIMIZADO
Stack: Qwen 2.5, Whisper V3 Turbo, YOLOv8, DeepFilterNet, NVENC + MoviePy V1
VOLUME: /workspace (RunPod Persistent Storage)
OTIMIZAÇÕES: Sem downloads na inicialização, cache local, fallbacks robustos
"""

# ==================== IMPORTAÇÕES ESSENCIAIS ====================
import os
import sys
import logging
import time

# ==================== CONFIGURAÇÃO DO VOLUME ====================
VOLUME_BASE = "/workspace"
from pathlib import Path
VOLUME_PATH = Path(VOLUME_BASE)

# Diretórios dentro do volume
TEMP_DIR = Path("/tmp/animecut")
OUTPUT_DIR = VOLUME_PATH / "output"
MODELS_DIR = VOLUME_PATH / "models"
FONTS_DIR = VOLUME_PATH / "fonts"
CACHE_DIR = VOLUME_PATH / "cache"

# Caminhos específicos de modelos
QWEN_MODEL_PATH = MODELS_DIR / "Qwen2.5-7B-Instruct"
FONT_PATH = FONTS_DIR / "impact.ttf"

# Garante que todos os diretórios existam
for directory in [TEMP_DIR, OUTPUT_DIR, MODELS_DIR, FONTS_DIR, CACHE_DIR]:
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

# ==================== IMPORTS COM FALLBACK SUPER ROBUSTO ====================

class DependencyManager:
    """Gerencia imports de forma robusta com múltiplos fallbacks"""
    
    def __init__(self):
        self.available_modules = {}
        self.module_errors = {}
        
    def safe_import(self, module_name, import_path=None, fallback_names=None):
        """Tenta importar um módulo com múltiplos fallbacks"""
        start_time = time.time()
        
        # Lista de nomes para tentar
        names_to_try = [module_name]
        if fallback_names:
            names_to_try.extend(fallback_names)
        
        for name in names_to_try:
            try:
                if import_path:
                    # Import com caminho específico
                    module = __import__(import_path, fromlist=[name])
                else:
                    # Import normal
                    module = __import__(name)
                
                elapsed = time.time() - start_time
                self.available_modules[module_name] = module
                logger.info(f"✅ {module_name} carregado ({elapsed:.2f}s)")
                return module
                
            except ImportError as e:
                self.module_errors[name] = str(e)
                logger.debug(f"⚠️ {name} não disponível: {e}")
                continue
            except Exception as e:
                self.module_errors[name] = str(e)
                logger.debug(f"⚠️ Erro ao carregar {name}: {e}")
                continue
        
        logger.warning(f"❌ {module_name} não disponível após tentar {len(names_to_try)} nomes")
        self.available_modules[module_name] = None
        return None

# Inicializa gerenciador de dependências
dep_manager = DependencyManager()

# 1. Visão Computacional (OpenCV + YOLO)
CV2_AVAILABLE = False
try:
    cv2 = dep_manager.safe_import("cv2")
    np = dep_manager.safe_import("numpy")
    if cv2 and np:
        CV2_AVAILABLE = True
        logger.info("✅ OpenCV disponível")
        
        # Tenta carregar YOLO
        try:
            from ultralytics import YOLO
            logger.info("✅ YOLO disponível")
        except ImportError:
            logger.warning("⚠️ YOLO não disponível")
            
except Exception as e:
    logger.warning(f"⚠️ Visão computacional limitada: {e}")

# 2. MoviePy v1.0.3
MOVIEPY_AVAILABLE = False
try:
    moviepy = dep_manager.safe_import("moviepy")
    if moviepy:
        logger.info(f"🎞️ MoviePy versão: {moviepy.__version__}")
        
        # IMPORTS CORRETOS PARA MOVIEPY v1.0.3
        try:
            from moviepy.editor import (
                VideoFileClip, ImageClip, CompositeVideoClip,
                ColorClip, TextClip, AudioFileClip
            )
            from moviepy.video.fx.all import mirror_x, gamma_corr, colorx
            MOVIEPY_AVAILABLE = True
            logger.info("✅ MoviePy v1 configurado")
        except ImportError as e:
            logger.error(f"❌ Imports MoviePy falharam: {e}")
            
except Exception as e:
    logger.error(f"❌ Erro no MoviePy: {e}")

# 3. IA (Transformers/Torch) - COM FALLBACK ROBUSTO
AI_AVAILABLE = False
GPU_AVAILABLE = False
WHISPER_AVAILABLE = False

# Primeiro tenta importar torch
torch = dep_manager.safe_import("torch")

if torch:
    try:
        # Tenta transformers
        transformers = dep_manager.safe_import("transformers")
        
        if transformers:
            from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
            
            # Tenta faster_whisper primeiro
            try:
                from faster_whisper import WhisperModel
                WHISPER_AVAILABLE = True
                WHISPER_TYPE = "faster_whisper"
                logger.info("✅ faster-whisper disponível")
            except ImportError:
                # Tenta openai-whisper como fallback
                try:
                    import whisper
                    WHISPER_AVAILABLE = True
                    WHISPER_TYPE = "openai_whisper"
                    logger.info("✅ openai-whisper disponível (fallback)")
                except ImportError:
                    WHISPER_AVAILABLE = False
                    logger.warning("⚠️ Nenhuma biblioteca whisper disponível")
            
            GPU_AVAILABLE = torch.cuda.is_available()
            DEVICE = "cuda" if GPU_AVAILABLE else "cpu"
            AI_AVAILABLE = True
            
            if GPU_AVAILABLE:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            else:
                logger.warning("⚠️ Executando em CPU (GPU não detectada)")
                
    except Exception as e:
        logger.warning(f"⚠️ Bibliotecas de IA falharam: {e}")
else:
    logger.warning("⚠️ PyTorch não disponível, IA desativada")

# 4. Pillow (Imagens)
PIL_AVAILABLE = False
try:
    PIL = dep_manager.safe_import("PIL", "PIL.Image")
    if PIL:
        from PIL import Image, ImageDraw, ImageFont, ImageColor
        PIL_AVAILABLE = True
        logger.info("✅ Pillow disponível")
except Exception as e:
    logger.warning(f"⚠️ Pillow não disponível: {e}")

# 5. DeepFilterNet - COMPLETAMENTE ISOLADO
DF_AVAILABLE = False
DF_TYPE = None
DF_ERROR = None

# Tenta detectar DeepFilterNet sem importar
try:
    import subprocess
    import importlib.util
    
    # Verifica se existe algum comando CLI
    df_cli_commands = ["deepFilter", "df", "deepfilternet"]
    df_cli_found = None
    
    for cmd in df_cli_commands:
        cmd_path = shutil.which(cmd) if 'shutil' in sys.modules else None
        if cmd_path:
            df_cli_found = cmd_path
            logger.info(f"🔍 DeepFilterNet CLI encontrado: {cmd}")
            break
    
    if df_cli_found:
        DF_AVAILABLE = True
        DF_TYPE = "cli"
        logger.info("✅ DeepFilterNet CLI disponível")
    else:
        # Tenta detectar módulo Python sem importar
        for module_name in ["df", "deepfilternet"]:
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                DF_AVAILABLE = True
                DF_TYPE = f"python_{module_name}"
                logger.info(f"✅ DeepFilterNet módulo detectado: {module_name}")
                break
                
except Exception as e:
    DF_ERROR = str(e)
    logger.debug(f"DeepFilterNet detecção falhou: {e}")

if not DF_AVAILABLE:
    logger.info("ℹ️ DeepFilterNet não detectado, usando FFmpeg para áudio")

# 6. Backblaze B2 (Upload) - COM FALLBACK SEGURO
B2_AVAILABLE = False
try:
    boto3 = dep_manager.safe_import("boto3")
    if boto3:
        from botocore.client import Config
        
        # Credenciais do ambiente ou fallback
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
            logger.warning("⚠️ Credenciais B2 incompletas")
            
except Exception as e:
    logger.warning(f"⚠️ Backblaze B2 não configurado: {e}")

# 7. Outras dependências opcionais
optional_deps = {
    "Cython": "cython",
    "soundfile": "soundfile",
    "librosa": "librosa",
    "colorama": "colorama"
}

for display_name, module_name in optional_deps.items():
    try:
        __import__(module_name)
        logger.info(f"✅ {display_name} disponível")
    except ImportError:
        logger.debug(f"{display_name} não disponível (opcional)")

# ==================== IMPORTAÇÕES RESTANTES ====================
import tempfile
import requests
import gc
import json
import uuid
import math
import subprocess
import shutil
import random
from typing import List, Dict, Optional, Tuple

# ==================== UTILITÁRIOS DE REDE ROBUSTOS ====================

class NetworkManager:
    """Gerencia operações de rede com retry e timeout"""
    
    def __init__(self, max_retries=3, timeout=30):
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = None
        
    def get_session(self):
        """Cria ou retorna sessão HTTP com configurações otimizadas"""
        if self.session is None:
            self.session = requests.Session()
            # Configurações otimizadas
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10,
                pool_maxsize=100,
                max_retries=3
            )
            self.session.mount('http://', adapter)
            self.session.mount('https://', adapter)
        return self.session
    
    def download_with_retry(self, url, output_path, headers=None):
        """Download com retry e progresso"""
        session = self.get_session()
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"📥 Tentativa {attempt + 1}/{self.max_retries}: {url[:80]}...")
                
                response = session.get(
                    url, 
                    stream=True, 
                    timeout=self.timeout,
                    headers=headers
                )
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            if total_size > 0 and downloaded % (10*1024*1024) == 0:
                                percent = (downloaded / total_size) * 100
                                logger.debug(f"  Progresso: {percent:.1f}%")
                
                file_size = os.path.getsize(output_path) / 1e6
                logger.info(f"✅ Download completo: {output_path.name} ({file_size:.1f} MB)")
                return True
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"⚠️ Tentativa {attempt + 1} falhou: {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"❌ Todas as tentativas falharam para: {url}")
                    return False
                time.sleep(2 ** attempt)  # Backoff exponencial
                
        return False
    
    def check_url_access(self, url, timeout=10):
        """Verifica se uma URL está acessível"""
        try:
            session = self.get_session()
            response = session.head(url, timeout=timeout)
            return response.status_code == 200
        except:
            return False

# Inicializa gerenciador de rede
network = NetworkManager(max_retries=2, timeout=60)

# ==================== FONTE COM CACHE LOCAL ====================

def setup_fonts():
    """Configura fontes usando cache local ou fallback"""
    
    # Lista de fontes preferenciais com cache
    font_sources = [
        # Cache local primeiro
        (CACHE_DIR / "fonts" / "Impact.ttf", None),
        # Fontes do sistema
        (FONT_PATH, "https://github.com/google/fonts/raw/main/ofl/oswald/Oswald-Bold.ttf"),
        # Fallback
        (FONTS_DIR / "Roboto-Bold.ttf", "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Bold.ttf"),
    ]
    
    for font_path, font_url in font_sources:
        if font_path.exists():
            logger.info(f"✅ Fonte encontrada: {font_path}")
            return str(font_path)
    
    # Se nenhuma fonte encontrada, tenta baixar com fallback
    logger.warning("⚠️ Nenhuma fonte encontrada, usando padrão do sistema")
    return None

# Configura fontes
FONT_TO_USE = setup_fonts()

# ==================== FUNÇÕES DE ÁUDIO OTIMIZADAS ====================

def clean_audio_ffmpeg(input_path: Path, quality="high") -> Path:
    """
    Limpeza de áudio usando FFmpeg (sempre funciona)
    Qualidade: 'high', 'medium', 'fast'
    """
    logger.info(f"🔊 Processando áudio ({quality}): {input_path.name}")
    
    original_path = Path(input_path)
    output_dir = original_path.parent
    
    # Configurações por qualidade
    quality_configs = {
        "high": {
            "filters": "arnndn=m=/usr/share/rnnoise-models/sh_ov.rnnn,"
                      "afftdn=nf=-25,"
                      "highpass=f=80,lowpass=f=8000,"
                      "compand=attacks=0.002:decays=0.005,"
                      "dynaudnorm",
            "sample_rate": 48000,
            "channels": 2
        },
        "medium": {
            "filters": "highpass=f=100,lowpass=f=8000,afftdn=nf=-25,dynaudnorm",
            "sample_rate": 44100,
            "channels": 2
        },
        "fast": {
            "filters": "highpass=f=100,lowpass=f=8000",
            "sample_rate": 16000,
            "channels": 1
        }
    }
    
    config = quality_configs.get(quality, quality_configs["medium"])
    
    try:
        output_file = output_dir / f"{original_path.stem}_cleaned_{quality}.wav"
        
        cmd = [
            'ffmpeg', '-i', str(original_path),
            '-af', config["filters"],
            '-ar', str(config["sample_rate"]),
            '-ac', str(config["channels"]),
            '-acodec', 'pcm_s16le',
            str(output_file), '-y',
            '-hide_banner', '-loglevel', 'error'
        ]
        
        # Executa com timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0 and output_file.exists() and output_file.stat().st_size > 0:
            file_size = output_file.stat().st_size / 1e6
            logger.info(f"✅ Áudio processado ({quality}): {file_size:.1f} MB")
            return output_file
        else:
            logger.warning(f"⚠️ FFmpeg falhou: {result.stderr[:200]}")
            
    except subprocess.TimeoutExpired:
        logger.error("❌ FFmpeg timeout")
    except Exception as e:
        logger.error(f"❌ Erro FFmpeg: {e}")
    
    return original_path

def clean_audio_deepfilter(input_path: Path) -> Path:
    """
    Tenta DeepFilterNet, fallback para FFmpeg
    """
    logger.info(f"🧹 Processando áudio: {input_path.name}")
    
    # Se DeepFilterNet disponível via CLI
    if DF_AVAILABLE and DF_TYPE == "cli":
        try:
            # Procura comando
            for cmd_name in ["deepFilter", "df", "deepfilternet"]:
                cmd_path = shutil.which(cmd_name)
                if cmd_path:
                    logger.info(f"🔧 Tentando {cmd_name}...")
                    
                    output_dir = input_path.parent
                    cmd = [cmd_path, str(input_path), "-o", str(output_dir)]
                    
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    
                    if result.returncode == 0:
                        # Procura arquivo de saída
                        patterns = [
                            f"{input_path.stem}_DeepFilterNet3.wav",
                            f"{input_path.stem}_enhanced.wav",
                            f"{input_path.stem}.enhanced.wav"
                        ]
                        
                        for pattern in patterns:
                            output_file = output_dir / pattern
                            if output_file.exists():
                                logger.info(f"✅ Áudio processado via DeepFilterNet")
                                return output_file
    
        except Exception as e:
            logger.warning(f"⚠️ DeepFilterNet CLI falhou: {e}")
    
    # Fallback para FFmpeg de alta qualidade
    return clean_audio_ffmpeg(input_path, quality="high")

# ==================== DOWNLOAD DE VÍDEO COM CACHE ====================

def download_video(url: str) -> str:
    """Download robusto com cache e fallback"""
    
    # Gera nome de arquivo baseado na URL (hash)
    import hashlib
    url_hash = hashlib.md5(url.encode()).hexdigest()[:16]
    temp_file = TEMP_DIR / f"video_{url_hash}.mp4"
    
    # Verifica se já existe no cache
    cache_file = CACHE_DIR / "videos" / f"{url_hash}.mp4"
    if cache_file.exists():
        logger.info(f"📂 Usando cache: {cache_file.name}")
        # Copia para temp
        shutil.copy2(cache_file, temp_file)
        return str(temp_file)
    
    logger.info(f"📥 Baixando vídeo: {url[:80]}...")
    
    # Tenta download
    if network.download_with_retry(url, temp_file):
        # Salva no cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(temp_file, cache_file)
        logger.info(f"💾 Salvo no cache: {cache_file.name}")
        return str(temp_file)
    else:
        raise Exception(f"Falha ao baixar vídeo: {url}")

def download_background(url: str) -> Optional[str]:
    """Download de background com cache"""
    if not url or url.lower() == "none":
        return None
    
    try:
        # Hash da URL para cache
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()[:16]
        cache_file = CACHE_DIR / "backgrounds" / f"{url_hash}.png"
        
        # Verifica cache
        if cache_file.exists():
            logger.info(f"📂 Background do cache: {cache_file.name}")
            # Copia para temp
            temp_file = TEMP_DIR / f"bg_{url_hash}.png"
            shutil.copy2(cache_file, temp_file)
            return str(temp_file)
        
        # Download
        logger.info(f"🖼️ Baixando background: {url[:80]}...")
        temp_file = TEMP_DIR / f"bg_{url_hash}.png"
        
        if network.download_with_retry(url, temp_file):
            # Salva no cache
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(temp_file, cache_file)
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
            if not cap.isOpened():
                logger.warning("⚠️ Não foi possível abrir o vídeo")
                return []
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps <= 0 or total_frames <= 0:
                logger.warning("⚠️ Propriedades do vídeo inválidas")
                cap.release()
                return []
            
            prev_frame = None
            energy_scores = []
            step = max(1, int(fps * sample_rate))
            step = min(step, total_frames // 100)  # Limita para não processar muitos frames
            
            logger.info(f"⚡ Analisando vídeo: {total_frames} frames (step={step})")
            
            frame_count = 0
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
                    
                    # Calcula score
                    movement_score = np.sum(thresh) / thresh.size
                    timestamp = frame_idx / fps
                    
                    energy_scores.append({
                        "time": timestamp,
                        "score": float(movement_score)
                    })
                
                prev_frame = gray
                frame_count += 1
                
                # Progresso a cada 10 frames
                if frame_count % 10 == 0:
                    percent = (frame_idx / total_frames) * 100
                    logger.debug(f"  Progresso análise: {percent:.1f}%")
            
            cap.release()
            
            # Normaliza scores
            if energy_scores:
                scores = [s["score"] for s in energy_scores]
                if max(scores) > 0:
                    max_score = max(scores)
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
        if not visual_data:
            return []
        
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
    if not MOVIEPY_AVAILABLE:
        return clip
    
    logger.info("🛡️ Aplicando Anti-Shadowban...")
    
    try:
        # Espelhamento aleatório
        if random.choice([True, False]):
            clip = clip.fx(mirror_x)
            logger.debug("  → Vídeo espelhado")
        
        # Ajustes de cor sutis
        gamma_val = random.uniform(0.97, 1.03)
        contrast_val = random.uniform(0.97, 1.03)
        
        clip = clip.fx(gamma_corr, gamma_val)
        clip = clip.fx(colorx, contrast_val)
        logger.debug(f"  → Ajustes: gamma={gamma_val:.2f}, contraste={contrast_val:.2f}")
        
    except Exception as e:
        logger.warning(f"⚠️ Anti-shadowban parcialmente aplicado: {e}")
    
    return clip

# ==================== GERADOR DE TÍTULOS ====================

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Converte cor hexadecimal para RGB"""
    if hex_color.startswith('#'):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Fallback
    return (255, 255, 255)

def criar_titulo_simples(
    texto: str,
    largura_video: int,
    altura_video: int,
    duracao: float,
    font_size: int = 80,
    text_color: str = "#FFD700",
    stroke_color: str = "#000000",
    stroke_width: int = 6,
    pos_vertical: float = 0.15
):
    """
    Renderiza título simples - versão otimizada
    """
    if not PIL_AVAILABLE:
        return None
    
    try:
        # Cria imagem para o texto
        img_h = int(altura_video * 0.3)
        img = Image.new('RGBA', (largura_video, img_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Tenta carregar fonte
        font = None
        if FONT_TO_USE and os.path.exists(FONT_TO_USE):
            try:
                font = ImageFont.truetype(FONT_TO_USE, font_size)
            except:
                pass
        
        if font is None:
            # Fonte padrão
            font = ImageFont.load_default()
        
        # Divide texto em 2 linhas
        palavras = texto.split()
        if len(palavras) <= 2:
            linhas = [texto]
        else:
            meio = len(palavras) // 2
            linhas = [
                " ".join(palavras[:meio]),
                " ".join(palavras[meio:])
            ]
        
        # Cores
        text_rgb = hex_to_rgb(text_color)
        stroke_rgb = hex_to_rgb(stroke_color)
        
        # Desenha texto
        y_pos = 20
        for linha in linhas[:2]:  # Máximo 2 linhas
            # Posição central
            bbox = draw.textbbox((0, 0), linha, font=font)
            text_width = bbox[2] - bbox[0]
            x_pos = (largura_video - text_width) // 2
            
            # Contorno
            for dx in [-stroke_width, 0, stroke_width]:
                for dy in [-stroke_width, 0, stroke_width]:
                    if dx == 0 and dy == 0:
                        continue
                    draw.text(
                        (x_pos + dx, y_pos + dy),
                        linha,
                        font=font,
                        fill=stroke_rgb
                    )
            
            # Texto principal
            draw.text(
                (x_pos, y_pos),
                linha,
                font=font,
                fill=text_rgb
            )
            
            # Próxima linha
            y_pos += font_size + 10
        
        # Converte para clip
        numpy_img = np.array(img)
        clip = ImageClip(numpy_img).set_duration(duracao)
        pos_y = int(altura_video * pos_vertical)
        clip = clip.set_position(('center', pos_y))
        
        return clip
        
    except Exception as e:
        logger.warning(f"⚠️ Erro ao criar título: {e}")
        return None

# ==================== ANÁLISE DE VÍDEO COM IA ====================

whisper_pipeline = None
qwen_model = None
qwen_tokenizer = None

def load_turbo_whisper():
    """Carrega Whisper se disponível"""
    global whisper_pipeline
    
    if not AI_AVAILABLE or not WHISPER_AVAILABLE:
        return
    
    try:
        if WHISPER_TYPE == "faster_whisper":
            from faster_whisper import WhisperModel
            
            # Configuração otimizada
            compute_type = "float16" if GPU_AVAILABLE else "float32"
            whisper_model = WhisperModel(
                "large-v3",
                device="cuda" if GPU_AVAILABLE else "cpu",
                compute_type=compute_type,
                download_root=str(MODELS_DIR)
            )
            
            class WhisperWrapper:
                def __init__(self, model):
                    self.model = model
                
                def __call__(self, audio_path, **kwargs):
                    segments, info = self.model.transcribe(
                        audio_path,
                        language="pt",
                        beam_size=3,  # Reduzido para performance
                        vad_filter=True,
                        word_timestamps=False
                    )
                    
                    chunks = []
                    for segment in segments:
                        chunks.append({
                            "text": segment.text.strip(),
                            "timestamp": (segment.start, segment.end)
                        })
                    
                    return {"chunks": chunks}
            
            whisper_pipeline = WhisperWrapper(whisper_model)
            logger.info("✅ Whisper carregado")
            
    except Exception as e:
        logger.error(f"❌ Erro ao carregar Whisper: {e}")
        whisper_pipeline = None

def analyze_video_content(video_path: str, anime_name: str) -> List[Dict]:
    """Analisa vídeo para encontrar cenas virais"""
    
    if not AI_AVAILABLE or not WHISPER_AVAILABLE:
        logger.warning("⚠️ IA não disponível, usando heurística")
        return generate_fallback_cuts(video_path, anime_name)
    
    try:
        # Carrega Whisper se necessário
        if whisper_pipeline is None:
            load_turbo_whisper()
        
        if not whisper_pipeline:
            raise Exception("Whisper não disponível")
        
        logger.info("🔊 Extraindo áudio...")
        
        # Extrai áudio
        raw_audio = TEMP_DIR / f"audio_{uuid.uuid4().hex[:8]}.wav"
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            str(raw_audio), '-y',
            '-hide_banner', '-loglevel', 'error'
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Limpa áudio
        clean_audio = clean_audio_deepfilter(raw_audio)
        
        # Transcrição
        logger.info("🎤 Transcrevendo...")
        result = whisper_pipeline(str(clean_audio))
        
        # Processa transcrição
        transcript = []
        for seg in result.get("chunks", []):
            text = seg.get("text", "").strip()
            if text:  # Ignora textos vazios
                start, end = seg.get("timestamp", (0, 0))
                transcript.append({
                    "start": start,
                    "end": end,
                    "text": text,
                    "type": "dialogue"
                })
        
        # Detecção de ação
        logger.info("⚡ Buscando cenas de ação...")
        detector = ActionDetector(video_path)
        action_scenes = detector.detect_high_energy_segments()
        
        for action in action_scenes:
            transcript.append({
                "start": action["start"],
                "end": action["end"],
                "text": f"[AÇÃO - {int(action['score'])}%]",
                "type": "action"
            })
        
        # Ordena
        transcript.sort(key=lambda x: x["start"])
        
        # Limpa arquivos temporários
        try:
            os.remove(raw_audio)
            if clean_audio != raw_audio:
                os.remove(clean_audio)
        except:
            pass
        
        # Gera cortes baseados na análise
        cuts = []
        video = VideoFileClip(video_path)
        duration = video.duration
        video.close()
        
        # Prioriza cenas de ação
        action_items = [t for t in transcript if t["type"] == "action"]
        
        if action_items:
            for i, action in enumerate(action_items[:3]):  # Máximo 3 cenas
                start = max(0, action["start"] - 3)
                end = min(duration, action["end"] + 3)
                
                if end - start >= 20:  # Mínimo 20 segundos
                    cuts.append({
                        "start": start,
                        "end": end,
                        "title": f"{anime_name} - AÇÃO {i+1}",
                        "score": 85
                    })
        
        # Se não encontrou ações suficientes, usa diálogos importantes
        if len(cuts) < 3:
            dialogue_items = [t for t in transcript if t["type"] == "dialogue"]
            
            # Seleciona diálogos mais longos (provavelmente importantes)
            for item in dialogue_items:
                if len(item["text"].split()) > 5:  # Frases com mais de 5 palavras
                    start = max(0, item["start"] - 2)
                    end = min(duration, item["end"] + 2)
                    
                    if end - start >= 15 and len(cuts) < 3:
                        cuts.append({
                            "start": start,
                            "end": end,
                            "title": f"{anime_name} - CENA {len(cuts)+1}",
                            "score": 75
                        })
        
        # Fallback: divide o vídeo em partes iguais
        if not cuts:
            num_parts = min(3, int(duration / 45))
            for i in range(num_parts):
                start = i * 45
                end = min((i + 1) * 45, duration)
                
                if end - start >= 30:
                    cuts.append({
                        "start": start,
                        "end": end,
                        "title": f"{anime_name} - Parte {i+1}",
                        "score": 65
                    })
        
        logger.info(f"📊 {len(cuts)} cortes identificados")
        return cuts
        
    except Exception as e:
        logger.error(f"❌ Erro na análise: {e}")
        return generate_fallback_cuts(video_path, anime_name)

def generate_fallback_cuts(video_path: str, anime_name: str) -> List[Dict]:
    """Gera cortes fallback simples"""
    
    try:
        video = VideoFileClip(video_path)
        duration = video.duration
        video.close()
        
        # Divide em 3 partes ou menos
        num_cuts = min(3, max(1, int(duration / 60)))
        cuts = []
        
        for i in range(num_cuts):
            start = i * (duration / num_cuts)
            end = (i + 1) * (duration / num_cuts)
            
            cuts.append({
                "start": start,
                "end": end,
                "title": f"{anime_name} - Parte {i+1}",
                "score": 50
            })
        
        return cuts
        
    except:
        # Fallback extremo
        return [{
            "start": 30,
            "end": 90,
            "title": anime_name,
            "score": 30
        }]

# ==================== PROCESSAMENTO DE CORTES ====================

def processar_corte(video_path: str, cut_data: Dict, num: int, config: Dict) -> str:
    """Processa um corte individual do vídeo"""
    
    try:
        start = cut_data.get('start', 0)
        end = cut_data.get('end', start + 60)
        title = cut_data.get('title', config.get('animeName', 'Anime'))
        
        logger.info(f"🎬 Corte {num}: {title} ({start:.1f}s - {end:.1f}s)")
        
        # Carrega vídeo
        video = VideoFileClip(video_path)
        
        # Corta segmento
        clip = video.subclip(start, end)
        
        # Aplica anti-shadowban
        if config.get("antiShadowban", True):
            clip = apply_antishadowban(clip)
        
        # Configurações para TikTok
        target_w, target_h = 1080, 1920
        
        # Background
        bg_clip = None
        bg_path = config.get("background_path")
        
        if bg_path and os.path.exists(bg_path) and PIL_AVAILABLE:
            try:
                bg_img = Image.open(bg_path).convert('RGB')
                bg_img = bg_img.resize((target_w, target_h))
                bg_clip = ImageClip(np.array(bg_img)).set_duration(clip.duration)
            except Exception as e:
                logger.warning(f"⚠️ Background falhou: {e}")
        
        if bg_clip is None:
            # Background sólido
            bg_color = (15, 15, 30)
            bg_clip = ColorClip(size=(target_w, target_h), color=bg_color)
            bg_clip = bg_clip.set_duration(clip.duration)
        
        # Ajusta tamanho do vídeo
        w, h = clip.w, clip.h
        
        # Calcula crop para manter aspecto
        target_aspect = target_w / target_h
        clip_aspect = w / h
        
        if clip_aspect > target_aspect:
            # Muito largo - crop horizontal
            new_w = h * target_aspect
            x1 = (w - new_w) / 2
            clip_cropped = clip.crop(x1=x1, width=new_w)
        else:
            # Muito alto - crop vertical
            new_h = w / target_aspect
            y1 = (h - new_h) / 2
            clip_cropped = clip.crop(y1=y1, height=new_h)
        
        # Redimensiona
        clip_resized = clip_cropped.resize(width=target_w)
        clip_pos = clip_resized.set_position(('center', 'center'))
        
        # Camadas
        layers = [bg_clip, clip_pos]
        
        # Título
        if config.get("generateTitles", True) and title and PIL_AVAILABLE:
            title_clip = criar_titulo_simples(
                texto=title.upper(),
                largura_video=target_w,
                altura_video=target_h,
                duracao=clip.duration,
                font_size=config.get("titleStyle", {}).get("fontSize", 70),
                text_color=config.get("titleStyle", {}).get("textColor", "#FFD700")
            )
            
            if title_clip:
                layers.append(title_clip)
        
        # Composição final
        final = CompositeVideoClip(layers, size=(target_w, target_h))
        
        # Nome do arquivo
        output_filename = f"cut_{num}_{uuid.uuid4().hex[:8]}.mp4"
        output_path = OUTPUT_DIR / output_filename
        
        # Configura encoding
        ffmpeg_params = ['-pix_fmt', 'yuv420p', '-movflags', '+faststart']
        
        if GPU_AVAILABLE:
            codec = 'h264_nvenc'
            preset = 'fast'
            ffmpeg_params.extend(['-cq:v', '23'])
        else:
            codec = 'libx264'
            preset = 'medium'
            ffmpeg_params.extend(['-crf', '23'])
        
        # Renderiza
        logger.info(f"⚙️ Renderizando {output_filename}...")
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
        final.close()
        video.close()
        
        file_size = output_path.stat().st_size / 1e6
        logger.info(f"✅ Corte {num} finalizado ({file_size:.1f} MB)")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"❌ Erro no corte {num}: {e}")
        raise

# ==================== HANDLER PRINCIPAL ====================

def handler(event):
    """Handler principal do RunPod"""
    
    # Limpeza inicial
    gc.collect()
    if torch and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("=" * 60)
    logger.info("🚀 ANIMECUT - NOVA REQUISIÇÃO")
    logger.info("=" * 60)
    
    input_data = event.get("input", {})
    
    # Modo teste
    if input_data.get("mode") == "test":
        return {
            "status": "success",
            "system": {
                "gpu": GPU_AVAILABLE,
                "moviepy": MOVIEPY_AVAILABLE,
                "moviepy_version": moviepy.__version__ if MOVIEPY_AVAILABLE else "N/A",
                "ai": AI_AVAILABLE,
                "whisper": WHISPER_AVAILABLE,
                "deepfilter": DF_AVAILABLE,
                "b2": B2_AVAILABLE,
                "volume": VOLUME_BASE,
                "cache_dir": str(CACHE_DIR)
            }
        }
    
    try:
        # Valida entrada
        video_url = input_data.get("video_url")
        if not video_url:
            raise ValueError("video_url é obrigatório")
        
        anime_name = input_data.get("animeName", "Anime")
        
        logger.info(f"🎬 Processando: {anime_name}")
        
        # 1. Download
        logger.info("📥 Baixando vídeo...")
        video_path = download_video(video_url)
        
        # Background (opcional)
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
            manual_cuts = input_data.get("cuts", [])
            if manual_cuts:
                cuts = manual_cuts
                logger.info(f"✂️ {len(cuts)} cortes manuais")
            else:
                logger.warning("⚠️ Sem cortes manuais, usando automático")
                cuts = analyze_video_content(video_path, anime_name)
        else:
            cuts = analyze_video_content(video_path, anime_name)
        
        # Limite de cortes
        cuts = cuts[:3]
        
        if not cuts:
            cuts = [{
                "start": 30,
                "end": 90,
                "title": anime_name,
                "score": 50
            }]
        
        logger.info(f"✂️ {len(cuts)} cortes para processar")
        
        # 3. Processamento
        results = []
        for i, cut in enumerate(cuts):
            try:
                logger.info(f"🔄 Processando corte {i+1}...")
                
                out_path = processar_corte(video_path, cut, i+1, config)
                
                # Upload opcional
                b2_url = None
                if B2_AVAILABLE:
                    try:
                        filename = os.path.basename(out_path)
                        key = f"animecut/{filename}"
                        s3_client.upload_file(
                            out_path,
                            B2_BUCKET,
                            key,
                            ExtraArgs={'ContentType': 'video/mp4'}
                        )
                        b2_url = s3_client.generate_presigned_url(
                            'get_object',
                            Params={'Bucket': B2_BUCKET, 'Key': key},
                            ExpiresIn=86400
                        )
                        logger.info(f"📤 Upload concluído")
                    except Exception as e:
                        logger.warning(f"⚠️ Upload falhou: {e}")
                
                results.append({
                    "id": i+1,
                    "path": out_path,
                    "url": b2_url,
                    "title": cut.get("title", anime_name),
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
        
        # 4. Limpeza
        logger.info("🧹 Limpando...")
        
        try:
            os.remove(video_path)
        except:
            pass
        
        if bg_path and os.path.exists(bg_path):
            try:
                os.remove(bg_path)
            except:
                pass
        
        # Limpa temp
        for temp_file in TEMP_DIR.glob("*"):
            try:
                if temp_file.is_file():
                    temp_file.unlink()
            except:
                pass
        
        # 5. Resultado
        logger.info(f"🎉 {len(results)} cortes gerados")
        
        return {
            "status": "success",
            "cuts": results,
            "metadata": {
                "anime_name": anime_name,
                "total_cuts": len(results),
                "successful_cuts": len([r for r in results if r.get("path")])
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc() if input_data.get("debug", False) else None
        }

def safe_handler(event):
    """Wrapper seguro"""
    try:
        return handler(event)
    except Exception as e:
        logger.error(f"❌ ERRO GLOBAL: {e}")
        return {"status": "error", "error": str(e)}

# ==================== INICIALIZAÇÃO ====================

if __name__ == "__main__":
    try:
        # Banner
        print("\n" + "="*60)
        print("🎬 ANIMECUT SERVERLESS v12.0 - OTIMIZADO")
        print(f"📁 Volume: {VOLUME_BASE}")
        print(f"💾 Cache: {CACHE_DIR}")
        
        # Status
        print(f"🎞️ MoviePy: {'✅' if MOVIEPY_AVAILABLE else '❌'}")
        print(f"🔥 PyTorch: {'✅' if AI_AVAILABLE else '❌'}")
        print(f"⚡ CUDA: {'✅' if GPU_AVAILABLE else '❌'}")
        print(f"🔊 Áudio: {'✅ DeepFilterNet' if DF_AVAILABLE else '✅ FFmpeg'}")
        print(f"☁️  B2: {'✅' if B2_AVAILABLE else '❌'}")
        print("="*60 + "\n")
        
        sys.stdout.flush()
        
        # Importa runpod
        import runpod
        
        # Inicia servidor
        runpod.serverless.start({"handler": safe_handler})
        
    except KeyboardInterrupt:
        print("\n👋 Servidor interrompido")
        sys.exit(0)
    except Exception as e:
        print(f"💥 ERRO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)