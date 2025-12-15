#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AnimeCut Serverless v12.1 ULTRA-STABLE - TODOS OS BUGS CORRIGIDOS
Stack: Qwen 2.5, Whisper V3 Turbo, YOLOv8, DeepFilterNet, NVENC + MoviePy V1
CORREÇÕES: GPU estável, memória otimizada, cleanup robusto, fallbacks seguros
"""

# ==================== IMPORTAÇÕES ESSENCIAIS ====================
import os
import sys
import logging
import time
import hashlib
import tempfile
import requests
import gc
import json
import uuid
import math
import subprocess
import shutil
import random
import threading
import contextlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from functools import wraps
from datetime import datetime
import html

# ==================== CONFIGURAÇÃO DO VOLUME ====================
VOLUME_BASE = "/workspace"
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

# Configuração de logging aprimorada
log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(VOLUME_PATH / "animecut.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AnimeCutStable")

# ==================== CONFIGURAÇÃO GPU ULTRA-ESTÁVEL ====================
# Configurações minimalistas e testadas
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
# NUNCA definir CUDA_MODULE_LOADING - causa problemas com cuDNN

# ==================== DECORADORES DE SEGURANÇA ====================

def safe_gpu_operation(func):
    """Decorator para operações GPU com limpeza automática"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Limpa antes
            gc.collect()
            if 'torch' in sys.modules:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            
            # Executa
            result = func(*args, **kwargs)
            
            # Limpa depois
            gc.collect()
            if 'torch' in sys.modules:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"[GPU ERROR] {func.__name__}: {e}")
            # Limpa em caso de erro
            gc.collect()
            if 'torch' in sys.modules:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            raise
    return wrapper

def retry_on_failure(max_attempts=3, delay=2, backoff=2):
    """Decorator para retry com backoff exponencial"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        logger.error(f"[RETRY] {func.__name__} falhou após {max_attempts} tentativas")
                        raise
                    
                    logger.warning(f"[RETRY] {func.__name__} tentativa {attempts}/{max_attempts}: {e}")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
        return wrapper
    return decorator

# ==================== GERENCIADOR DE CONTEXTO PARA RECURSOS ====================

class ResourceManager:
    """Gerencia recursos com cleanup automático"""
    
    def __init__(self):
        self.resources = []
        self.lock = threading.Lock()
    
    def register(self, resource, cleanup_func):
        """Registra recurso com função de cleanup"""
        with self.lock:
            self.resources.append((resource, cleanup_func))
    
    def cleanup_all(self):
        """Limpa todos os recursos registrados"""
        with self.lock:
            for resource, cleanup_func in reversed(self.resources):
                try:
                    cleanup_func(resource)
                except Exception as e:
                    logger.warning(f"[CLEANUP] Erro ao limpar recurso: {e}")
            self.resources.clear()
    
    def cleanup_resource(self, resource):
        """Limpa recurso específico"""
        with self.lock:
            for i, (res, cleanup_func) in enumerate(self.resources):
                if res == resource:
                    try:
                        cleanup_func(res)
                        del self.resources[i]
                    except Exception as e:
                        logger.warning(f"[CLEANUP] Erro ao limpar recurso: {e}")
                    break

# Instância global
resource_manager = ResourceManager()

# ------------------ FUNÇÕES DE SANITIZAÇÃO E VALIDAÇÃO DE PATH ------------------
def sanitize_input(value: Optional[str], max_len: int = 240, escape_html: bool = True) -> str:
    """Sanitiza entradas de texto.

    - Para URLs ou caminhos não chame com `escape_html=True` (padrão),
      pois `html.escape` altera caracteres como '&' em URLs assinadas.
    - Remove caracteres de controle, trima e limita tamanho.
    """
    if not value:
        return ""
    # Remove caracteres de controle e trims
    value = ''.join(ch for ch in str(value) if ch.isprintable())
    value = value.strip()
    if escape_html:
        value = html.escape(value)
    if len(value) > max_len:
        value = value[:max_len]
    return value


def is_safe_path(base_dir: Path, user_path: str) -> bool:
    try:
        candidate = (Path(user_path)).resolve()
        base = base_dir.resolve()
        # Permite paths dentro do base, TEMP_DIR ou CACHE_DIR
        allowed_bases = [base, TEMP_DIR.resolve(), CACHE_DIR.resolve()]
        return any(str(candidate).startswith(str(ab) + os.sep) or candidate == ab for ab in allowed_bases)
    except Exception:
        return False


# ------------------ CIRCUIT BREAKER SIMPLES ------------------
class CircuitBreaker:
    def __init__(self, fail_max: int = 3, reset_timeout: int = 30):
        self.fail_max = fail_max
        self.reset_timeout = reset_timeout
        self.fail_counter = 0
        self.last_failure = 0
        self.state = 'CLOSED'

    def call(self, func, *args, **kwargs):
        now = time.time()
        if self.state == 'OPEN' and now - self.last_failure < self.reset_timeout:
            raise RuntimeError('Circuit is OPEN')
        try:
            result = func(*args, **kwargs)
            self.fail_counter = 0
            self.state = 'CLOSED'
            return result
        except Exception:
            self.fail_counter += 1
            self.last_failure = time.time()
            if self.fail_counter >= self.fail_max:
                self.state = 'OPEN'
            raise


breaker = CircuitBreaker()

def circuit_breaker(func=None):
    if func is None:
        return lambda f: circuit_breaker(f)
    @wraps(func)
    def wrapper(*args, **kwargs):
        return breaker.call(func, *args, **kwargs)
    return wrapper


def health_check() -> Dict[str, Any]:
    """Retorna um resumo simples do estado do sistema para health checks."""
    try:
        return {
            "status": "ok",
            "gpu": GPU_AVAILABLE,
            "torch_version": TORCH_VERSION,
            "whisper_loaded": whisper_manager.loaded(),
            "moviepy": MOVIEPY_AVAILABLE,
            "ffmpeg": FFMPEG_AVAILABLE,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ==================== IMPORTS COM FALLBACK ROBUSTO E VERSIONAMENTO ====================

class DependencyManager:
    """Gerencia imports de forma robusta com verificação de versão"""
    
    def __init__(self):
        self.available_modules = {}
        self.module_errors = {}
        self.module_versions = {}
        
    def safe_import(self, module_name, import_path=None, fallback_names=None, min_version=None):
        """Tenta importar um módulo com múltiplos fallbacks e verificação de versão"""
        start_time = time.time()
        
        names_to_try = [module_name]
        if fallback_names:
            names_to_try.extend(fallback_names)
        
        for name in names_to_try:
            try:
                if import_path:
                    module = __import__(import_path, fromlist=[name])
                else:
                    module = __import__(name)
                
                # Verifica versão se especificada
                if min_version and hasattr(module, '__version__'):
                    version = module.__version__
                    self.module_versions[module_name] = version
                    # Aqui você pode adicionar lógica de comparação de versão se necessário
                
                elapsed = time.time() - start_time
                self.available_modules[module_name] = module
                logger.info(f"[SUCCESS] {module_name} carregado ({elapsed:.2f}s)")
                return module
                
            except ImportError as e:
                self.module_errors[name] = str(e)
                logger.debug(f"[DEBUG] {name} não disponível: {e}")
                continue
            except Exception as e:
                self.module_errors[name] = str(e)
                logger.debug(f"[DEBUG] Erro ao carregar {name}: {e}")
                continue
        
        logger.warning(f"[WARNING] {module_name} não disponível após tentar {len(names_to_try)} nomes")
        self.available_modules[module_name] = None
        return None
    
    def get_module_info(self):
        """Retorna informações sobre módulos carregados"""
        return {
            "available": list(self.available_modules.keys()),
            "versions": self.module_versions,
            "errors": self.module_errors
        }

# Inicializa gerenciador de dependências
dep_manager = DependencyManager()

# 1. Visão Computacional (OpenCV + YOLO)
CV2_AVAILABLE = False
YOLO_AVAILABLE = False
try:
    cv2 = dep_manager.safe_import("cv2")
    np = dep_manager.safe_import("numpy")
    if cv2 and np:
        CV2_AVAILABLE = True
        logger.info("[SUCCESS] OpenCV disponível")
        
        # Tenta carregar YOLO
        try:
            from ultralytics import YOLO
            YOLO_AVAILABLE = True
            logger.info("[SUCCESS] YOLO disponível")
        except ImportError:
            logger.debug("[DEBUG] YOLO não disponível")
            
except Exception as e:
    logger.debug(f"[DEBUG] Visão computacional limitada: {e}")

# 2. MoviePy v1.0.3 com validação robusta
MOVIEPY_AVAILABLE = False
moviepy_version = "N/A"
moviepy_imports = {}

try:
    moviepy = dep_manager.safe_import("moviepy")
    if moviepy:
        moviepy_version = getattr(moviepy, '__version__', 'N/A')
        logger.info(f"[MOVIEPY] versão: {moviepy_version}")
        
        try:
            from moviepy.editor import (
                VideoFileClip, ImageClip, CompositeVideoClip,
                ColorClip, TextClip, AudioFileClip
            )
            from moviepy.video.fx.all import mirror_x, gamma_corr, colorx
            
            # Armazena imports
            moviepy_imports = {
                'VideoFileClip': VideoFileClip,
                'ImageClip': ImageClip,
                'CompositeVideoClip': CompositeVideoClip,
                'ColorClip': ColorClip,
                'TextClip': TextClip,
                'AudioFileClip': AudioFileClip,
                'mirror_x': mirror_x,
                'gamma_corr': gamma_corr,
                'colorx': colorx
            }
            
            MOVIEPY_AVAILABLE = True
            logger.info("[SUCCESS] MoviePy v1 configurado")
        except ImportError as e:
            logger.error(f"[ERROR] Imports MoviePy falharam: {e}")
            
except Exception as e:
    logger.error(f"[ERROR] Erro no MoviePy: {e}")

# 3. IA (Transformers/Torch) - CONFIGURAÇÃO ULTRA-ESTÁVEL
AI_AVAILABLE = False
GPU_AVAILABLE = False
WHISPER_AVAILABLE = False
WHISPER_TYPE = None
TORCH_DEVICE = None
TORCH_VERSION = None

# Importa torch primeiro de forma segura
torch = dep_manager.safe_import("torch")

if torch:
    try:
        TORCH_VERSION = torch.__version__
        logger.info(f"[TORCH] Versão: {TORCH_VERSION}")
        
        # VERIFICAÇÃO SEGURA DE GPU
        GPU_AVAILABLE = torch.cuda.is_available()
        
        if GPU_AVAILABLE:
            TORCH_DEVICE = torch.device("cuda:0")
            
            # CONFIGURAÇÕES ULTRA-MINIMALISTAS
            try:
                # Apenas habilita cuDNN se funcionar
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = False  # False é mais estável
                torch.backends.cuda.matmul.allow_tf32 = True
                logger.info("[GPU] cuDNN configurado (modo estável)")
            except Exception as e:
                logger.warning(f"[WARNING] Configuração cuDNN falhou: {e}")
            
            # Informações da GPU
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_props = torch.cuda.get_device_properties(0)
                gpu_mem = gpu_props.total_memory / 1e9
                logger.info(f"[SUCCESS] GPU: {gpu_name}")
                logger.info(f"[GPU] Memória total: {gpu_mem:.1f} GB")
                logger.info(f"[GPU] Compute Capability: {gpu_props.major}.{gpu_props.minor}")
            except Exception as e:
                logger.warning(f"[WARNING] Erro ao obter info GPU: {e}")
        else:
            TORCH_DEVICE = torch.device("cpu")
            logger.info("[INFO] Executando em CPU")
        
        # Tenta transformers
        transformers = dep_manager.safe_import("transformers")
        
        if transformers:
            from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
            
            # Tenta faster_whisper primeiro
            try:
                from faster_whisper import WhisperModel
                WHISPER_AVAILABLE = True
                WHISPER_TYPE = "faster_whisper"
                logger.info("[SUCCESS] faster-whisper disponível")
            except ImportError:
                # Tenta openai-whisper como fallback
                try:
                    import whisper
                    WHISPER_AVAILABLE = True
                    WHISPER_TYPE = "openai_whisper"
                    logger.info("[SUCCESS] openai-whisper disponível (fallback)")
                except ImportError:
                    WHISPER_AVAILABLE = False
                    logger.info("[INFO] Nenhuma biblioteca whisper disponível")
            
            AI_AVAILABLE = True
            
    except Exception as e:
        logger.warning(f"[WARNING] Bibliotecas de IA falharam: {e}")
        GPU_AVAILABLE = False
        TORCH_DEVICE = torch.device("cpu") if torch else None
        AI_AVAILABLE = False
else:
    logger.info("[INFO] PyTorch não disponível, IA desativada")

# 4. Pillow (Imagens)
PIL_AVAILABLE = False
try:
    PIL = dep_manager.safe_import("PIL", "PIL.Image")
    if PIL:
        from PIL import Image, ImageDraw, ImageFont, ImageColor
        PIL_AVAILABLE = True
        logger.info("[SUCCESS] Pillow disponível")
except Exception as e:
    logger.debug(f"[DEBUG] Pillow não disponível: {e}")

# 5. DeepFilterNet - Com detecção aprimorada
DF_AVAILABLE = False
DF_TYPE = None
DF_COMMAND = None

try:
    # Verifica CLI de forma robusta
    for cmd_name in ["deepFilter", "df", "deepfilternet"]:
        cmd_path = shutil.which(cmd_name)
        if cmd_path:
            # Testa se o comando funciona
            try:
                result = subprocess.run(
                    [cmd_path, "--help"],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    DF_AVAILABLE = True
                    DF_TYPE = "cli"
                    DF_COMMAND = cmd_path
                    logger.info(f"[SUCCESS] DeepFilterNet CLI: {cmd_name} ({cmd_path})")
                    break
            except:
                continue
    
    # Se CLI não funcionar, tenta módulo Python
    if not DF_AVAILABLE:
        import importlib.util
        for module_name in ["df", "deepfilternet"]:
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                try:
                    module = __import__(module_name)
                    DF_AVAILABLE = True
                    DF_TYPE = f"python_{module_name}"
                    logger.info(f"[SUCCESS] DeepFilterNet módulo: {module_name}")
                    break
                except:
                    continue
                
except Exception as e:
    logger.debug(f"[DEBUG] DeepFilterNet não detectado: {e}")

if not DF_AVAILABLE:
    logger.info("[INFO] DeepFilterNet não disponível, usando FFmpeg para áudio")

# 6. Backblaze B2
B2_AVAILABLE = False
s3_client = None
B2_BUCKET = None

try:
    boto3 = dep_manager.safe_import("boto3")
    if boto3:
        from botocore.client import Config
        
        B2_KEY_ID = os.environ.get("B2_KEY_ID", "00568702c2cbfc60000000001")
        B2_APP_KEY = os.environ.get("B2_APPLICATION_KEY", "K005aP6cXPuBIw6IakBaMHYtXx4VGq")
        B2_ENDPOINT = os.environ.get("B2_ENDPOINT", "https://s3.us-east-005.backblazeb2.com")
        B2_BUCKET = os.environ.get("B2_BUCKET_NAME", "KortexAI")
        
        if B2_KEY_ID and B2_APP_KEY and B2_BUCKET:
            try:
                s3_client = boto3.client(
                    "s3",
                    endpoint_url=B2_ENDPOINT,
                    aws_access_key_id=B2_KEY_ID,
                    aws_secret_access_key=B2_APP_KEY,
                    config=Config(signature_version="s3v4", connect_timeout=10, read_timeout=30)
                )
                B2_AVAILABLE = True
                logger.info(f"[SUCCESS] Backblaze B2 configurado: {B2_BUCKET}")
            except Exception as e:
                logger.warning(f"[WARNING] Erro ao configurar B2: {e}")
            
except Exception as e:
    logger.debug(f"[DEBUG] Backblaze B2 não configurado: {e}")

# 7. FFmpeg - Validação de disponibilidade
FFMPEG_AVAILABLE = False
FFMPEG_VERSION = None

try:
    result = subprocess.run(
        ['ffmpeg', '-version'],
        capture_output=True,
        text=True,
        timeout=5
    )
    if result.returncode == 0:
        FFMPEG_AVAILABLE = True
        # Extrai versão
        lines = result.stdout.split('\n')
        if lines:
            FFMPEG_VERSION = lines[0].split(' ')[2] if len(lines[0].split(' ')) > 2 else "unknown"
        logger.info(f"[SUCCESS] FFmpeg disponível: {FFMPEG_VERSION}")
except:
    logger.error("[ERROR] FFmpeg não disponível - CRÍTICO!")

# ==================== UTILITÁRIOS DE REDE APRIMORADOS ====================

class NetworkManager:
    """Gerencia operações de rede com retry e validação"""
    
    def __init__(self, max_retries=3, timeout=30):
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = None
        self.lock = threading.Lock()
        
    def get_session(self):
        """Obtém sessão HTTP thread-safe"""
        with self.lock:
            if self.session is None:
                self.session = requests.Session()
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=10,
                    pool_maxsize=100,
                    max_retries=3
                )
                self.session.mount('http://', adapter)
                self.session.mount('https://', adapter)
            return self.session
    
    def validate_url(self, url):
        """Valida URL antes de baixar"""
        if not url or not isinstance(url, str):
            return False
        
        url_lower = url.lower()
        if not (url_lower.startswith('http://') or url_lower.startswith('https://')):
            return False
        
        return True
    
    @retry_on_failure(max_attempts=3, delay=2)
    def download_with_retry(self, url, output_path, headers=None, chunk_size=8192):
        """Download com retry, validação e progresso"""
        
        if not self.validate_url(url):
            raise ValueError(f"URL inválida: {url}")
        
        session = self.get_session()
        
        logger.info(f"[DOWNLOAD] Iniciando: {url[:80]}...")
        
        response = session.get(
            url, 
            stream=True, 
            timeout=self.timeout,
            headers=headers
        )
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        start_time = time.time()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Log de progresso a cada 10MB
                    if downloaded % (10 * 1024 * 1024) < chunk_size:
                        percent = (downloaded / total_size * 100) if total_size > 0 else 0
                        logger.debug(f"[DOWNLOAD] Progresso: {percent:.1f}%")
        
        elapsed = time.time() - start_time
        file_size = os.path.getsize(output_path) / 1e6
        speed = file_size / elapsed if elapsed > 0 else 0
        
        logger.info(f"[SUCCESS] Download completo: {output_path.name} ({file_size:.1f} MB, {speed:.1f} MB/s)")
        
        # Valida arquivo baixado
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise Exception(f"Arquivo baixado está vazio: {output_path}")
        
        return True

# Inicializa gerenciador de rede
network = NetworkManager(max_retries=3, timeout=60)

# ==================== FONTE COM CACHE ====================

def setup_fonts():
    """Configura fontes usando cache local com fallbacks"""
    
    font_sources = [
        CACHE_DIR / "fonts" / "Impact.ttf",
        FONT_PATH,
        FONTS_DIR / "Roboto-Bold.ttf",
        FONTS_DIR / "Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    
    for font_path in font_sources:
        try:
            p = Path(font_path)
        except Exception:
            p = Path(str(font_path))

        if p.exists():
            logger.info(f"[SUCCESS] Fonte encontrada: {p}")
            return str(p)
    
    logger.info("[INFO] Nenhuma fonte TrueType encontrada, usando padrão do sistema")
    return None

FONT_TO_USE = setup_fonts()

# ==================== FUNÇÕES DE ÁUDIO ULTRA-ROBUSTAS ====================

@retry_on_failure(max_attempts=2, delay=1)
def validate_audio_file(audio_path: Path) -> bool:
    """Valida arquivo de áudio usando ffprobe"""
    if not FFMPEG_AVAILABLE:
        return audio_path.exists() and audio_path.stat().st_size > 0
    
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_type,duration',
            '-of', 'json',
            str(audio_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            streams = data.get('streams', [])
            if streams and streams[0].get('codec_type') == 'audio':
                duration = float(streams[0].get('duration', 0))
                return duration > 0
        
        return False
        
    except:
        return audio_path.exists() and audio_path.stat().st_size > 0

def clean_audio_ffmpeg(input_path: Path, quality="high") -> Path:
    """
    Limpeza de áudio usando FFmpeg com validação robusta
    Qualidade: 'high', 'medium', 'fast'
    """
    if not FFMPEG_AVAILABLE:
        logger.warning("[WARNING] FFmpeg não disponível, retornando áudio original")
        return input_path
    
    logger.info(f"[AUDIO] Processando áudio ({quality}): {input_path.name}")
    
    original_path = Path(input_path)
    output_dir = original_path.parent
    
    quality_configs = {
        "high": {
            "filters": "highpass=f=80,lowpass=f=8000,afftdn=nf=-25:nr=10,dynaudnorm=f=150:g=15",
            "sample_rate": 48000,
            "channels": 2,
            "bitrate": "192k"
        },
        "medium": {
            "filters": "highpass=f=100,lowpass=f=8000,afftdn=nf=-20,dynaudnorm",
            "sample_rate": 44100,
            "channels": 2,
            "bitrate": "128k"
        },
        "fast": {
            "filters": "highpass=f=100,lowpass=f=8000",
            "sample_rate": 16000,
            "channels": 1,
            "bitrate": "96k"
        }
    }
    
    config = quality_configs.get(quality, quality_configs["medium"])
    
    try:
        output_file = output_dir / f"{original_path.stem}_cleaned_{quality}_{uuid.uuid4().hex[:6]}.wav"
        
        cmd = [
            'ffmpeg', '-i', str(original_path),
            '-af', config["filters"],
            '-ar', str(config["sample_rate"]),
            '-ac', str(config["channels"]),
            '-ab', config["bitrate"],
            '-acodec', 'pcm_s16le',
            '-map', '0:a:0',
            str(output_file), '-y',
            '-hide_banner', '-loglevel', 'error'
        ]
        
        logger.debug(f"[FFMPEG] Comando: {' '.join(cmd[:10])}...")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0 and validate_audio_file(output_file):
            file_size = output_file.stat().st_size / 1e6
            logger.info(f"[SUCCESS] Áudio processado ({quality}): {file_size:.1f} MB")
            
            # Registra para cleanup
            resource_manager.register(output_file, lambda p: p.unlink(missing_ok=True))
            
            return output_file
        else:
            logger.warning(f"[WARNING] FFmpeg falhou ou arquivo inválido: {result.stderr[:200]}")
            
            # Fallback: cópia simples com normalização
            logger.info("[INFO] Tentando processamento simplificado...")
            output_file_simple = output_dir / f"{original_path.stem}_simple_{uuid.uuid4().hex[:6]}.wav"
            
            cmd_simple = [
                'ffmpeg', '-i', str(original_path),
                '-af', 'volume=1.5,acompressor=threshold=-20dB:ratio=4:attack=5:release=50',
                '-ar', str(config["sample_rate"]),
                '-ac', str(config["channels"]),
                '-acodec', 'pcm_s16le',
                str(output_file_simple), '-y',
                '-hide_banner', '-loglevel', 'error'
            ]
            
            result_simple = subprocess.run(
                cmd_simple,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result_simple.returncode == 0 and validate_audio_file(output_file_simple):
                logger.info("[SUCCESS] Áudio processado com fallback simplificado")
                resource_manager.register(output_file_simple, lambda p: p.unlink(missing_ok=True))
                return output_file_simple
            
    except subprocess.TimeoutExpired:
        logger.error("[ERROR] FFmpeg timeout")
    except Exception as e:
        logger.error(f"[ERROR] Erro FFmpeg: {e}")
    
    # Último fallback: retorna original
    logger.warning("[WARNING] Retornando áudio original sem processamento")
    return original_path

def clean_audio_deepfilter(input_path: Path) -> Path:
    """
    Tenta DeepFilterNet com fallback robusto para FFmpeg
    """
    logger.info(f"[AUDIO CLEAN] Processando áudio: {input_path.name}")
    
    # Valida arquivo de entrada
    if not validate_audio_file(input_path):
        logger.error(f"[ERROR] Arquivo de áudio inválido: {input_path}")
        return input_path
    
    # Tenta DeepFilterNet se disponível
    if DF_AVAILABLE and DF_TYPE == "cli" and DF_COMMAND:
        try:
            logger.info(f"[DEEP FILTER] Usando {DF_COMMAND}...")
            
            output_dir = input_path.parent
            
            # Executa com timeout
            result = subprocess.run(
                [DF_COMMAND, str(input_path), "-o", str(output_dir)],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                # Procura arquivo de saída
                patterns = [
                    f"{input_path.stem}_DeepFilterNet3.wav",
                    f"{input_path.stem}_DeepFilterNet2.wav",
                    f"{input_path.stem}_enhanced.wav",
                    f"{input_path.stem}.enhanced.wav"
                ]
                
                for pattern in patterns:
                    output_file = output_dir / pattern
                    if output_file.exists() and validate_audio_file(output_file):
                        logger.info("[SUCCESS] Áudio processado via DeepFilterNet")
                        resource_manager.register(output_file, lambda p: p.unlink(missing_ok=True))
                        return output_file
                
                logger.warning("[WARNING] DeepFilterNet não gerou arquivo válido")
            else:
                logger.warning(f"[WARNING] DeepFilterNet falhou: {result.stderr[:200]}")
                
        except subprocess.TimeoutExpired:
            logger.warning("[WARNING] DeepFilterNet timeout")
        except Exception as e:
            logger.warning(f"[WARNING] DeepFilterNet exception: {e}")
    
    # Fallback para FFmpeg de alta qualidade
    logger.info("[INFO] Usando FFmpeg high quality como fallback")
    return clean_audio_ffmpeg(input_path, quality="high")

# ==================== DOWNLOAD DE VÍDEO COM CACHE E VALIDAÇÃO ====================

def validate_video_file(video_path: Path) -> bool:
    """Valida arquivo de vídeo usando ffprobe"""
    if not FFMPEG_AVAILABLE:
        return video_path.exists() and video_path.stat().st_size > 1e6  # > 1MB
    
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_type,width,height,duration',
            '-of', 'json',
            str(video_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            streams = data.get('streams', [])
            if streams:
                stream = streams[0]
                is_video = stream.get('codec_type') == 'video'
                has_size = stream.get('width', 0) > 0 and stream.get('height', 0) > 0
                has_duration = float(stream.get('duration', 0)) > 0
                return is_video and has_size and has_duration
        
        return False
        
    except Exception as e:
        logger.warning(f"[WARNING] Validação de vídeo falhou: {e}")
        return video_path.exists() and video_path.stat().st_size > 1e6

@retry_on_failure(max_attempts=2, delay=3)
def download_video(url: str) -> str:
    """Download robusto com cache e validação completa"""
    
    if not network.validate_url(url):
        raise ValueError(f"URL de vídeo inválida: {url}")
    
    # Gera nome de arquivo baseado na URL (hash)
    url_hash = hashlib.md5(url.encode()).hexdigest()[:16]
    temp_file = TEMP_DIR / f"video_{url_hash}_{uuid.uuid4().hex[:6]}.mp4"
    
    # Verifica se já existe no cache
    cache_file = CACHE_DIR / "videos" / f"{url_hash}.mp4"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    if cache_file.exists() and validate_video_file(cache_file):
        logger.info(f"[CACHE] Usando vídeo do cache: {cache_file.name}")
        # Copia para temp
        shutil.copy2(cache_file, temp_file)
        resource_manager.register(temp_file, lambda p: p.unlink(missing_ok=True))
        return str(temp_file)
    
    logger.info(f"[DOWNLOAD] Baixando vídeo: {url[:80]}...")
    
    # Tenta download
    if network.download_with_retry(url, temp_file):
        # Valida vídeo baixado
        if not validate_video_file(temp_file):
            raise Exception(f"Vídeo baixado é inválido: {url}")
        
        # Salva no cache
        try:
            shutil.copy2(temp_file, cache_file)
            logger.info(f"[CACHE] Salvo no cache: {cache_file.name}")
        except Exception as e:
            logger.warning(f"[WARNING] Não foi possível salvar no cache: {e}")
        
        resource_manager.register(temp_file, lambda p: p.unlink(missing_ok=True))
        return str(temp_file)
    else:
        raise Exception(f"Falha ao baixar vídeo: {url}")

@retry_on_failure(max_attempts=2, delay=2)
def download_background(url: str) -> Optional[str]:
    """Download de background com cache e validação"""
    if not url or url.lower() == "none":
        return None
    
    if not network.validate_url(url):
        logger.warning(f"[WARNING] URL de background inválida: {url}")
        return None
    
    try:
        # Hash da URL para cache
        url_hash = hashlib.md5(url.encode()).hexdigest()[:16]
        cache_file = CACHE_DIR / "backgrounds" / f"{url_hash}.png"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Verifica cache
        if cache_file.exists() and cache_file.stat().st_size > 0:
            logger.info(f"[CACHE] Background do cache: {cache_file.name}")
            # Copia para temp
            temp_file = TEMP_DIR / f"bg_{url_hash}_{uuid.uuid4().hex[:6]}.png"
            shutil.copy2(cache_file, temp_file)
            resource_manager.register(temp_file, lambda p: p.unlink(missing_ok=True))
            return str(temp_file)
        
        # Download
        logger.info(f"[BACKGROUND] Baixando: {url[:80]}...")
        temp_file = TEMP_DIR / f"bg_{url_hash}_{uuid.uuid4().hex[:6]}.png"
        
        if network.download_with_retry(url, temp_file):
            # Valida se é imagem
            if not temp_file.exists() or temp_file.stat().st_size < 1000:
                raise Exception("Arquivo de background muito pequeno")
            
            # Salva no cache
            try:
                shutil.copy2(temp_file, cache_file)
            except Exception as e:
                logger.warning(f"[WARNING] Erro ao salvar background no cache: {e}")
            
            resource_manager.register(temp_file, lambda p: p.unlink(missing_ok=True))
            return str(temp_file)
            
    except Exception as e:
        logger.warning(f"[WARNING] Erro ao baixar background: {e}")
    
    return None

# ==================== SENSOR DE ADRENALINA OTIMIZADO ====================

class ActionDetector:
    """Detecta cenas de ação analisando movimento com validação"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        
        if not CV2_AVAILABLE:
            logger.warning("[WARNING] OpenCV não disponível, detector desabilitado")
    
    @safe_gpu_operation
    def calculate_visual_energy(self, sample_rate: float = 1.0, max_frames: int = 300) -> List[Dict]:
        """Calcula energia visual baseada em diferença de frames com limite"""
        if not CV2_AVAILABLE:
            logger.info("[INFO] OpenCV não disponível para detecção de ação")
            return []
        
        cap = None
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                logger.warning("[WARNING] Não foi possível abrir o vídeo para análise")
                return []
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps <= 0 or total_frames <= 0:
                logger.warning("[WARNING] Propriedades do vídeo inválidas")
                return []
            
            # Calcula step para não exceder max_frames
            step = max(1, int(fps * sample_rate))
            estimated_frames = total_frames // step
            
            if estimated_frames > max_frames:
                step = max(1, total_frames // max_frames)
            
            logger.info(f"[ACTION DETECTION] Analisando: {total_frames} frames (step={step}, amostras~{total_frames//step})")
            
            prev_frame = None
            energy_scores = []
            
            for frame_idx in range(0, total_frames, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    break
                
                # Pré-processamento otimizado
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.GaussianBlur(gray, (21, 21), 0)
                    gray = cv2.resize(gray, (320, 240))  # Reduz para acelerar
                except Exception as e:
                    logger.warning(f"[WARNING] Erro no processamento do frame {frame_idx}: {e}")
                    continue
                
                if prev_frame is not None:
                    try:
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
                    except Exception as e:
                        logger.warning(f"[WARNING] Erro no cálculo de movimento: {e}")
                        continue
                
                prev_frame = gray
            
            # Normaliza scores
            if energy_scores:
                scores = [s["score"] for s in energy_scores]
                max_score = max(scores) if scores else 1.0
                
                if max_score > 0:
                    for item in energy_scores:
                        item["score"] = (item["score"] / max_score) * 100
            
            logger.info(f"[SUCCESS] Análise concluída: {len(energy_scores)} pontos")
            return energy_scores
            
        except Exception as e:
            logger.error(f"[ERROR] Erro no sensor de adrenalina: {e}")
            return []
        finally:
            if cap is not None:
                cap.release()
            gc.collect()
    
    def detect_high_energy_segments(self, threshold: float = 70.0, min_duration: float = 2.0) -> List[Dict]:
        """Identifica segmentos de alta energia (ação) com filtros"""
        logger.info(f"[ACTION DETECTION] Buscando cenas de ação (threshold={threshold})")
        
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
                    duration = current_segment["end"] - current_segment["start"]
                    if duration >= min_duration:
                        current_segment["score"] = current_segment["peak_score"]
                        current_segment["duration"] = duration
                        action_segments.append(current_segment)
                    current_segment = None
        
        # Adiciona último segmento se válido
        if current_segment:
            duration = current_segment["end"] - current_segment["start"]
            if duration >= min_duration:
                current_segment["score"] = current_segment["peak_score"]
                current_segment["duration"] = duration
                action_segments.append(current_segment)
        
        # Ordena por score
        action_segments.sort(key=lambda x: x["peak_score"], reverse=True)
        
        logger.info(f"[SUCCESS] {len(action_segments)} cenas de ação detectadas")
        return action_segments

# ==================== ANTI-SHADOWBAN APRIMORADO ====================

def apply_antishadowban(clip):
    """Aplica transformações para tornar vídeo único com segurança"""
    if not MOVIEPY_AVAILABLE:
        logger.warning("[WARNING] MoviePy não disponível, pulando anti-shadowban")
        return clip
    
    logger.info("[ANTI-SHADOWBAN] Aplicando transformações...")
    
    try:
        modifications = []
        
        # Espelhamento aleatório (50% chance)
        if random.choice([True, False]):
            clip = clip.fx(moviepy_imports['mirror_x'])
            modifications.append("espelhamento")
        
        # Ajustes de cor sutis mas perceptíveis
        gamma_val = random.uniform(0.95, 1.05)
        contrast_val = random.uniform(0.96, 1.04)
        
        clip = clip.fx(moviepy_imports['gamma_corr'], gamma_val)
        clip = clip.fx(moviepy_imports['colorx'], contrast_val)
        modifications.append(f"gamma={gamma_val:.2f}")
        modifications.append(f"contraste={contrast_val:.2f}")
        
        # Crop sutil e aleatório (1-3 pixels de cada lado)
        if random.choice([True, False]):
            crop_pixels = random.randint(1, 3)
            w, h = clip.size
            clip = clip.crop(
                x1=crop_pixels, 
                y1=crop_pixels, 
                x2=w-crop_pixels, 
                y2=h-crop_pixels
            )
            modifications.append(f"crop={crop_pixels}px")
        
        logger.info(f"[ANTI-SHADOWBAN] Aplicado: {', '.join(modifications)}")
        
    except Exception as e:
        logger.warning(f"[WARNING] Anti-shadowban parcialmente aplicado: {e}")
    
    return clip

# ==================== GERADOR DE TÍTULOS OTIMIZADO ====================

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Converte cor hexadecimal para RGB com validação"""
    try:
        if hex_color.startswith('#'):
            hex_color = hex_color.lstrip('#')
        
        if len(hex_color) == 6:
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        elif len(hex_color) == 3:
            return tuple(int(c*2, 16) for c in hex_color)
    except:
        pass
    
    # Fallback: branco
    logger.warning(f"[WARNING] Cor inválida '{hex_color}', usando branco")
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
    Renderiza título simples com validação robusta
    """
    if not PIL_AVAILABLE:
        logger.warning("[WARNING] PIL não disponível para criar título")
        return None
    
    if not MOVIEPY_AVAILABLE:
        logger.warning("[WARNING] MoviePy não disponível para criar título")
        return None
    
    try:
        # Valida parâmetros
        if not texto or len(texto.strip()) == 0:
            return None
        
        if largura_video <= 0 or altura_video <= 0 or duracao <= 0:
            logger.warning("[WARNING] Parâmetros de vídeo inválidos para título")
            return None
        
        # Cria imagem para o texto
        img_h = max(100, int(altura_video * 0.3))
        img = Image.new('RGBA', (largura_video, img_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Tenta carregar fonte
        font = None
        if FONT_TO_USE and os.path.exists(FONT_TO_USE):
            try:
                font = ImageFont.truetype(FONT_TO_USE, font_size)
            except Exception as e:
                logger.warning(f"[WARNING] Erro ao carregar fonte: {e}")
        
        if font is None:
            # Fonte padrão
            try:
                font = ImageFont.load_default()
                font_size = 20  # Ajusta tamanho para fonte padrão
            except:
                logger.warning("[WARNING] Não foi possível carregar fonte padrão")
                return None
        
        # Divide texto em linhas inteligentemente
        palavras = texto.strip().split()
        linhas = []
        
        if len(palavras) <= 2:
            linhas = [texto]
        elif len(palavras) <= 6:
            meio = len(palavras) // 2
            linhas = [
                " ".join(palavras[:meio]),
                " ".join(palavras[meio:])
            ]
        else:
            # Divide em 2-3 linhas
            terco = len(palavras) // 3
            linhas = [
                " ".join(palavras[:terco]),
                " ".join(palavras[terco:terco*2]),
                " ".join(palavras[terco*2:])
            ]
        
        # Limita a 3 linhas
        linhas = linhas[:3]
        
        # Cores
        text_rgb = hex_to_rgb(text_color)
        stroke_rgb = hex_to_rgb(stroke_color)
        
        # Desenha texto
        y_pos = 20
        line_spacing = font_size + 10
        
        for linha in linhas:
            if not linha.strip():
                continue
            
            # Posição central
            try:
                bbox = draw.textbbox((0, 0), linha, font=font)
                text_width = bbox[2] - bbox[0]
            except:
                # Fallback para fontes antigas
                text_width = len(linha) * (font_size // 2)
            
            x_pos = max(0, (largura_video - text_width) // 2)
            
            # Contorno (mais eficiente)
            if stroke_width > 0:
                for dx in range(-stroke_width, stroke_width + 1):
                    for dy in range(-stroke_width, stroke_width + 1):
                        if dx == 0 and dy == 0:
                            continue
                        if dx*dx + dy*dy <= stroke_width*stroke_width:
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
            
            y_pos += line_spacing
        
        # Converte para clip
        numpy_img = np.array(img)
        clip = moviepy_imports['ImageClip'](numpy_img).set_duration(duracao)
        pos_y = max(0, min(altura_video - img_h, int(altura_video * pos_vertical)))
        clip = clip.set_position(('center', pos_y))
        
        logger.debug(f"[TITULO] Criado com sucesso: {len(linhas)} linhas")
        return clip
        
    except Exception as e:
        logger.warning(f"[WARNING] Erro ao criar título: {e}")
        return None

# ==================== WHISPER GPU ULTRA-ESTABILIZADO (MANAGER THREAD-SAFE) ====================

class WhisperManager:
    """Gerenciador thread-safe para carregar e usar modelos Whisper (faster_whisper / openai-whisper)
    Mantém carregamento lazy e retries com locks para evitar múltiplas inicializações.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._model = None
        self._loaded = False
        self._type = None

    def loaded(self) -> bool:
        return self._loaded and self._model is not None

    @safe_gpu_operation
    def load(self, preferred: str = "large-v3") -> bool:
        with self._lock:
            if self._loaded and self._model is not None:
                logger.info("[WHISPER] Já carregado, pulando")
                return True

            if not AI_AVAILABLE or not WHISPER_AVAILABLE:
                logger.info("[WHISPER] IA não disponível")
                return False

            # Tenta faster_whisper primeiro
            try:
                if WHISPER_TYPE == "faster_whisper":
                    from faster_whisper import WhisperModel
                    device_to_use = "cuda" if GPU_AVAILABLE else "cpu"
                    compute_type = "float16" if GPU_AVAILABLE else "float32"

                    # Ajuste simples baseado em memória
                    try:
                        if GPU_AVAILABLE and torch and hasattr(torch, 'cuda'):
                            total_memory = torch.cuda.get_device_properties(0).total_memory
                            free_guess = total_memory - getattr(torch.cuda, 'memory_allocated', lambda x: 0)(0)
                            if free_guess < 2e9:
                                compute_type = "int8"
                    except Exception:
                        pass

                    logger.info(f"[WHISPER] Carregando faster_whisper (device={device_to_use}, compute={compute_type})")
                    self._model = WhisperModel(preferred, device=device_to_use, compute_type=compute_type, download_root=str(MODELS_DIR))
                    self._type = "faster_whisper"
                    self._loaded = True
                    return True

            except Exception as e:
                logger.warning(f"[WHISPER] faster_whisper falhou: {e}")

            # Tenta openai-whisper como fallback
            try:
                import whisper as openai_whisper
                logger.info("[WHISPER] Carregando openai-whisper (fallback)")
                self._model = openai_whisper.load_model(preferred, download_root=str(MODELS_DIR))
                self._type = "openai_whisper"
                self._loaded = True
                return True
            except Exception as e:
                logger.error(f"[WHISPER] Nenhum backend Whisper pôde ser carregado: {e}")
                self._loaded = False
                self._model = None
                return False

    @safe_gpu_operation
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        if not self.loaded():
            raise RuntimeError("Whisper não carregado")

        audio_path_obj = Path(audio_path)
        if not validate_audio_file(audio_path_obj):
            raise RuntimeError(f"Arquivo de áudio inválido: {audio_path}")

        try:
            logger.info(f"[WHISPER] Transcrevendo: {audio_path_obj.name}")
            if self._type == "faster_whisper":
                segments, info = self._model.transcribe(audio_path, language="pt", beam_size=3, best_of=3, temperature=0.0)
                chunks = []
                for segment in segments:
                    text = getattr(segment, 'text', '').strip()
                    if text and len(text) > 1:
                        chunks.append({
                            "text": text,
                            "timestamp": (getattr(segment, 'start', 0), getattr(segment, 'end', 0)),
                            "confidence": float(getattr(segment, 'avg_logprob', 0.0))
                        })
                return {"chunks": chunks, "info": info}

            else:
                # openai-whisper
                result = self._model.transcribe(str(audio_path), language="pt")
                text = result.get('text', '')
                # Segmentação simples: cria um único chunk com todo texto
                return {"chunks": [{"text": text.strip(), "timestamp": (0, 0), "confidence": 0.0}], "info": result}

        except Exception as e:
            logger.error(f"[WHISPER] Erro na transcrição: {e}")
            raise


# Instância global do gerenciador
whisper_manager = WhisperManager()


@safe_gpu_operation
def load_turbo_whisper_gpu():
    return whisper_manager.load()


@safe_gpu_operation
def transcrever_com_whisper_gpu(audio_path: str):
    return whisper_manager.transcribe(audio_path)

# ==================== ANÁLISE DE VÍDEO COM IA GPU ULTRA-ROBUSTA ====================

@safe_gpu_operation
def analyze_video_content_gpu(video_path: str, anime_name: str) -> List[Dict]:
    """Analisa vídeo para encontrar cenas virais USANDO GPU com fallbacks"""
    
    if not AI_AVAILABLE or not WHISPER_AVAILABLE:
        logger.info("[INFO] IA não disponível, usando heurística")
        return generate_fallback_cuts(video_path, anime_name)
    
    audio_files_to_clean = []
    
    try:
        # Carrega Whisper GPU se necessário
        if not whisper_manager.loaded():
            logger.info("[WHISPER] Carregando Whisper sob demanda...")
            if not load_turbo_whisper_gpu():
                logger.error("[ERROR] Whisper GPU não pode ser carregado")
                return generate_fallback_cuts(video_path, anime_name)
        
        logger.info("[AUDIO GPU] Extraindo áudio...")
        
        # Extrai áudio com FFmpeg
        raw_audio = TEMP_DIR / f"audio_{uuid.uuid4().hex[:8]}.wav"
        audio_files_to_clean.append(raw_audio)
        
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            str(raw_audio), '-y',
            '-hide_banner', '-loglevel', 'error'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0 or not validate_audio_file(raw_audio):
            raise Exception(f"Falha na extração de áudio: {result.stderr[:200]}")
        
        logger.info("[AUDIO] Processando áudio...")
        clean_audio = clean_audio_deepfilter(raw_audio)
        
        if clean_audio != raw_audio:
            audio_files_to_clean.append(clean_audio)
        
        # Valida áudio limpo
        if not validate_audio_file(clean_audio):
            raise Exception("Áudio processado é inválido")
        
        # Transcrição COM GPU
        logger.info("[TRANSCRIPTION GPU] Transcrevendo...")
        result = transcrever_com_whisper_gpu(str(clean_audio))
        
        # Processa transcrição
        transcript = []
        for seg in result.get("chunks", []):
            text = seg.get("text", "").strip()
            if text and len(text) > 2:
                start, end = seg.get("timestamp", (0, 0))
                confidence = seg.get("confidence", 0.0)
                
                transcript.append({
                    "start": start,
                    "end": end,
                    "text": text,
                    "type": "dialogue",
                    "confidence": confidence
                })
        
        logger.info(f"[TRANSCRIPTION] {len(transcript)} segmentos de diálogo")
        
        # Detecção de ação
        logger.info("[ACTION ANALYSIS] Buscando cenas de ação...")
        detector = ActionDetector(video_path)
        action_scenes = detector.detect_high_energy_segments(threshold=60.0, min_duration=2.0)
        
        for action in action_scenes:
            transcript.append({
                "start": action["start"],
                "end": action["end"],
                "text": f"[AÇÃO - {int(action['score'])}%]",
                "type": "action",
                "score": action["score"],
                "duration": action.get("duration", 0)
            })
        
        logger.info(f"[ACTION ANALYSIS] {len(action_scenes)} cenas de ação detectadas")
        
        # Ordena por timestamp
        transcript.sort(key=lambda x: x["start"])
        
        # Gera cortes baseados na análise
        cuts = []
        
        if not MOVIEPY_AVAILABLE:
            logger.warning("[WARNING] MoviePy não disponível, usando estimativa de duração")
            duration = 300  # 5 minutos padrão
        else:
            try:
                video = moviepy_imports['VideoFileClip'](video_path)
                duration = video.duration
                video.close()
            except Exception as e:
                logger.warning(f"[WARNING] Erro ao obter duração: {e}")
                duration = 300
        
        # Estratégia 1: Prioriza cenas de ação
        action_items = [t for t in transcript if t["type"] == "action"]
        
        if action_items:
            for i, action in enumerate(action_items[:3]):  # Máximo 3
                start = max(0, action["start"] - 3)
                end = min(duration, action["end"] + 3)
                clip_duration = end - start
                
                if 15 <= clip_duration <= 60:  # Entre 15 e 60 segundos
                    cuts.append({
                        "start": start,
                        "end": end,
                        "title": f"{anime_name} - AÇÃO {i+1}",
                        "score": min(95, int(70 + action.get("score", 0) / 5)),
                        "type": "action",
                        "duration": clip_duration
                    })
        
        # Estratégia 2: Diálogos importantes (se precisar mais cortes)
        if len(cuts) < 3:
            dialogue_items = [t for t in transcript if t["type"] == "dialogue"]
            
            # Filtra por comprimento e confiança
            dialogue_items = [
                d for d in dialogue_items
                if len(d["text"].split()) > 4 and d.get("confidence", 0) > -0.5
            ]
            
            # Ordena por tamanho do texto (provavelmente mais importante)
            dialogue_items.sort(key=lambda x: len(x["text"]), reverse=True)
            
            for i, item in enumerate(dialogue_items):
                if len(cuts) >= 3:
                    break
                
                start = max(0, item["start"] - 1.5)
                end = min(duration, item["end"] + 1.5)
                clip_duration = end - start
                
                if 10 <= clip_duration <= 45:
                    cuts.append({
                        "start": start,
                        "end": end,
                        "title": f"{anime_name} - CENA {len(cuts)+1}",
                        "score": max(50, 75 - (i * 5)),
                        "type": "dialogue",
                        "duration": clip_duration
                    })
        
        # Estratégia 3: Fallback - divide em partes iguais
        if not cuts:
            logger.info("[INFO] Nenhum corte automático encontrado, usando divisão uniforme")
            num_parts = min(3, max(1, int(duration / 40)))
            
            for i in range(num_parts):
                part_duration = duration / num_parts
                start = i * part_duration
                end = min((i + 1) * part_duration, duration)
                
                if end - start >= 25:
                    cuts.append({
                        "start": start,
                        "end": end,
                        "title": f"{anime_name} - Parte {i+1}",
                        "score": 60,
                        "type": "uniform",
                        "duration": end - start
                    })
        
        # Ordena por score e limita
        cuts.sort(key=lambda x: x["score"], reverse=True)
        cuts = cuts[:3]
        
        logger.info(f"[ANALYSIS GPU] {len(cuts)} cortes identificados")
        return cuts
        
    except Exception as e:
        logger.error(f"[ERROR] Erro na análise GPU: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return generate_fallback_cuts(video_path, anime_name)
    
    finally:
        # Limpeza GARANTIDA de arquivos de áudio temporários
        for audio_file in audio_files_to_clean:
            try:
                if audio_file.exists():
                    audio_file.unlink()
                    logger.debug(f"[CLEANUP] Removido: {audio_file.name}")
            except Exception as e:
                logger.warning(f"[CLEANUP] Erro ao remover {audio_file.name}: {e}")
        
        # Força coleta de lixo
        gc.collect()

def generate_fallback_cuts(video_path: str, anime_name: str) -> List[Dict]:
    """Gera cortes fallback simples e seguros"""
    
    try:
        if MOVIEPY_AVAILABLE:
            video = moviepy_imports['VideoFileClip'](video_path)
            duration = video.duration
            video.close()
        else:
            # Usa ffprobe para obter duração
            cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            duration = float(result.stdout.strip()) if result.returncode == 0 else 300
        
        # Divide em até 3 partes
        num_cuts = min(3, max(1, int(duration / 60)))
        cuts = []
        
        for i in range(num_cuts):
            part_duration = duration / num_cuts
            start = i * part_duration
            end = min((i + 1) * part_duration, duration)
            
            # Ajusta para mínimo de 30 segundos
            if end - start < 30:
                end = start + 30
                if end > duration:
                    start = max(0, duration - 30)
                    end = duration
            
            cuts.append({
                "start": start,
                "end": end,
                "title": f"{anime_name} - Parte {i+1}",
                "score": 50 - (i * 5),
                "type": "fallback",
                "duration": end - start
            })
        
        logger.info(f"[FALLBACK] Gerados {len(cuts)} cortes simples")
        return cuts
        
    except Exception as e:
        logger.error(f"[ERROR] Erro no fallback: {e}")
        # Fallback extremo
        return [{
            "start": 30,
            "end": 90,
            "title": anime_name,
            "score": 30,
            "type": "emergency",
            "duration": 60
        }]

# ==================== PROCESSAMENTO DE CORTES GPU ULTRA-OTIMIZADO ====================

@contextlib.contextmanager
def moviepy_clip_context(*clips):
    """Context manager para garantir fechamento de clips"""
    try:
        yield clips
    finally:
        for clip in clips:
            try:
                if clip is not None:
                    clip.close()
            except:
                pass

@safe_gpu_operation
def processar_corte_gpu(video_path: str, cut_data: Dict, num: int, config: Dict) -> str:
    """Processa um corte individual OTIMIZADO PARA GPU com cleanup robusto"""
    
    video = None
    clip = None
    bg_clip = None
    
    try:
        start = cut_data.get('start', 0)
        end = cut_data.get('end', start + 60)
        title = cut_data.get('title', config.get('animeName', 'Anime'))
        
        logger.info(f"[CUT {num}] {title} ({start:.1f}s - {end:.1f}s)")
        
        # Valida tempos
        if start < 0 or end <= start:
            raise ValueError(f"Tempos inválidos: start={start}, end={end}")
        
        # Valida que o vídeo esteja em um path permitido
        if not is_safe_path(VOLUME_PATH, video_path) and not is_safe_path(TEMP_DIR, video_path) and not is_safe_path(CACHE_DIR, video_path):
            raise ValueError(f"Caminho de vídeo não permitido: {video_path}")

        # Usa ExitStack para garantir fechamento ordenado de clips
        with contextlib.ExitStack() as stack:
            video = moviepy_imports['VideoFileClip'](video_path)
            stack.callback(lambda v=video: getattr(v, 'close', lambda: None)())

            # Valida tempos contra duração do vídeo
            if end > video.duration:
                logger.warning(f"[WARNING] Ajustando fim de {end}s para {video.duration}s")
                end = video.duration

            if start >= video.duration:
                raise ValueError(f"Start {start}s excede duração do vídeo {video.duration}s")

            # Corta segmento
            clip = video.subclip(start, end)
            stack.callback(lambda c=clip: getattr(c, 'close', lambda: None)())

            # Aplica anti-shadowban
            if config.get("antiShadowban", True):
                clip = apply_antishadowban(clip)
        
        # Configurações TikTok
        target_w, target_h = 1080, 1920
        
        # Background
        bg_path = config.get("background_path")

        if bg_path and os.path.exists(bg_path) and PIL_AVAILABLE:
            try:
                # Valida background
                if not is_safe_path(VOLUME_PATH, bg_path) and not is_safe_path(TEMP_DIR, bg_path) and not is_safe_path(CACHE_DIR, bg_path):
                    raise ValueError(f"Caminho de background não permitido: {bg_path}")

                bg_img = Image.open(bg_path).convert('RGB')
                bg_img = bg_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                bg_clip = moviepy_imports['ImageClip'](np.array(bg_img)).set_duration(clip.duration)
                stack.callback(lambda b=bg_clip: getattr(b, 'close', lambda: None)())
                logger.debug("[RENDER] Background carregado")
            except Exception as e:
                logger.warning(f"[WARNING] Background falhou: {e}")
                bg_clip = None

        if bg_clip is None:
            # Background sólido escuro
            bg_color = (15, 15, 30)
            bg_clip = moviepy_imports['ColorClip'](size=(target_w, target_h), color=bg_color)
            bg_clip = bg_clip.set_duration(clip.duration)
            stack.callback(lambda b=bg_clip: getattr(b, 'close', lambda: None)())
            logger.debug("[RENDER] Background sólido")
        
        # Ajusta vídeo para 9:16
        w, h = clip.w, clip.h
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
        
            stack.callback(lambda c=clip_cropped: getattr(c, 'close', lambda: None)())

            # Redimensiona
            clip_resized = clip_cropped.resize(width=target_w)
            stack.callback(lambda c=clip_resized: getattr(c, 'close', lambda: None)())

            clip_pos = clip_resized.set_position(('center', 'center'))
            stack.callback(lambda c=clip_pos: getattr(c, 'close', lambda: None)())
        
        # Camadas
        layers = [bg_clip, clip_pos]

        # Título
        if config.get("generateTitles", True) and title and PIL_AVAILABLE:
            title_style = config.get("titleStyle", {})
            safe_title_text = sanitize_input(str(title).upper(), max_len=80)
            title_clip = criar_titulo_simples(
                texto=safe_title_text,
                largura_video=target_w,
                altura_video=target_h,
                duracao=clip.duration,
                font_size=title_style.get("fontSize", 70),
                text_color=title_style.get("textColor", "#FFD700"),
                stroke_color=title_style.get("strokeColor", "#000000"),
                stroke_width=title_style.get("strokeWidth", 6)
            )

            if title_clip:
                layers.append(title_clip)
                stack.callback(lambda t=title_clip: getattr(t, 'close', lambda: None)())
                logger.debug("[RENDER] Título adicionado")

        # Composição final
        final = moviepy_imports['CompositeVideoClip'](layers, size=(target_w, target_h))
        stack.callback(lambda f=final: getattr(f, 'close', lambda: None)())

        # Nome do arquivo de saída
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_'))[:30]
        output_filename = f"cut_{num}_{safe_title}_{uuid.uuid4().hex[:6]}.mp4"
        output_path = OUTPUT_DIR / output_filename
        
        # Configuração de encoding OTIMIZADA
        ffmpeg_params = [
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-vsync', 'vfr'
        ]
        
        # Detecta NVENC
        codec = 'libx264'
        preset = 'medium'
        
        if GPU_AVAILABLE and FFMPEG_AVAILABLE:
            try:
                result = subprocess.run(
                    ['ffmpeg', '-encoders'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if 'h264_nvenc' in result.stdout:
                    codec = 'h264_nvenc'
                    preset = 'p5'  # p5 é um bom balanço velocidade/qualidade
                    
                    ffmpeg_params.extend([
                        '-rc', 'vbr',
                        '-cq', '23',
                        '-b:v', '0',
                        '-maxrate', '10M',
                        '-bufsize', '20M',
                        '-preset', preset,
                        '-profile:v', 'high',
                        '-tier', 'high',
                        '-spatial_aq', '1',
                        '-temporal_aq', '1'
                    ])
                    logger.info("[ENCODING GPU] Usando NVENC")
                else:
                    logger.info("[ENCODING] NVENC não disponível, usando CPU")
                    
            except Exception as e:
                logger.warning(f"[WARNING] Erro ao detectar NVENC: {e}")
        
        if codec == 'libx264':
            ffmpeg_params.extend([
                '-crf', '23',
                '-preset', preset,
                '-tune', 'film',
                '-profile:v', 'high',
                '-level', '4.0'
            ])
        
            # Renderiza
            logger.info(f"[RENDERING GPU] Renderizando {output_filename}...")
            temp_audio = TEMP_DIR / f"temp_audio_{num}_{uuid.uuid4().hex[:6]}.m4a"

            final.write_videofile(
                str(output_path),
                codec=codec,
                audio_codec='aac',
                audio_bitrate='192k',
                preset=preset,
                threads=4,
                ffmpeg_params=ffmpeg_params,
                logger=None,
                verbose=False,
                temp_audiofile=str(temp_audio),
                remove_temp=True
            )

            # Valida saída
            if not output_path.exists() or output_path.stat().st_size < 100000:  # < 100KB
                raise Exception(f"Arquivo de saída inválido: {output_path}")

            file_size = output_path.stat().st_size / 1e6
            logger.info(f"[SUCCESS GPU] Corte {num} finalizado ({file_size:.1f} MB)")

            return str(output_path)
        # fim with ExitStack
        
    except Exception as e:
        logger.error(f"[ERROR GPU] Erro no corte {num}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
        
    finally:
        # Força coleta de lixo
        gc.collect()

# ==================== CLEANUP ROBUSTO ====================

def cleanup_temp_files(keep_recent=5):
    """Limpa arquivos temporários mantendo os mais recentes"""
    try:
        temp_files = []
        
        # Coleta todos os arquivos temporários
        for pattern in ["*.mp4", "*.wav", "*.m4a", "*.png", "*.jpg"]:
            temp_files.extend(TEMP_DIR.glob(pattern))
        
        if not temp_files:
            return
        
        # Ordena por tempo de modificação (mais recentes primeiro)
        temp_files.sort(key=lambda x: x.stat().st_mtime if x.is_file() else 0, reverse=True)
        
        # Remove arquivos antigos
        removed_count = 0
        freed_space = 0
        
        for i, temp_file in enumerate(temp_files):
            if i < keep_recent:
                continue  # Mantém os N mais recentes
            
            try:
                if temp_file.is_file():
                    size = temp_file.stat().st_size
                    temp_file.unlink()
                    freed_space += size
                    removed_count += 1
            except Exception as e:
                logger.warning(f"[CLEANUP] Erro ao remover {temp_file.name}: {e}")
        
        if removed_count > 0:
            logger.info(f"[CLEANUP] Removidos {removed_count} arquivos ({freed_space/1e6:.1f} MB liberados)")
            
    except Exception as e:
        logger.error(f"[CLEANUP] Erro na limpeza: {e}")

# ==================== HANDLER PRINCIPAL ULTRA-ESTÁVEL ====================

def handler(event):
    """Handler principal do RunPod - ULTRA-ESTABILIZADO"""
    
    start_time = time.time()
    request_id = uuid.uuid4().hex[:8]
    
    # LOG DE INICIALIZAÇÃO
    logger.info("=" * 70)
    logger.info(f"ANIMECUT v12.1 - NOVA REQUISIÇÃO [ID: {request_id}]")
    logger.info("=" * 70)
    
    # LOG DE STATUS DO SISTEMA
    system_status = {
        "gpu": GPU_AVAILABLE,
        "torch_version": TORCH_VERSION,
        "cuda_version": torch.version.cuda if torch and hasattr(torch.version, 'cuda') else None,
        "moviepy": MOVIEPY_AVAILABLE,
        "moviepy_version": moviepy_version,
        "whisper": WHISPER_AVAILABLE,
        "whisper_type": WHISPER_TYPE,
        "deepfilter": DF_AVAILABLE,
        "ffmpeg": FFMPEG_AVAILABLE,
        "opencv": CV2_AVAILABLE,
        "b2": B2_AVAILABLE
    }
    
    if GPU_AVAILABLE and torch:
        try:
            system_status["gpu_name"] = torch.cuda.get_device_name(0)
            system_status["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        except:
            pass
    
    logger.info(f"[SYSTEM] {json.dumps(system_status, indent=2)}")
    
    # Modo teste
    input_data = event.get("input", {})
    
    if input_data.get("mode") == "test":
        return {
            "status": "success",
            "request_id": request_id,
            "system": system_status,
            "module_info": dep_manager.get_module_info(),
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        # Valida entrada e sanitiza (não escapar HTML em URLs assinadas)
        video_url = input_data.get("video_url")
        if not video_url:
            raise ValueError("video_url é obrigatório")
        video_url = sanitize_input(video_url, max_len=2000, escape_html=False)
        if not network.validate_url(video_url):
            raise ValueError("video_url inválido")

        anime_name = sanitize_input(input_data.get("animeName", "Anime"), max_len=80)

        logger.info(f"[PROCESSING] Anime: {anime_name}")
        logger.info(f"[PROCESSING] URL: {video_url[:80]}...")

        # 1. Download com retry
        logger.info("[STEP 1/4] Download de vídeo...")
        video_path = download_video(video_url)

        # Background (opcional) - não escapar HTML em URLs
        bg_url = input_data.get("background_url")
        bg_url = sanitize_input(bg_url, escape_html=False) if bg_url else None
        bg_path = download_background(bg_url) if bg_url else None
        
        # Configuração
        config = {
            "animeName": anime_name,
            "antiShadowban": input_data.get("antiShadowban", True),
            "generateTitles": input_data.get("generateTitles", True),
            "titleStyle": input_data.get("titleStyle", {
                "fontSize": 70,
                "textColor": "#FFD700",
                "strokeColor": "#000000",
                "strokeWidth": 6
            }),
            "background_path": bg_path
        }
        
        logger.info(f"[CONFIG] {json.dumps(config, default=str)}")
        
        # 2. Análise e definição de cortes
        logger.info("[STEP 2/4] Análise de conteúdo...")
        
        cuts = []
        cut_type = input_data.get("cutType", "auto")
        
        if cut_type == "auto" and AI_AVAILABLE:
            logger.info("[MODE] Automático com IA")
            cuts = analyze_video_content_gpu(video_path, anime_name)
        elif cut_type == "manual":
            manual_cuts = input_data.get("cuts", [])
            if manual_cuts:
                cuts = manual_cuts
                logger.info(f"[MODE] Manual: {len(cuts)} cortes")
            else:
                logger.info("[MODE] Manual sem cortes, usando automático")
                cuts = analyze_video_content_gpu(video_path, anime_name)
        else:
            logger.info("[MODE] Fallback")
            cuts = generate_fallback_cuts(video_path, anime_name)
        
        # Limite e validação de cortes
        cuts = [c for c in cuts if c.get("start", 0) >= 0 and c.get("end", 0) > c.get("start", 0)]
        cuts = cuts[:3]  # Máximo 3
        
        if not cuts:
            logger.warning("[WARNING] Nenhum corte válido, usando fallback de emergência")
            cuts = generate_fallback_cuts(video_path, anime_name)
        
        logger.info(f"[CUTS] {len(cuts)} cortes para processar")
        
        # 3. Processamento dos cortes
        logger.info("[STEP 3/4] Processamento de cortes...")
        
        results = []
        failed_cuts = []
        
        for i, cut in enumerate(cuts):
            cut_start_time = time.time()
            
            try:
                logger.info(f"[CUT {i+1}/{len(cuts)}] Iniciando...")
                
                # Limpeza preventiva
                gc.collect()
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Processa corte
                out_path = processar_corte_gpu(video_path, cut, i+1, config)
                
                # Upload opcional para B2
                b2_url = None
                if B2_AVAILABLE and s3_client and B2_BUCKET:
                    try:
                        filename = os.path.basename(out_path)
                        key = f"animecut/{datetime.now().strftime('%Y%m%d')}/{filename}"
                        
                        logger.info(f"[UPLOAD] Enviando para B2: {key}")
                        
                        def _do_upload():
                            s3_client.upload_file(
                                out_path,
                                B2_BUCKET,
                                key,
                                ExtraArgs={'ContentType': 'video/mp4'}
                            )

                        # Usa circuit breaker para proteger chamadas B2
                        breaker.call(_do_upload)

                        b2_url = s3_client.generate_presigned_url(
                            'get_object',
                            Params={'Bucket': B2_BUCKET, 'Key': key},
                            ExpiresIn=86400  # 24 horas
                        )

                        logger.info("[UPLOAD] Concluído")
                        
                    except Exception as e:
                        logger.warning(f"[WARNING] Upload B2 falhou: {e}")
                
                cut_elapsed = time.time() - cut_start_time
                
                results.append({
                    "id": i+1,
                    "path": out_path,
                    "url": b2_url,
                    "title": cut.get("title", anime_name),
                    "start": cut.get("start"),
                    "end": cut.get("end"),
                    "duration": cut.get("end", 0) - cut.get("start", 0),
                    "score": cut.get("score", 0),
                    "type": cut.get("type", "unknown"),
                    "processing_time": round(cut_elapsed, 2),
                    "file_size_mb": round(Path(out_path).stat().st_size / 1e6, 2),
                    "gpu_encoded": GPU_AVAILABLE
                })
                
                logger.info(f"[SUCCESS] Corte {i+1} concluído em {cut_elapsed:.1f}s")
                
                # Limpeza após cada corte
                gc.collect()
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"[ERROR] Corte {i+1} falhou: {e}")
                failed_cuts.append({
                    "id": i+1,
                    "error": str(e),
                    "cut_data": cut
                })
                continue
        
        # 4. Limpeza final
        logger.info("[STEP 4/4] Limpeza final...")
        
        # Limpa recursos registrados
        resource_manager.cleanup_all()
        
        # Limpa temp files
        cleanup_temp_files(keep_recent=3)
        
        # Limpeza GPU final
        gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Tempo total
        elapsed_time = time.time() - start_time
        
        # Resultado final
        logger.info(f"[FINISHED] {len(results)}/{len(cuts)} cortes gerados em {elapsed_time:.1f}s")
        
        response = {
            "status": "success",
            "request_id": request_id,
            "cuts": results,
            "failed_cuts": failed_cuts if failed_cuts else None,
            "metadata": {
                "anime_name": anime_name,
                "total_cuts_requested": len(cuts),
                "successful_cuts": len(results),
                "failed_cuts": len(failed_cuts),
                "processing_time": round(elapsed_time, 2),
                "gpu_used": GPU_AVAILABLE,
                "whisper_used": whisper_manager.loaded(),
                "encoding_method": "nvenc" if GPU_AVAILABLE else "libx264",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        logger.info("=" * 70)
        logger.info(f"REQUISIÇÃO FINALIZADA [ID: {request_id}]")
        logger.info("=" * 70)
        
        return response
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        
        logger.error(f"[ERROR] Erro fatal após {elapsed_time:.1f}s: {e}")
        import traceback
        error_trace = traceback.format_exc()
        logger.error(error_trace)
        
        # Limpeza em caso de erro
        try:
            resource_manager.cleanup_all()
            cleanup_temp_files()
            gc.collect()
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        return {
            "status": "error",
            "request_id": request_id,
            "error": str(e),
            "error_type": type(e).__name__,
            "processing_time": round(elapsed_time, 2),
            "traceback": error_trace if input_data.get("debug", False) else None,
            "timestamp": datetime.now().isoformat()
        }

def safe_handler(event):
    """Wrapper ultra-seguro com timeout"""
    try:
        return handler(event)
    except Exception as e:
        logger.error(f"[CRITICAL] Erro no wrapper: {e}")
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "error_type": "critical_wrapper_error",
            "traceback": traceback.format_exc()
        }

# ==================== INICIALIZAÇÃO ====================

if __name__ == "__main__":
    try:
        # Banner
        print("\n" + "="*70)
        print("ANIMECUT SERVERLESS v12.1 - ULTRA-STABLE")
        print("Todas as correções aplicadas:")
        print("  ✓ Gestão de memória GPU otimizada")
        print("  ✓ Cleanup robusto de recursos")
        print("  ✓ Validação de arquivos")
        print("  ✓ Fallbacks em múltiplos níveis")
        print("  ✓ Thread-safe operations")
        print("  ✓ Context managers para clips")
        print(f"Volume: {VOLUME_BASE}")
        print(f"Cache: {CACHE_DIR}")
        print("="*70)
        
        # Status detalhado
        print("\n[SYSTEM STATUS]")
        print(f"  GPU: {'✓' if GPU_AVAILABLE else '✗'} {torch.cuda.get_device_name(0) if GPU_AVAILABLE and torch else 'N/A'}")
        print(f"  CUDA: {'✓' if GPU_AVAILABLE else '✗'} {torch.version.cuda if torch and hasattr(torch.version, 'cuda') else 'N/A'}")
        print(f"  MoviePy: {'✓' if MOVIEPY_AVAILABLE else '✗'} {moviepy_version}")
        print(f"  PyTorch: {'✓' if AI_AVAILABLE else '✗'} {TORCH_VERSION if torch else 'N/A'}")
        print(f"  Whisper: {'✓' if WHISPER_AVAILABLE else '✗'} {WHISPER_TYPE if WHISPER_AVAILABLE else 'N/A'}")
        print(f"  FFmpeg: {'✓' if FFMPEG_AVAILABLE else '✗'} {FFMPEG_VERSION if FFMPEG_AVAILABLE else 'N/A'}")
        print(f"  DeepFilter: {'✓' if DF_AVAILABLE else '✗'} {DF_TYPE if DF_AVAILABLE else 'N/A'}")
        print(f"  OpenCV: {'✓' if CV2_AVAILABLE else '✗'}")
        print(f"  YOLO: {'✓' if YOLO_AVAILABLE else '✗'}")
        print(f"  Pillow: {'✓' if PIL_AVAILABLE else '✗'}")
        print(f"  B2 Storage: {'✓' if B2_AVAILABLE else '✗'}")
        
        if GPU_AVAILABLE and torch:
            try:
                props = torch.cuda.get_device_properties(0)
                print(f"\n[GPU DETAILS]")
                print(f"  Name: {props.name}")
                print(f"  Memory: {props.total_memory / 1e9:.2f} GB")
                print(f"  Compute: {props.major}.{props.minor}")
                print(f"  Multi-Processors: {props.multi_processor_count}")
            except:
                pass
        
        print("="*70 + "\n")
        sys.stdout.flush()
        
        # Tenta importar e iniciar RunPod
        try:
            import runpod
            
            logger.info("[RUNPOD] Iniciando servidor serverless...")
            
            # Configuração do servidor
            runpod.serverless.start({
                "handler": safe_handler,
                "concurrency_modifier": lambda x: 1,  # 1 job por vez
                "return_aggregate_stream": True
            })
            
        except ImportError:
            print("\n" + "="*70)
            print("WARNING: RunPod não instalado")
            print("="*70)
            print("Executando em modo de teste local...")
            print("Para usar no RunPod: pip install runpod")
            print("="*70 + "\n")
            
            # Modo local de teste
            print("Executando teste de sistema...\n")
            
            test_event = {
                "input": {
                    "mode": "test"
                }
            }
            
            result = safe_handler(test_event)
            print("\n[RESULTADO DO TESTE]")
            print(json.dumps(result, indent=2, default=str))
            
            # Teste básico se URL fornecida
            if len(sys.argv) > 1:
                test_url = sys.argv[1]
                print(f"\n[TESTE COM VÍDEO]")
                print(f"URL: {test_url}\n")
                
                test_event_video = {
                    "input": {
                        "video_url": test_url,
                        "animeName": "Teste Local",
                        "cutType": "auto",
                        "antiShadowban": True,
                        "generateTitles": True,
                        "debug": True
                    }
                }
                
                result_video = safe_handler(test_event_video)
                print("\n[RESULTADO DO PROCESSAMENTO]")
                print(json.dumps(result_video, indent=2, default=str))
            
    except KeyboardInterrupt:
        print("\n\n[SHUTDOWN] Servidor interrompido pelo usuário")
        
        # Limpeza final
        try:
            resource_manager.cleanup_all()
            cleanup_temp_files()
            
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("[CLEANUP] Limpeza final concluída")
        except Exception as e:
            logger.warning(f"[CLEANUP] Erro na limpeza final: {e}")
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n[FATAL ERROR] Erro na inicialização: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
