#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AnimeCut Serverless v15.9.7 FFMPEG COM NVENC
BUILD: 2025-12-22 21:00 - GPU ENCODING REAL
Stack: Whisper V3 Turbo, YOLOv8, DeepFilterNet, FFmpeg NVENC
CORREÇÕES: 
- FFMPEG: Compilado do source com suporte NVENC
- NVENC: h264_nvenc e hevc_nvenc disponíveis
- GPU: Encoding ~2-3x mais rápido que CPU
- PATH: Corrigido erro "referenced before assignment"
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

# ==================== BIBLIOTECA DE TÍTULOS CRIATIVOS v1.0 ====================
# 50+ títulos para cada gênero - NUNCA use títulos genéricos

TITULOS_ACAO = [
    "A BATALHA QUE MUDOU TUDO!", "ELE SIMPLESMENTE DESTRUIU GERAL!", "NINGUÉM ESPERAVA ESSE GOLPE!",
    "O PODER MÁXIMO FOI LIBERADO!", "ESSA LUTA VAI TE DEIXAR SEM AR!", "FOI NESSE MOMENTO QUE ELE PERDEU TUDO!",
    "A TÉCNICA SECRETA FINALMENTE REVELADA!", "QUANDO O HERÓI PAROU DE BRINCAR!", "O INIMIGO TREMEU DE MEDO!",
    "ESSE SOCO ATRAVESSOU DIMENSÕES!", "A EXPLOSÃO MAIS ÉPICA DO ANIME!", "ELE SOZINHO CONTRA UM EXÉRCITO!",
    "O MOMENTO QUE DEFINIU A GUERRA!", "PODER ALÉM DO LIMITE HUMANO!", "A TRANSFORMAÇÃO QUE CHOCOU A TODOS!",
    "NUNCA SUBESTIME ESSE PERSONAGEM!", "O CONFRONTO FINAL COMEÇOU!", "ESSA CENA QUEBROU A INTERNET!",
    "O GOLPE QUE NINGUÉM VIU CHEGAR!", "ELE ACORDOU O MONSTRO INTERIOR!", "A VINGANÇA FOI SERVIDA GELADA!",
    "QUANDO A RAIVA TOMA CONTA!", "O SACRIFÍCIO QUE SALVOU TODOS!", "ESSE CARA É SIMPLESMENTE INSANO!",
    "A TÉCNICA PROIBIDA FOI USADA!", "ELE SUPEROU SEUS PRÓPRIOS LIMITES!", "O MOMENTO MAIS BRUTAL DO ANIME!",
    "FOI ASSIM QUE ELE SE TORNOU LENDA!", "A CENA QUE TODO FÃ ESPERAVA!", "NÍVEL DE PODER: ABSURDO!",
    "O HERÓI FINALMENTE FICOU SÉRIO!", "ESSA SEQUÊNCIA É PERFEITA!", "O VILÃO CONHECEU SEU DESTINO!",
    "ADRENALINA PURA DO INÍCIO AO FIM!", "ELE MOSTROU DO QUE É CAPAZ!", "A BATALHA DOS TITÃS!",
    "QUANDO DOIS MONSTROS SE ENCONTRAM!", "O DESPERTAR DO VERDADEIRO PODER!", "ESSA CENA DÁ ARREPIOS!",
    "O MOMENTO QUE MUDOU O PROTAGONISTA!", "A LUTA MAIS INTENSA DA TEMPORADA!", "ELE LUTOU ATÉ O ÚLTIMO SUSPIRO!",
    "O CONTRA-ATAQUE DEVASTADOR!", "QUANDO O FRACO VIRA O MAIS FORTE!", "A EXPLOSÃO DE PODER INCONTROLÁVEL!",
    "ELE PROTEGEU TODOS COM UM GOLPE!", "O CLÍMAX DA BATALHA!", "ESSE MOMENTO FICOU PRA HISTÓRIA!",
    "A VITÓRIA MAIS ÉPICA DE TODAS!", "O VILÃO NÃO SABIA COM QUEM MEXIA!",
]

TITULOS_DIALOGO = [
    "ESSA FALA MUDOU MINHA VISÃO!", "AS PALAVRAS MAIS FORTES DO ANIME!", "ELE DISSE A VERDADE NA CARA!",
    "ESSE DISCURSO É PERFEITO!", "PALAVRAS QUE ATRAVESSAM A ALMA!", "O MOMENTO MAIS PROFUNDO!",
    "ESSA FRASE FICOU NA MINHA CABEÇA!", "ELE FALOU O QUE TODOS PENSAVAM!", "A VERDADE FINALMENTE FOI DITA!",
    "ESSE DIÁLOGO É OBRA DE ARTE!", "QUANDO AS PALAVRAS MACHUCAM MAIS!", "A FRASE QUE DEFINIU O PERSONAGEM!",
    "ELE CALOU TODO MUNDO!", "O DISCURSO QUE MOTIVOU O HERÓI!", "PALAVRAS DE UM VERDADEIRO LÍDER!",
    "ESSA RESPOSTA FOI DESTRUIDORA!", "O VILÃO TINHA RAZÃO NESSA!", "A FILOSOFIA POR TRÁS DO ANIME!",
    "ELE EXPLICOU TUDO EM UMA FRASE!", "O MOMENTO DE MAIOR SABEDORIA!", "ESSA FALA DÁ ARREPIOS!",
    "QUANDO O SILÊNCIO FALA MAIS!", "A PROMESSA QUE MUDOU TUDO!", "ELE JUROU COM A PRÓPRIA VIDA!",
    "PALAVRAS QUE NUNCA VOU ESQUECER!", "O CONSELHO QUE TODOS PRECISAM!", "ESSA CENA É PURA REFLEXÃO!",
    "A MENSAGEM ESCONDIDA DO ANIME!", "ELE TOCOU NO PONTO CERTO!", "O DIÁLOGO MAIS INTENSO!",
    "QUANDO A VERDADE DÓI!", "A FRASE QUE VIROU MEME!", "ELE DESTRUIU O ARGUMENTO!",
    "O MOMENTO MAIS FILOSÓFICO!", "PALAVRAS DE QUEM JÁ SOFREU!", "ESSA FALA RESUME O ANIME!",
    "A LIÇÃO MAIS IMPORTANTE!", "ELE DISSE SEM MEDO!", "O DISCURSO DA VITÓRIA!",
    "QUANDO AS PALAVRAS CURAM!", "A FRASE QUE INICIOU A GUERRA!", "ELE REVELOU SEU VERDADEIRO EU!",
    "O MOMENTO DE VULNERABILIDADE!", "PALAVRAS QUE DERAM FORÇA!", "ESSA CENA PRECISA SER VISTA!",
    "A CONFISSÃO QUE NINGUÉM ESPERAVA!", "ELE FINALMENTE SE ABRIU!", "O DIÁLOGO QUE FEZ CHORAR!",
    "QUANDO AS PALAVRAS SÃO ARMAS!", "A FRASE MAIS MARCANTE DA SÉRIE!",
]

TITULOS_EMOCAO = [
    "EU NÃO ESTAVA PREPARADO PRA ISSO!", "ESSA CENA ME DESTRUIU!", "QUEM CORTOU AS CEBOLAS?!",
    "O MOMENTO MAIS TRISTE DO ANIME!", "EU CHOREI LITROS COM ISSO!", "A DESPEDIDA QUE DOEU DEMAIS!",
    "ELE DEU TUDO PELOS AMIGOS!", "O SACRIFÍCIO MAIS NOBRE!", "ESSA CENA PARTIU MEU CORAÇÃO!",
    "NUNCA VOU SUPERAR ISSO!", "O ADEUS QUE NINGUÉM QUERIA!", "ELE MORREU COMO HERÓI!",
    "A PERDA QUE MUDOU TUDO!", "ESSE MOMENTO ME MARCOU PRA SEMPRE!", "QUANDO A DOR É INSUPORTÁVEL!",
    "O FLASHBACK MAIS TRISTE!", "ELE GUARDOU ISSO O TEMPO TODO!", "A VERDADE POR TRÁS DA MÁSCARA!",
    "ESSE PERSONAGEM MERECIA MAIS!", "O MOMENTO QUE O HERÓI QUEBROU!", "LÁGRIMAS DE UM GUERREIRO!",
    "A CENA QUE TODO MUNDO CHOROU!", "ELE NUNCA MAIS SERIA O MESMO!", "O PESO DE CARREGAR ESSE FARDO!",
    "QUANDO A ESPERANÇA MORRE!", "A ÚLTIMA LEMBRANÇA!", "ELE SORRIU NO FINAL!",
    "O ARREPENDIMENTO CHEGOU TARDE!", "ESSA HISTÓRIA É MUITO INJUSTA!", "O MOMENTO MAIS HUMANO!",
    "QUANDO HERÓIS TAMBÉM CHORAM!", "A CICATRIZ QUE NUNCA SOME!", "ELE PERDEU TUDO QUE AMAVA!",
    "O PREÇO DO PODER!", "ESSA CENA ME FAZ CHORAR SEMPRE!", "A SOLIDÃO DO MAIS FORTE!",
    "ELE CARREGOU ESSE PESO SOZINHO!", "O MOMENTO DE MAIOR DESESPERO!", "QUANDO TUDO PARECE PERDIDO!",
    "A FERIDA QUE NUNCA CICATRIZA!", "ELE ESCOLHEU O CAMINHO DIFÍCIL!", "O LUTO QUE TRANSFORMOU!",
    "ESSA BACKSTORY DESTRUIU MINHA ALMA!", "A DOR DE QUEM FICOU!", "ELE GUARDOU PRO RESTO DA VIDA!",
    "O MOMENTO MAIS REAL DO ANIME!", "QUANDO A TRISTEZA VIRA FORÇA!", "A CENA QUE DEFINIU O TRAUMA!",
    "ELE NUNCA CONTOU PRA NINGUÉM!", "O PESO DE SER O ESCOLHIDO!",
]

TITULOS_HUMOR = [
    "EU RACHEI DE RIR COM ISSO!", "A CENA MAIS ENGRAÇADA!", "NÃO TEM COMO NÃO RIR!",
    "ESSE ANIME É MUITO BOM!", "A PIADA QUE ME PEGOU!", "HUMOR DE QUALIDADE!",
    "EU RI MAIS DO QUE DEVIA!", "ESSA CENA É COMÉDIA PURA!", "O TIMING PERFEITO!",
    "NÃO ESPERAVA ESSA!", "A REAÇÃO DELE FOI DEMAIS!", "MORRI DE RIR LITERALMENTE!",
    "ESSE PERSONAGEM É HILÁRIO!", "A CENA MAIS RANDOM!", "QUANDO O ANIME TE SURPREENDE!",
    "EU NÃO CONSIGO PARAR DE RIR!", "O MOMENTO MAIS ABSURDO!", "ESSA PIADA É GENIAL!",
    "A CARA DELE DISSE TUDO!", "COMÉDIA DO MAIS ALTO NÍVEL!", "EU CUSPI MINHA ÁGUA!",
    "ESSE ANIME NÃO TEM LIMITE!", "A CENA QUE VIROU MEME!", "HUMOR INESPERADO!",
    "EU RIO TODA VEZ QUE VEJO!", "O PERSONAGEM MAIS ENGRAÇADO!", "ESSA REAÇÃO FOI PERFEITA!",
    "QUANDO O ANIME QUEBRA A QUARTA PAREDE!", "A PIADA MAIS INTELIGENTE!", "EU CHOREI DE RIR!",
    "ESSE MOMENTO É ICÔNICO!", "A EXPRESSÃO FACIAL PERFEITA!", "COMÉDIA QUE FAZ SENTIDO!",
    "O GAG MAIS ENGRAÇADO!", "ESSA CENA NUNCA ENVELHECE!", "HUMOR REFINADO!",
    "A SITUAÇÃO MAIS CONSTRANGEDORA!", "EU PERDI TUDO NESSA PARTE!", "O ANIME MAIS ENGRAÇADO!",
    "QUANDO A COMÉDIA É PERFEITA!", "ESSA CENA ME MATA!", "O TIMING IMPECÁVEL!",
    "HUMOR QUE FUNCIONA!", "A REAÇÃO MAIS EXAGERADA!", "EU AMO ESSE TIPO DE HUMOR!",
    "COMÉDIA PURA E SIMPLES!", "O MOMENTO MAIS ALEATÓRIO!", "ESSA PIADA PEGOU GERAL!",
    "QUANDO O ABSURDO É ENGRAÇADO!", "A CENA QUE EU MOSTRO PRA TODO MUNDO!",
]

TITULOS_EPICO = [
    "ISSO É SIMPLESMENTE ÉPICO!", "A CENA MAIS ICÔNICA!", "MASTERPIECE EM FORMA DE ANIME!",
    "ESSE MOMENTO É PERFEITO!", "ANIMAÇÃO DE OUTRO NÍVEL!", "A CENA MAIS BEM FEITA!",
    "OBRA PRIMA!", "ESSE ANIME É INCRÍVEL!", "O MOMENTO MAIS MEMORÁVEL!",
    "QUALIDADE ABSURDA!", "ESSA CENA É HISTÓRICA!", "ANIMAÇÃO IMPECÁVEL!",
    "O ÁPICE DO ANIME!", "PRODUÇÃO DE CINEMA!", "ESSE MOMENTO VAI FICAR!",
    "A TRILHA SONORA PERFEITA!", "CINEMATOGRAFIA INCRÍVEL!", "O MOMENTO MAIS BONITO!",
    "ESSA CENA É ARTE!", "DIREÇÃO IMPECÁVEL!", "O ANIME SE SUPEROU!",
    "MOMENTO HISTÓRICO!", "A ANIMAÇÃO MAIS FLUIDA!", "ESSE É O PODER DO ANIME!",
    "CENA DIGNA DE OSCAR!", "PRODUÇÃO ESPETACULAR!", "O CLÍMAX PERFEITO!",
    "ESSA CENA REPRESENTA O ANIME!", "NÍVEL DE QUALIDADE ABSURDO!", "O ESTÚDIO SE SUPEROU!",
    "MOMENTO PRA GUARDAR!", "A CENA MAIS IMPACTANTE!", "ANIMAÇÃO CINEMATOGRÁFICA!",
    "ESSE MOMENTO É ATEMPORAL!", "QUALIDADE PREMIUM!", "O ANIME EM SUA MELHOR FORMA!",
    "ESSA CENA ELEVOU O ANIME!", "PRODUÇÃO DE ELITE!", "O MOMENTO DEFINITIVO!",
    "MASTERCLASS DE ANIMAÇÃO!", "ESSA CENA É REFERÊNCIA!", "NÍVEL DE ARTE ELEVADO!",
    "O ANIME MOSTROU DO QUE É CAPAZ!", "PRODUÇÃO IMPECÁVEL!", "MOMENTO ANTOLÓGICO!",
    "A CENA QUE DEFINIU A SÉRIE!", "ANIMAÇÃO DE TIRAR O FÔLEGO!", "ESSE MOMENTO É ETERNO!",
    "QUALIDADE SEM PRECEDENTES!", "O ÁPICE DA TEMPORADA!",
]

TITULOS_REVELACAO = [
    "EU NÃO ACREDITO NISSO!", "O PLOT TWIST DO SÉCULO!", "MINHA MENTE EXPLODIU!",
    "NINGUÉM ESPERAVA ESSA!", "A REVELAÇÃO MAIS CHOCANTE!", "EU FIQUEI SEM REAÇÃO!",
    "O ANIME ME ENGANOU!", "TUDO FAZ SENTIDO AGORA!", "A VERDADE ERA ESSA?!",
    "EU PRECISO REVER TUDO!", "O MAIOR PLOT TWIST!", "MINHA TEORIA ESTAVA CERTA!",
    "ELE ERA O VILÃO O TEMPO TODO!", "A REVELAÇÃO QUE MUDOU TUDO!", "EU NÃO VI ISSO CHEGANDO!",
    "O SEGREDO FINALMENTE REVELADO!", "ESSE ANIME É GENIAL!", "A REVIRAVOLTA MAIS ÉPICA!",
    "EU FIQUEI EM CHOQUE!", "TUDO ERA UMA MENTIRA!", "O MOMENTO MAIS IMPACTANTE!",
    "ELE ESCONDEU ISSO O TEMPO TODO!", "A VERDADE POR TRÁS DE TUDO!", "MEU QUEIXO CAIU!",
    "O PLOT TWIST PERFEITO!", "EU PRECISO PROCESSAR ISSO!", "A REVELAÇÃO MAIS BEM FEITA!",
    "NADA É O QUE PARECE!", "ELE NÃO É QUEM EU PENSAVA!", "O ANIME ME SURPREENDEU!",
    "A VERDADE DOLOROSA!", "EU NUNCA VOU SUPERAR!", "O SEGREDO MAIS GUARDADO!",
    "ESSA REVELAÇÃO MUDOU O ANIME!", "EU ESTAVA CEGO O TEMPO TODO!", "A TRAIÇÃO MAIS INESPERADA!",
    "ELE PLANEJOU TUDO DESDE O INÍCIO!", "O MOMENTO QUE MUDOU A HISTÓRIA!", "EU PRECISO REASSISTIR!",
    "A PISTA ESTAVA LÁ!", "O PLOT TWIST MAIS ELABORADO!", "ELE MANIPULOU TODO MUNDO!",
    "A REVELAÇÃO QUE FEZ CHORAR!", "EU ENTENDI A REFERÊNCIA!", "O ANIME JOGOU NA MINHA CARA!",
    "A VERDADE ERA ÓBVIA!", "EU DEVERIA TER PERCEBIDO!", "O MOMENTO MAIS GENIAL!",
    "A REVELAÇÃO QUE CONECTOU TUDO!", "MEU CÉREBRO NÃO COMPUTA!",
]

TITULOS_ROMANCE = [
    "MEU CORAÇÃO NÃO AGUENTA!", "O SHIP FINALMENTE ACONTECEU!", "ESSA CENA É MUITO FOFA!",
    "ELE FINALMENTE SE DECLAROU!", "O MOMENTO MAIS ROMÂNTICO!", "ESSES DOIS SÃO PERFEITOS JUNTOS!",
    "A QUÍMICA ENTRE ELES É REAL!", "QUANDO O AMOR FALA MAIS ALTO!", "O BEIJO QUE TODOS ESPERAVAM!",
    "ESSA CENA AQUECE O CORAÇÃO!", "ELE ESPEROU ESSE MOMENTO!", "A CONFISSÃO MAIS LINDA!",
    "MINHA SHIPPER SOUL ESTÁ GRITANDO!", "O CASAL MAIS FOFO DO ANIME!", "QUANDO DUAS ALMAS SE ENCONTRAM!",
    "ELE FEZ DE TUDO POR ELA!", "A CENA QUE TODO FÃ QUERIA!", "O MOMENTO MAIS DOCE!",
    "ESSES DOIS ME DERAM DIABETES!", "QUANDO O TSUNDERE CEDE!", "A DECLARAÇÃO MAIS ÉPICA!",
    "ELE ATRAVESSOU O MUNDO POR ELA!", "O AMOR QUE SUPEROU TUDO!", "ESSA CENA É PURO AÇÚCAR!",
    "FINALMENTE JUNTOS!", "O ABRAÇO QUE CUROU TUDO!", "QUANDO O AMOR É VERDADEIRO!",
    "ELE SEMPRE A PROTEGEU!", "A PROMESSA DE FICAREM JUNTOS!", "MINHA HEART EXPLODIU!",
    "O MOMENTO MAIS KAWAII!", "ESSES DOIS SÃO GOALS!", "QUANDO O ORGULHO SOME!",
    "ELE FINALMENTE ENTENDEU!", "A CENA MAIS PURA DO ANIME!", "O CASAL QUE EU SHIPPO DEMAIS!",
    "ESSE MOMENTO FOI PERFEITO!", "QUANDO OLHARES DIZEM TUDO!", "ELE SERIA CAPAZ DE TUDO!",
    "A CENA QUE ME FEZ SUSPIRAR!", "O AMOR MAIS BONITO DO ANIME!", "ESSES DOIS ME FAZEM ACREDITAR!",
    "QUANDO O DESTINO UNE!", "ELE NUNCA DESISTIU DELA!", "A QUÍMICA É INEXPLICÁVEL!",
    "O MOMENTO MAIS INTENSO!", "ESSE CASAL É CANON NO MEU CORAÇÃO!", "QUANDO O AMOR VENCE!",
    "ELE FEZ A ESCOLHA CERTA!", "A CENA QUE EU ASSISTO MIL VEZES!",
]

TITULOS_MOTIVACIONAL = [
    "ISSO ME DEU FORÇA PRA CONTINUAR!", "A CENA MAIS MOTIVACIONAL!", "EU ME SENTI INSPIRADO!",
    "NUNCA DESISTA DOS SEUS SONHOS!", "A MENSAGEM MAIS PODEROSA!", "ESSE ANIME ME MOTIVOU!",
    "O MOMENTO DE SUPERAÇÃO!", "EU ACREDITEI NELE!", "A FORÇA DE VONTADE INCRÍVEL!",
    "ISSO ME FEZ QUERER SER MELHOR!", "O HERÓI VERDADEIRO!", "ELE PROVOU QUE ERA POSSÍVEL!",
    "A DETERMINAÇÃO MAIS FORTE!", "ESSE MOMENTO ME INSPIRA!", "QUANDO A CORAGEM FALA MAIS ALTO!",
    "ELE NÃO DESISTIU!", "A VITÓRIA MAIS MERECIDA!", "ISSO ME DEU ARREPIOS DE MOTIVAÇÃO!",
    "O ESFORÇO FOI RECOMPENSADO!", "ELE SUPEROU O IMPOSSÍVEL!", "A PERSEVERANÇA VENCEU!",
    "ESSE É O VERDADEIRO PODER!", "ELE INSPIROU A TODOS!", "O MOMENTO DE VIRADA!",
    "NUNCA É TARDE PRA MUDAR!", "ELE PROVOU TODOS ERRADOS!", "A FORÇA INTERIOR!",
    "ISSO ME FEZ ACREDITAR!", "O TRIUNFO DO ESFORÇO!", "ELE SE LEVANTOU DE NOVO!",
    "A MENSAGEM QUE EU PRECISAVA!", "O HERÓI QUE TODOS MERECEM!", "ELE LUTOU ATÉ O FIM!",
    "A CORAGEM DE CONTINUAR!", "ISSO É SER PROTAGONISTA!", "ELE FEZ O IMPOSSÍVEL POSSÍVEL!",
    "A DETERMINAÇÃO INABALÁVEL!", "O MOMENTO DE GLÓRIA!", "ELE HONROU SEU CAMINHO!",
    "A VITÓRIA DA PERSISTÊNCIA!", "ISSO ME FEZ CHORAR DE ORGULHO!", "O VERDADEIRO SIGNIFICADO DE FORÇA!",
    "ELE CARREGOU TODOS NAS COSTAS!", "A REDENÇÃO MAIS BONITA!", "QUANDO O FRACO SE TORNA FORTE!",
    "ELE TRANSFORMOU DOR EM PODER!", "A HISTÓRIA MAIS INSPIRADORA!", "O MOMENTO QUE MUDOU MINHA VIDA!",
    "ELE É A PROVA DE QUE DÁ PRA VENCER!", "A CENA MAIS EMPOLGANTE!",
]

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
        
        # CONFIGURAÇÃO B2 - v12.7.3 FORÇADA
        # IMPORTANTE: Bucket correto é KortexClipAI2
        B2_KEY_ID = os.environ.get("B2_KEY_ID", "00568702c2cbfc60000000002")
        B2_APP_KEY = os.environ.get("B2_APPLICATION_KEY", "K005aP6cXPuBIw6IakBaMHYtXx4VGq")
        B2_ENDPOINT = os.environ.get("B2_ENDPOINT", "https://s3.us-east-005.backblazeb2.com")
        
        # FORÇA o bucket correto - ignora variável de ambiente antiga
        env_bucket = os.environ.get("B2_BUCKET_NAME", "")
        if env_bucket in ["KortexAI", "kortexai", ""]:
            # Variável antiga ou vazia - usa o correto
            B2_BUCKET = "KortexClipAI2"
            logger.info(f"[B2] Bucket corrigido: {env_bucket} -> KortexClipAI2")
        else:
            B2_BUCKET = env_bucket
        
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
    """Configura fontes usando cache local com fallbacks
    
    v12.7: Suporte para fontes personalizadas em /workspace/fonts
    """
    
    # Fontes personalizadas do usuário têm prioridade (Volume persistente)
    custom_fonts = []
    if FONTS_DIR.exists():
        # Procura qualquer fonte .ttf ou .otf no diretório de fontes
        for ext in ['*.ttf', '*.TTF', '*.otf', '*.OTF']:
            custom_fonts.extend(FONTS_DIR.glob(ext))
        
        # Ordena por preferência (Impact, Roboto, Arial primeiro)
        preferred = ['impact', 'roboto', 'arial', 'helvetica', 'opensans', 'montserrat']
        def font_priority(f):
            name = f.stem.lower()
            for i, pref in enumerate(preferred):
                if pref in name:
                    return i
            return 100
        custom_fonts.sort(key=font_priority)
        
        if custom_fonts:
            logger.info(f"[FONTS] {len(custom_fonts)} fontes encontradas em {FONTS_DIR}")
    
    font_sources = [
        *custom_fonts,  # Fontes do usuário primeiro
        CACHE_DIR / "fonts" / "Impact.ttf",
        FONT_PATH,
        FONTS_DIR / "Roboto-Bold.ttf",
        FONTS_DIR / "Arial.ttf",
        FONTS_DIR / "impact.ttf",
        FONTS_DIR / "Impact.ttf",
        # Fontes do sistema
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    ]
    
    for font_path in font_sources:
        try:
            p = Path(font_path) if not isinstance(font_path, Path) else font_path
        except Exception:
            p = Path(str(font_path))

        if p.exists():
            logger.info(f"[SUCCESS] Fonte encontrada: {p}")
            return str(p)
    
    logger.warning("[WARNING] Nenhuma fonte TrueType encontrada, títulos podem não aparecer corretamente")
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

@retry_on_failure(max_attempts=3, delay=2)
def download_background(url: str) -> Optional[str]:
    """Download de background com cache e validação
    
    v15.4: Usa S3 API com credenciais quando URL pública retorna 401
    """
    logger.info("=" * 50)
    logger.info("[BACKGROUND v15.4] INICIANDO DOWNLOAD")
    logger.info(f"  URL recebida: {url}")
    logger.info("=" * 50)
    
    if not url:
        logger.info("[BACKGROUND] URL vazia ou None")
        return None
    
    # Limpa a URL
    url = str(url).strip()
    
    # Verifica valores inválidos
    if url.lower() in ["none", "null", "", "undefined"]:
        logger.info(f"[BACKGROUND] URL inválida: '{url}'")
        return None
    
    # Verifica se é uma URL válida
    if not url.startswith(('http://', 'https://')):
        logger.warning(f"[BACKGROUND] URL não começa com http/https: {url}")
        return None
    
    try:
        logger.info(f"[BACKGROUND] Processando: {url[:100]}...")
        
        # Hash da URL para cache
        url_hash = hashlib.md5(url.encode()).hexdigest()[:16]
        
        # Detecta extensão
        ext = ".png"
        url_lower = url.lower()
        if any(x in url_lower for x in [".jpg", ".jpeg"]):
            ext = ".jpg"
        elif ".webp" in url_lower:
            ext = ".webp"
        elif ".gif" in url_lower:
            ext = ".gif"
        
        # Arquivo temporário
        temp_file = TEMP_DIR / f"bg_{url_hash}_{uuid.uuid4().hex[:6]}{ext}"
        
        # Headers para evitar bloqueios
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        
        download_success = False
        got_401 = False
        
        # ========== MÉTODO 1: Download direto (URL pública) ==========
        try:
            import requests
            logger.info(f"[BACKGROUND] Método 1: URL pública...")
            
            response = requests.get(url, headers=headers, timeout=60, stream=True, allow_redirects=True)
            logger.info(f"[BACKGROUND] Status HTTP: {response.status_code}")
            
            if response.status_code == 200:
                with open(temp_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                if temp_file.exists() and temp_file.stat().st_size > 500:
                    logger.info(f"[BACKGROUND] ✓ Download OK via URL pública")
                    download_success = True
                    
            elif response.status_code == 401:
                logger.warning(f"[BACKGROUND] HTTP 401 - Bucket não é público, tentando via S3 API...")
                got_401 = True
                
        except Exception as req_error:
            logger.warning(f"[BACKGROUND] Erro URL pública: {req_error}")
        
        # ========== MÉTODO 2: Download via S3 API (quando 401) ==========
        if not download_success and 'backblazeb2.com' in url.lower():
            logger.info("[BACKGROUND] Método 2: S3 API com credenciais...")
            
            try:
                # Extrai bucket e file path da URL
                # Formatos possíveis:
                # https://f005.backblazeb2.com/file/BUCKET/path/to/file.png
                # https://s3.us-east-005.backblazeb2.com/BUCKET/path/to/file.png
                import re
                
                file_key = None
                bucket_from_url = None
                
                # Tenta formato f005 (URL pública)
                match = re.search(r'backblazeb2\.com/file/([^/]+)/(.+)', url)
                if match:
                    bucket_from_url = match.group(1)
                    file_key = match.group(2)
                    logger.info(f"[BACKGROUND] Formato f005 detectado")
                
                # Tenta formato s3 (URL S3 API)
                if not file_key:
                    match = re.search(r's3\.[^/]+\.backblazeb2\.com/([^/]+)/(.+)', url)
                    if match:
                        bucket_from_url = match.group(1)
                        file_key = match.group(2)
                        logger.info(f"[BACKGROUND] Formato S3 detectado")
                
                if file_key and bucket_from_url:
                    logger.info(f"[BACKGROUND] Bucket: {bucket_from_url}")
                    logger.info(f"[BACKGROUND] Key: {file_key}")
                    
                    # Usa o s3_client global que já está configurado
                    if s3_client and B2_AVAILABLE:
                        try:
                            logger.info(f"[BACKGROUND] Baixando via S3 API...")
                            s3_client.download_file(
                                Bucket=bucket_from_url,
                                Key=file_key,
                                Filename=str(temp_file)
                            )
                            
                            if temp_file.exists() and temp_file.stat().st_size > 500:
                                file_size = temp_file.stat().st_size
                                logger.info(f"[BACKGROUND] ✓ Download OK via S3 API ({file_size/1024:.1f} KB)")
                                download_success = True
                                
                        except Exception as s3_error:
                            logger.warning(f"[BACKGROUND] Erro S3 API: {s3_error}")
                            
                            # Tenta com get_object como alternativa
                            try:
                                logger.info("[BACKGROUND] Tentando get_object...")
                                response = s3_client.get_object(Bucket=bucket_from_url, Key=file_key)
                                with open(temp_file, 'wb') as f:
                                    f.write(response['Body'].read())
                                
                                if temp_file.exists() and temp_file.stat().st_size > 500:
                                    logger.info(f"[BACKGROUND] ✓ Download OK via get_object")
                                    download_success = True
                            except Exception as get_error:
                                logger.warning(f"[BACKGROUND] Erro get_object: {get_error}")
                    else:
                        logger.warning("[BACKGROUND] S3 client não disponível!")
                else:
                    logger.warning(f"[BACKGROUND] Não conseguiu extrair bucket/key da URL")
                    
            except Exception as b2_error:
                logger.warning(f"[BACKGROUND] Erro B2: {b2_error}")
                import traceback
                logger.warning(traceback.format_exc())
        
        # ========== MÉTODO 3: Fallback urllib ==========
        if not download_success and not got_401:
            try:
                import urllib.request
                logger.info("[BACKGROUND] Método 3: urllib...")
                
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=60) as response:
                    with open(temp_file, 'wb') as f:
                        f.write(response.read())
                
                if temp_file.exists() and temp_file.stat().st_size > 500:
                    logger.info(f"[BACKGROUND] ✓ Download OK via urllib")
                    download_success = True
                    
            except Exception as urllib_error:
                logger.warning(f"[BACKGROUND] Erro urllib: {urllib_error}")
        
        # ========== MÉTODO 4: Fallback curl ==========
        if not download_success and not got_401:
            try:
                logger.info("[BACKGROUND] Método 4: curl...")
                cmd = ['curl', '-L', '-s', '-o', str(temp_file), '-A', headers['User-Agent'], url]
                result = subprocess.run(cmd, capture_output=True, timeout=60)
                
                if temp_file.exists() and temp_file.stat().st_size > 500:
                    logger.info(f"[BACKGROUND] ✓ Download OK via curl")
                    download_success = True
                    
            except Exception as curl_error:
                logger.warning(f"[BACKGROUND] Erro curl: {curl_error}")
        
        # ========== RESULTADO ==========
        if download_success and temp_file.exists():
            file_size = temp_file.stat().st_size
            logger.info("=" * 50)
            logger.info(f"[BACKGROUND] ✓ SUCESSO!")
            logger.info(f"  Arquivo: {temp_file}")
            logger.info(f"  Tamanho: {file_size/1024:.1f} KB")
            logger.info("=" * 50)
            return str(temp_file)
            
    except Exception as e:
        logger.error(f"[BACKGROUND] Erro geral: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.error("=" * 50)
    logger.error("[BACKGROUND] ✗ TODAS AS TENTATIVAS FALHARAM")
    if got_401:
        logger.error("[BACKGROUND] ⚠️ CAUSA: Bucket B2 retornou 401 (não público)")
        logger.error("[BACKGROUND] ⚠️ E o download via S3 API também falhou")
        logger.error("[BACKGROUND] ⚠️ VERIFICAR: Credenciais B2 têm permissão de leitura?")
    logger.error("=" * 50)
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

def apply_antishadowban(clip, options: Dict = None):
    """
    Aplica transformações para tornar vídeo único com segurança
    
    Args:
        clip: Clip do MoviePy
        options: Dict com opções individuais:
            - enabled: bool - Ativar anti-shadowban
            - mirror: bool - Espelhar vídeo
            - colorGrading: bool - Aplicar color grading aleatório
            - microZoom: bool - Aplicar micro-zoom (respiração)
            - filmGrain: bool - Adicionar ruído de película
    """
    if not MOVIEPY_AVAILABLE:
        logger.warning("[WARNING] MoviePy não disponível, pulando anti-shadowban")
        return clip
    
    # Retrocompatibilidade: se options é None ou não é dict, usa defaults
    if options is None or not isinstance(options, dict):
        options = {
            "enabled": True,
            "mirror": True,
            "colorGrading": True,
            "microZoom": False,
            "filmGrain": False
        }
    
    if not options.get("enabled", True):
        logger.info("[ANTI-SHADOWBAN] Desativado pelo usuário")
        return clip
    
    logger.info("[ANTI-SHADOWBAN] Aplicando transformações...")
    
    try:
        modifications = []
        
        # Espelhamento (controlável pelo usuário, com aleatoriedade se ativado)
        if options.get("mirror", True):
            if random.choice([True, False]):  # 50% chance mesmo quando ativado
                clip = clip.fx(moviepy_imports['mirror_x'])
                modifications.append("espelhamento")
        
        # Color Grading aleatório
        if options.get("colorGrading", True):
            gamma_val = random.uniform(0.95, 1.05)
            contrast_val = random.uniform(0.96, 1.04)
            
            clip = clip.fx(moviepy_imports['gamma_corr'], gamma_val)
            clip = clip.fx(moviepy_imports['colorx'], contrast_val)
            modifications.append(f"gamma={gamma_val:.2f}")
            modifications.append(f"contraste={contrast_val:.2f}")
        
        # Micro-zoom (respiração) - zoom muito sutil
        if options.get("microZoom", False):
            try:
                # Zoom muito sutil de 1.01 a 1.03
                zoom_factor = random.uniform(1.01, 1.03)
                w, h = clip.size
                new_w = int(w * zoom_factor)
                new_h = int(h * zoom_factor)
                
                # Redimensiona e recorta para manter tamanho original
                clip = clip.resize((new_w, new_h))
                x_offset = (new_w - w) // 2
                y_offset = (new_h - h) // 2
                clip = clip.crop(x1=x_offset, y1=y_offset, x2=x_offset+w, y2=y_offset+h)
                modifications.append(f"micro-zoom={zoom_factor:.2f}x")
            except Exception as e:
                logger.warning(f"[WARNING] Micro-zoom falhou: {e}")
        
        # Crop sutil (sempre aplicado se colorGrading está ativo)
        if options.get("colorGrading", True):
            if random.choice([True, False]):
                crop_pixels = random.randint(1, 3)
                w, h = clip.size
                if w > crop_pixels * 4 and h > crop_pixels * 4:
                    clip = clip.crop(
                        x1=crop_pixels, 
                        y1=crop_pixels, 
                        x2=w-crop_pixels, 
                        y2=h-crop_pixels
                    )
                    modifications.append(f"crop={crop_pixels}px")
        
        # Film Grain (ruído de película) - implementação básica
        if options.get("filmGrain", False):
            try:
                # Adiciona ruído muito sutil ajustando contraste rapidamente
                # Nota: Film grain real requer processamento frame a frame
                noise_intensity = random.uniform(0.98, 1.02)
                clip = clip.fx(moviepy_imports['colorx'], noise_intensity)
                modifications.append("film-grain")
            except Exception as e:
                logger.warning(f"[WARNING] Film grain falhou: {e}")
        
        if modifications:
            logger.info(f"[ANTI-SHADOWBAN] Aplicado: {', '.join(modifications)}")
        else:
            logger.info("[ANTI-SHADOWBAN] Nenhuma modificação aplicada")
        
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
    pos_vertical: float = 0.15,
    font_family: str = None
):
    """
    Renderiza título simples com validação robusta
    
    v14.2: Logs detalhados + Correção de tamanho de fonte
    """
    if not PIL_AVAILABLE:
        logger.warning("[WARNING] PIL não disponível para criar título")
        return None
    
    if not MOVIEPY_AVAILABLE:
        logger.warning("[WARNING] MoviePy não disponível para criar título")
        return None
    
    try:
        # ========== LOG DE DIAGNÓSTICO ==========
        logger.info("=" * 60)
        logger.info("[TITULO v14.2] CRIANDO TÍTULO")
        logger.info(f"  Texto: '{texto[:50]}'")
        logger.info(f"  font_size RECEBIDO: {font_size} (tipo: {type(font_size).__name__})")
        logger.info(f"  text_color: {text_color}")
        logger.info(f"  stroke_color: {stroke_color}")
        logger.info(f"  stroke_width: {stroke_width}")
        logger.info(f"  pos_vertical: {pos_vertical}")
        logger.info(f"  font_family: {font_family}")
        logger.info("=" * 60)
        
        # Valida parâmetros
        if not texto or len(texto.strip()) == 0:
            logger.warning("[TITULO] Texto vazio!")
            return None
        
        if largura_video <= 0 or altura_video <= 0 or duracao <= 0:
            logger.warning("[WARNING] Parâmetros de vídeo inválidos para título")
            return None
        
        # GARANTE que font_size é um inteiro válido
        original_font_size = font_size
        if isinstance(font_size, str):
            try:
                font_size = int(float(font_size))
                logger.info(f"[TITULO] Convertido font_size de string para int: {font_size}")
            except:
                font_size = 80
                logger.warning(f"[TITULO] font_size inválido '{original_font_size}', usando 80")
        
        font_size = max(30, min(300, int(font_size)))  # Entre 30 e 300 pixels
        logger.info(f"[TITULO] font_size FINAL: {font_size}")
        
        # Cria imagem para o texto - altura proporcional ao tamanho da fonte
        img_h = max(font_size * 4, int(altura_video * 0.4))
        img = Image.new('RGBA', (largura_video, img_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Tenta carregar fonte customizada ou padrão
        font = None
        font_path = None
        
        # v15.5b: Múltiplos diretórios de fontes (sistema + customizadas)
        font_dirs = [
            Path("/workspace/fonts"),
            Path("/app/fonts"),
            Path("/usr/local/share/fonts/custom"),
            Path("/usr/share/fonts/truetype/dejavu"),
            Path("/usr/share/fonts/truetype/liberation"),
            Path("/usr/share/fonts/truetype/ubuntu"),
            Path("/usr/share/fonts/truetype/freefont"),
            Path("/usr/share/fonts/truetype")
        ]
        
        # Mapeamento de nomes de fontes para arquivos do sistema
        system_font_mapping = {
            'dejavusans-bold': '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
            'dejavusans': '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            'liberation': '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
            'liberationsans-bold': '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
            'ubuntu': '/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf',
            'ubuntu-bold': '/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf',
            'freesans': '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
            'freesansbold': '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
        }
        
        # Se font_family foi especificado, procura
        if font_family:
            logger.info(f"[TITULO] Procurando fonte '{font_family}'...")
            
            # Primeiro, verifica mapeamento do sistema
            font_key = font_family.lower().replace(' ', '').replace('-', '')
            if font_key in system_font_mapping:
                mapped_path = system_font_mapping[font_key]
                if os.path.exists(mapped_path):
                    font_path = mapped_path
                    logger.info(f"[TITULO] ✓ Fonte mapeada do sistema: {font_path}")
            
            # Se não encontrou no mapeamento, procura nas pastas
            if not font_path:
                for fonts_dir in font_dirs:
                    if not fonts_dir.exists():
                        continue
                        
                    possible_extensions = ['.ttf', '.otf', '.TTF', '.OTF', '']
                    
                    # Procura pelo nome exato
                    for ext in possible_extensions:
                        candidate = fonts_dir / f"{font_family}{ext}"
                        if candidate.exists():
                            font_path = str(candidate)
                            logger.info(f"[TITULO] ✓ Fonte encontrada: {font_path}")
                            break
                    
                    if font_path:
                        break
                    
                    # Se não encontrou, procura case-insensitive
                    try:
                        for f in fonts_dir.iterdir():
                            if f.is_file():
                                f_lower = f.stem.lower().replace(' ', '').replace('-', '')
                                search_lower = font_family.lower().replace(' ', '').replace('-', '')
                                
                                # Verifica nome exato
                                if f_lower == search_lower and f.suffix.lower() in ['.ttf', '.otf']:
                                    font_path = str(f)
                                    logger.info(f"[TITULO] ✓ Fonte encontrada (case-insensitive): {font_path}")
                                    break
                                # Verifica se contém o nome
                                if search_lower in f_lower and f.suffix.lower() in ['.ttf', '.otf']:
                                    font_path = str(f)
                                    logger.info(f"[TITULO] ✓ Fonte parcial encontrada: {font_path}")
                                    break
                    except:
                        pass
                    
                    if font_path:
                        break
            
            # Se ainda não encontrou, tenta baixar do B2 (fontes customizadas)
            if not font_path and s3_client and B2_AVAILABLE:
                try:
                    custom_fonts_b2 = [
                        'Heinan.otf', 'HeroesLegend.ttf', 'Karina.ttf', 'PersonaAura.otf',
                        'SuperCrawler.ttf', 'Kotton.otf', 'Blustrue.otf', 'Basketball.otf'
                    ]
                    
                    for b2_font in custom_fonts_b2:
                        if font_family.lower() in b2_font.lower().replace('.otf', '').replace('.ttf', ''):
                            b2_key = f"fonts/{b2_font}"
                            local_path = Path("/workspace/fonts") / b2_font
                            
                            try:
                                logger.info(f"[TITULO] Tentando baixar fonte do B2: {b2_key}")
                                local_path.parent.mkdir(parents=True, exist_ok=True)
                                s3_client.download_file(B2_BUCKET, b2_key, str(local_path))
                                
                                if local_path.exists():
                                    font_path = str(local_path)
                                    logger.info(f"[TITULO] ✓ Fonte baixada do B2: {font_path}")
                                    break
                            except Exception as b2_err:
                                logger.debug(f"[TITULO] Fonte não encontrada no B2: {b2_err}")
                except Exception as e:
                    logger.debug(f"[TITULO] Erro ao buscar fonte no B2: {e}")
            
            if not font_path:
                logger.warning(f"[TITULO] ✗ Fonte '{font_family}' NÃO encontrada, usando fallback")
        
        # Se não encontrou fonte customizada, usa FONT_TO_USE padrão
        if not font_path and FONT_TO_USE and os.path.exists(FONT_TO_USE):
            font_path = FONT_TO_USE
            logger.info(f"[TITULO] Usando fonte padrão: {font_path}")
        
        # Carrega a fonte COM O TAMANHO CORRETO
        if font_path:
            try:
                font = ImageFont.truetype(font_path, font_size)
                logger.info(f"[TITULO] ✓ Fonte carregada: {font_path} tamanho={font_size}")
            except Exception as e:
                logger.warning(f"[WARNING] Erro ao carregar fonte {font_path}: {e}")
        
        # Se não conseguiu carregar fonte, tenta fallbacks do sistema
        if font is None:
            fallback_fonts = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"
            ]
            
            for fb_font in fallback_fonts:
                if os.path.exists(fb_font):
                    try:
                        font = ImageFont.truetype(fb_font, font_size)
                        logger.info(f"[TITULO] Usando fonte fallback: {fb_font} tamanho={font_size}")
                        break
                    except:
                        continue
        
        # Último recurso: fonte padrão PIL (mas MANTÉM tamanho grande simulado)
        if font is None:
            try:
                font = ImageFont.load_default()
                # NÃO reduz o tamanho! Apenas loga warning
                logger.warning(f"[TITULO] ⚠ Usando fonte padrão PIL - tamanho visual pode variar")
            except:
                logger.error("[TITULO] ✗ Não foi possível carregar NENHUMA fonte!")
                return None
        
        # ==================== v15.9: MÁXIMO 2 LINHAS COM AUTO-REDIMENSIONAMENTO ====================
        # Margem lateral (10% de cada lado = 80% da largura útil)
        margem_lateral = int(largura_video * 0.10)
        largura_util = largura_video - (margem_lateral * 2)
        
        # Cria imagem temporária para calcular tamanho do texto
        temp_img = Image.new('RGBA', (largura_video, 100), (0, 0, 0, 0))
        draw = ImageDraw.Draw(temp_img)
        
        def calcular_linhas_com_fonte(texto_full, fonte_atual, max_largura):
            """Calcula quantas linhas o texto ocuparia com a fonte atual"""
            palavras = texto_full.strip().split()
            if not palavras:
                return [], 0
            
            linhas = []
            linha_atual = ""
            
            for palavra in palavras:
                teste = f"{linha_atual} {palavra}".strip() if linha_atual else palavra
                try:
                    bbox = draw.textbbox((0, 0), teste, font=fonte_atual)
                    largura_teste = bbox[2] - bbox[0]
                except:
                    largura_teste = len(teste) * (font_size // 2)
                
                if largura_teste <= max_largura:
                    linha_atual = teste
                else:
                    if linha_atual:
                        linhas.append(linha_atual)
                    linha_atual = palavra
            
            if linha_atual:
                linhas.append(linha_atual)
            
            return linhas, len(linhas)
        
        # Tenta com o tamanho original
        linhas, num_linhas = calcular_linhas_com_fonte(texto, font, largura_util)
        
        # Se mais de 2 linhas, REDUZ a fonte até caber em 2 linhas
        tamanho_atual = font_size
        min_font_size = 40  # Tamanho mínimo legível
        
        while num_linhas > 2 and tamanho_atual > min_font_size:
            tamanho_atual -= 5  # Reduz 5px por vez
            
            # Recarrega fonte com novo tamanho
            try:
                if font_path:
                    font = ImageFont.truetype(font_path, tamanho_atual)
                else:
                    # Usa fallback
                    for fb_font in [
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
                    ]:
                        if os.path.exists(fb_font):
                            font = ImageFont.truetype(fb_font, tamanho_atual)
                            break
            except:
                pass
            
            linhas, num_linhas = calcular_linhas_com_fonte(texto, font, largura_util)
        
        # Se ainda não coube, força 2 linhas cortando no meio
        if num_linhas > 2:
            palavras = texto.strip().split()
            meio = len(palavras) // 2
            linhas = [
                " ".join(palavras[:meio]),
                " ".join(palavras[meio:])
            ]
            logger.warning(f"[TITULO] Forçando 2 linhas (texto muito longo)")
        
        # Garante MAX 2 linhas
        linhas = linhas[:2]
        
        # Remove linhas vazias
        linhas = [l for l in linhas if l.strip()]
        
        # Se ficou vazio, usa texto original em 1 linha
        if not linhas:
            linhas = [texto[:50]]  # Trunca se necessário
        
        logger.info(f"[TITULO] Linhas finais ({len(linhas)}): {linhas}")
        logger.info(f"[TITULO] Fonte final: {tamanho_atual}px (original: {font_size}px)")
        
        # Atualiza font_size para o valor final
        font_size = tamanho_atual
        
        # Cores
        text_rgb = hex_to_rgb(text_color)
        stroke_rgb = hex_to_rgb(stroke_color)
        
        logger.info(f"[TITULO] Cores: texto={text_rgb}, borda={stroke_rgb}")
        
        # ==================== CRIA IMAGEM DO TÍTULO ====================
        # Altura da imagem baseada no número de linhas
        line_spacing = int(font_size * 1.3)
        img_h = line_spacing * len(linhas) + 40  # +40 para padding
        
        # Cria imagem transparente
        img = Image.new('RGBA', (largura_video, img_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Desenha texto centralizado
        y_pos = 20  # Padding top
        
        for linha in linhas:
            if not linha.strip():
                continue
            
            # Calcula posição central
            try:
                bbox = draw.textbbox((0, 0), linha, font=font)
                text_width = bbox[2] - bbox[0]
            except:
                text_width = len(linha) * (font_size // 2)
            
            x_pos = max(margem_lateral, (largura_video - text_width) // 2)
            
            # Contorno (borda)
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
        
        # Posição vertical (pos_vertical é 0.0 a 1.0)
        # 0.0 = topo, 0.5 = meio, 1.0 = fundo
        pos_y = max(0, min(altura_video - img_h, int(altura_video * pos_vertical)))
        clip = clip.set_position(('center', pos_y))
        
        logger.info(f"[TITULO] ✓ Criado com sucesso: '{texto[:30]}' ({len(linhas)} linhas, tamanho={font_size})")
        return clip
        
    except Exception as e:
        logger.error(f"[TITULO] ✗ Erro ao criar título: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# ==================== WHISPER GPU ULTRA-ESTABILIZADO (MANAGER THREAD-SAFE) ====================

def criar_titulo_png_transparente(texto, largura_video, altura_video, pos_vertical, title_style, output_path):
    """
    Cria PNG do título com fundo TRANSPARENTE (canal alpha correto).
    Retorna o path do PNG ou None se falhar.
    
    IMPORTANTE: Esta função NÃO usa MoviePy para preservar a transparência!
    """
    try:
        if not PIL_AVAILABLE:
            return None
        
        # Configurações do título
        font_size = title_style.get("fontSize", 70)
        text_color = title_style.get("textColor", "#FFFFFF")
        stroke_color = title_style.get("strokeColor", "#000000")
        stroke_width = title_style.get("strokeWidth", 4)
        font_family = title_style.get("fontFamily", None)
        
        # Margens
        margem_lateral = int(largura_video * 0.10)  # 10% de cada lado
        largura_util = largura_video - (margem_lateral * 2)
        
        # Carrega fonte
        font = None
        font_paths_to_try = []
        
        if font_family:
            # Tenta encontrar a fonte pelo nome
            for fonts_dir in [FONTS_DIR, Path("/app/fonts"), Path("/workspace/fonts")]:
                if fonts_dir.exists():
                    for f in fonts_dir.glob("*"):
                        if f.suffix.lower() in ['.ttf', '.otf']:
                            fname = f.stem.lower().replace(' ', '').replace('-', '').replace('_', '')
                            target = font_family.lower().replace(' ', '').replace('-', '').replace('_', '')
                            if fname == target or target in fname:
                                font_paths_to_try.insert(0, str(f))
        
        # Adiciona fontes padrão
        for fonts_dir in [FONTS_DIR, Path("/app/fonts"), Path("/workspace/fonts")]:
            if fonts_dir.exists():
                for f in fonts_dir.glob("*.ttf"):
                    font_paths_to_try.append(str(f))
                for f in fonts_dir.glob("*.otf"):
                    font_paths_to_try.append(str(f))
        
        # Tenta carregar fonte
        for font_path in font_paths_to_try:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue
        
        if font is None:
            font = ImageFont.load_default()
        
        # Quebra texto em linhas (MAX 2 LINHAS)
        palavras = texto.split()
        linhas = []
        linha_atual = ""
        
        temp_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_img)
        
        for palavra in palavras:
            teste = f"{linha_atual} {palavra}".strip() if linha_atual else palavra
            try:
                bbox = temp_draw.textbbox((0, 0), teste, font=font)
                w = bbox[2] - bbox[0]
            except:
                w = len(teste) * (font_size // 2)
            
            if w <= largura_util:
                linha_atual = teste
            else:
                if linha_atual:
                    linhas.append(linha_atual)
                linha_atual = palavra
        
        if linha_atual:
            linhas.append(linha_atual)
        
        # Reduz fonte se mais de 2 linhas
        min_font_size = 40
        while len(linhas) > 2 and font_size > min_font_size:
            font_size -= 5
            try:
                for font_path in font_paths_to_try:
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                        break
                    except:
                        continue
            except:
                pass
            
            # Recalcula linhas
            linhas = []
            linha_atual = ""
            for palavra in palavras:
                teste = f"{linha_atual} {palavra}".strip() if linha_atual else palavra
                try:
                    bbox = temp_draw.textbbox((0, 0), teste, font=font)
                    w = bbox[2] - bbox[0]
                except:
                    w = len(teste) * (font_size // 2)
                
                if w <= largura_util:
                    linha_atual = teste
                else:
                    if linha_atual:
                        linhas.append(linha_atual)
                    linha_atual = palavra
            
            if linha_atual:
                linhas.append(linha_atual)
        
        # Força máximo 2 linhas
        linhas = linhas[:2]
        
        # Calcula dimensões do título
        line_spacing = int(font_size * 1.3)
        titulo_h = line_spacing * len(linhas) + 40
        
        # Converte cores
        text_rgb = hex_to_rgb(text_color)
        stroke_rgb = hex_to_rgb(stroke_color)
        
        # ==================== CRIA IMAGEM COMPLETA 1080x1920 COM TRANSPARÊNCIA ====================
        # Cria imagem do tamanho total do vídeo com fundo TOTALMENTE TRANSPARENTE
        full_img = Image.new('RGBA', (largura_video, altura_video), (0, 0, 0, 0))
        draw = ImageDraw.Draw(full_img)
        
        # Calcula posição Y do título
        titulo_y = int(altura_video * pos_vertical)
        titulo_y = max(0, min(altura_video - titulo_h, titulo_y))
        
        # Desenha cada linha
        y_pos = titulo_y + 20  # Padding top
        
        for linha in linhas:
            if not linha.strip():
                continue
            
            # Calcula posição central
            try:
                bbox = draw.textbbox((0, 0), linha, font=font)
                text_width = bbox[2] - bbox[0]
            except:
                text_width = len(linha) * (font_size // 2)
            
            x_pos = max(margem_lateral, (largura_video - text_width) // 2)
            
            # Contorno (borda) - desenha várias vezes ao redor
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
                                fill=stroke_rgb + (255,)  # Adiciona alpha opaco
                            )
            
            # Texto principal
            draw.text(
                (x_pos, y_pos),
                linha,
                font=font,
                fill=text_rgb + (255,)  # Adiciona alpha opaco
            )
            
            y_pos += line_spacing
        
        # Salva PNG com transparência
        full_img.save(str(output_path), 'PNG')
        
        logger.info(f"[TITULO PNG] ✓ Criado com transparência: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"[TITULO PNG] ✗ Erro: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

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

# ==================== ANÁLISE DE VÍDEO COM IA GPU v14.0 ====================

@safe_gpu_operation
def analyze_video_content_gpu(
    video_path: str, 
    anime_name: str,
    scene_preference: str = "balanced",
    cut_duration: Dict = None
) -> List[Dict]:
    """
    v14.0: ANÁLISE COMPLETA SEM LIMITE DE CORTES
    
    - Ouve e transcreve TODO o episódio
    - Detecta TODAS as cenas de ação e diálogo importantes
    - Gera TÍTULOS ÚNICOS baseados no conteúdo real de cada cena
    - SEM LIMITE de cortes (gera quantos forem necessários)
    - GARANTE que não há cortes duplicados
    
    Args:
        video_path: Caminho do vídeo
        anime_name: Nome do anime (para contexto)
        scene_preference: "balanced", "action", "dialogue", "humor"
        cut_duration: {"min": X, "max": Y} em segundos
    """
    
    if cut_duration is None:
        cut_duration = {"min": 30, "max": 90}
    
    min_duration = max(15, cut_duration.get("min", 30))
    max_duration = min(180, cut_duration.get("max", 90))
    
    logger.info("=" * 60)
    logger.info("[ANÁLISE v14.0] INICIANDO ANÁLISE COMPLETA DO EPISÓDIO")
    logger.info(f"  Anime: {anime_name}")
    logger.info(f"  Preferência: {scene_preference}")
    logger.info(f"  Duração dos cortes: {min_duration}s - {max_duration}s")
    logger.info(f"  MODO: SEM LIMITE DE CORTES")
    logger.info("=" * 60)
    
    if not AI_AVAILABLE or not WHISPER_AVAILABLE:
        logger.warning("[WARNING] IA não disponível, usando análise básica")
        return generate_smart_fallback_cuts(video_path, anime_name, min_duration, max_duration)
    
    audio_files_to_clean = []
    
    try:
        # ==================== ETAPA 1: CARREGAR WHISPER ====================
        if not whisper_manager.loaded():
            logger.info("[WHISPER] Carregando modelo de transcrição...")
            if not load_turbo_whisper_gpu():
                logger.error("[ERROR] Falha ao carregar Whisper")
                return generate_smart_fallback_cuts(video_path, anime_name, min_duration, max_duration)
        
        # ==================== ETAPA 2: EXTRAIR ÁUDIO ====================
        logger.info("[AUDIO] Extraindo áudio do episódio...")
        raw_audio = TEMP_DIR / f"audio_{uuid.uuid4().hex[:8]}.wav"
        audio_files_to_clean.append(raw_audio)
        
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            str(raw_audio), '-y',
            '-hide_banner', '-loglevel', 'error'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0 or not validate_audio_file(raw_audio):
            raise Exception(f"Falha na extração de áudio")
        
        logger.info("[AUDIO] Áudio extraído com sucesso")
        
        # ==================== ETAPA 3: LIMPAR ÁUDIO ====================
        logger.info("[AUDIO] Processando e limpando áudio...")
        clean_audio = clean_audio_deepfilter(raw_audio)
        
        if clean_audio != raw_audio:
            audio_files_to_clean.append(clean_audio)
        
        if not validate_audio_file(clean_audio):
            clean_audio = raw_audio
        
        # ==================== ETAPA 4: TRANSCREVER EPISÓDIO COMPLETO ====================
        logger.info("[TRANSCRIÇÃO] Transcrevendo episódio completo...")
        transcription_result = transcrever_com_whisper_gpu(str(clean_audio))
        
        # Processa todos os segmentos de transcrição
        all_segments = []
        for seg in transcription_result.get("chunks", []):
            text = seg.get("text", "").strip()
            if text and len(text) > 3:
                start, end = seg.get("timestamp", (0, 0))
                if start is not None and end is not None and end > start:
                    all_segments.append({
                        "start": float(start),
                        "end": float(end),
                        "text": text,
                        "type": "dialogue",
                        "word_count": len(text.split()),
                        "has_emotion": any(char in text for char in "!?...")
                    })
        
        logger.info(f"[TRANSCRIÇÃO] {len(all_segments)} segmentos de diálogo encontrados")
        
        # ==================== ETAPA 5: DETECTAR CENAS DE AÇÃO ====================
        logger.info("[AÇÃO] Analisando cenas de ação...")
        try:
            detector = ActionDetector(video_path)
            action_scenes = detector.detect_high_energy_segments(threshold=50.0, min_duration=3.0)
            logger.info(f"[AÇÃO] {len(action_scenes)} cenas de ação detectadas")
        except Exception as e:
            logger.warning(f"[AÇÃO] Erro na detecção: {e}")
            action_scenes = []
        
        # ==================== ETAPA 6: OBTER DURAÇÃO DO VÍDEO ====================
        try:
            video = moviepy_imports['VideoFileClip'](video_path)
            video_duration = video.duration
            video.close()
        except:
            cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                   '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            video_duration = float(result.stdout.strip()) if result.returncode == 0 else 1200
        
        logger.info(f"[VÍDEO] Duração total: {video_duration:.1f}s ({video_duration/60:.1f} minutos)")
        
        # ==================== ETAPA 7: IDENTIFICAR MOMENTOS IMPORTANTES ====================
        logger.info("[ANÁLISE] Identificando momentos importantes...")
        
        important_moments = []
        
        # Função helper para buscar transcrição em um intervalo de tempo
        def get_transcription_for_range(start_time, end_time, segments):
            """Busca a melhor frase da transcrição no intervalo de tempo"""
            best_text = ""
            best_score = 0
            
            for seg in segments:
                # Verifica se o segmento está dentro do intervalo
                if seg["start"] >= start_time - 5 and seg["end"] <= end_time + 5:
                    text = seg["text"].strip()
                    
                    # Calcula score baseado em critérios
                    score = len(text.split())  # Mais palavras = melhor
                    
                    # Bonus para frases com emoção
                    if seg.get("has_emotion"):
                        score += 10
                    
                    # Bonus para palavras de impacto
                    impact_words = ["eu", "você", "poder", "força", "nunca", "sempre", 
                                   "vou", "proteger", "morrer", "viver", "destino"]
                    text_lower = text.lower()
                    score += sum(3 for word in impact_words if word in text_lower)
                    
                    if score > best_score and len(text) > 10:
                        best_score = score
                        best_text = text
            
            return best_text
        
        # Adiciona cenas de ação COM TRANSCRIÇÃO REAL
        for action in action_scenes:
            # Busca a melhor frase da transcrição durante essa cena
            action_text = get_transcription_for_range(
                action["start"], 
                action["end"], 
                all_segments
            )
            
            # Se não achou transcrição, usa marcador genérico
            if not action_text:
                action_text = ""  # Vazio vai para fallback
            
            important_moments.append({
                "start": action["start"],
                "end": action["end"],
                "type": "action",
                "score": action.get("score", 70),
                "text": action_text,
                "title_hint": "action"
            })
        
        # Adiciona diálogos importantes (frases longas ou emocionais)
        for seg in all_segments:
            importance_score = 0
            
            # Frases mais longas são mais importantes
            if seg["word_count"] >= 8:
                importance_score += 30
            elif seg["word_count"] >= 5:
                importance_score += 20
            
            # Frases com emoção
            if seg["has_emotion"]:
                importance_score += 25
            
            # Palavras-chave de anime
            keywords = ["poder", "força", "luta", "proteger", "amigo", "inimigo", 
                       "batalha", "morrer", "viver", "sonho", "nunca", "sempre",
                       "promessa", "destino", "mundo", "coração", "alma"]
            text_lower = seg["text"].lower()
            keyword_count = sum(1 for kw in keywords if kw in text_lower)
            importance_score += keyword_count * 15
            
            if importance_score >= 30:
                important_moments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "type": "dialogue",
                    "score": min(95, importance_score),
                    "text": seg["text"],
                    "title_hint": "dialogue"
                })
        
        # Ordena por score
        important_moments.sort(key=lambda x: x["score"], reverse=True)
        
        logger.info(f"[ANÁLISE] {len(important_moments)} momentos importantes identificados")
        
        # ==================== ETAPA 8: FILTRAR POR PREFERÊNCIA ====================
        if scene_preference == "action":
            filtered_moments = [m for m in important_moments if m["type"] == "action"]
            filtered_moments += [m for m in important_moments if m["type"] != "action"][:5]
        elif scene_preference == "dialogue":
            filtered_moments = [m for m in important_moments if m["type"] == "dialogue"]
            filtered_moments += [m for m in important_moments if m["type"] != "dialogue"][:5]
        elif scene_preference == "humor":
            # Humor: frases curtas com emoção
            filtered_moments = [m for m in important_moments 
                               if m["type"] == "dialogue" and m.get("text", "").count("!") > 0]
            filtered_moments += important_moments[:10]
        else:  # balanced
            filtered_moments = important_moments
        
        logger.info(f"[ANÁLISE] {len(filtered_moments)} momentos após filtro de preferência")
        
        # ==================== ETAPA 9: GERAR CORTES SEM LIMITE ====================
        logger.info("[CORTES] Gerando cortes únicos...")
        
        cuts = []
        used_ranges = []  # Para evitar duplicação
        used_titles = set()  # v15.6: Evita títulos repetidos
        
        def ranges_overlap(start1, end1, start2, end2, min_gap=10):
            """Verifica se dois ranges se sobrepõem"""
            return not (end1 + min_gap < start2 or end2 + min_gap < start1)
        
        def is_range_used(start, end):
            """Verifica se o range já foi usado"""
            for used_start, used_end in used_ranges:
                if ranges_overlap(start, end, used_start, used_end):
                    return True
            return False
        
        def generate_unique_title_v158(text, moment_type, index, anime_name, start_time, all_segs):
            """
            v15.8: GERAÇÃO DE TÍTULOS - CORRIGIDA
            
            REGRAS:
            1. NUNCA usa nome do anime como título
            2. NUNCA repete títulos (tracked via used_titles)
            3. USA texto da transcrição se for BOM
            4. USA biblioteca de 400+ títulos como fallback
            """
            
            def detectar_tipo_cena(texto):
                """Detecta tipo de cena baseado no texto"""
                if not texto:
                    return moment_type or 'epic'
                
                texto_lower = texto.lower()
                
                if any(w in texto_lower for w in ['luta', 'batalha', 'ataque', 'golpe', 'matar', 'destruir', 'poder', 'força']):
                    return 'action'
                if any(w in texto_lower for w in ['amo', 'amor', 'gostar', 'coração', 'te amo', 'gosto']):
                    return 'romance'
                if any(w in texto_lower for w in ['chorar', 'lágrimas', 'morreu', 'perdi', 'adeus', 'saudade', 'triste']):
                    return 'emotion'
                if any(w in texto_lower for w in ['haha', 'idiota', 'burro', 'engraçado', 'ridículo']):
                    return 'humor'
                if any(w in texto_lower for w in ['verdade', 'segredo', 'mentira', 'traidor', 'não acredito']):
                    return 'revelation'
                if any(w in texto_lower for w in ['nunca desistir', 'sonho', 'acreditar', 'conseguir', 'vencer']):
                    return 'motivational'
                
                return moment_type or 'epic'
            
            def escolher_da_biblioteca(tipo, idx):
                """Escolhe título da biblioteca baseado no tipo"""
                bibliotecas = {
                    'action': TITULOS_ACAO,
                    'acao': TITULOS_ACAO,
                    'dialogue': TITULOS_DIALOGO,
                    'dialogo': TITULOS_DIALOGO,
                    'emotion': TITULOS_EMOCAO,
                    'emocao': TITULOS_EMOCAO,
                    'romance': TITULOS_ROMANCE,
                    'humor': TITULOS_HUMOR,
                    'revelation': TITULOS_REVELACAO,
                    'motivational': TITULOS_MOTIVACIONAL,
                    'epic': TITULOS_EPICO,
                    'segment': TITULOS_EPICO,
                    'fallback': TITULOS_EPICO,
                }
                
                lista = bibliotecas.get(tipo.lower(), TITULOS_EPICO)
                
                # Usa combinação de idx e um contador para garantir variedade
                seed_value = idx * 137 + int(start_time * 10) % 1000
                random.seed(seed_value)
                shuffled = lista.copy()
                random.shuffle(shuffled)
                
                # Encontra um título não usado
                for titulo in shuffled:
                    if titulo not in used_titles:
                        return titulo
                
                # Se todos usados, cria variação única
                base = shuffled[idx % len(shuffled)]
                for i in range(100):
                    modified = f"{base[:-1]} #{idx+1+i}!"
                    if modified not in used_titles:
                        return modified
                
                # Último recurso
                return f"CENA ÉPICA #{idx+1}!"
            
            def tentar_criar_do_texto(texto):
                """Tenta criar título a partir do texto da transcrição"""
                if not texto or texto.startswith("[") or len(texto.strip()) <= 8:
                    return None
                
                clean_text = texto.strip().replace("[", "").replace("]", "")
                palavras = clean_text.split()
                
                # Se tem menos de 3 palavras, rejeita
                if len(palavras) < 3:
                    return None
                
                # Verifica se é frase de verdade
                texto_lower = clean_text.lower()
                palavras_de_frase = ['eu', 'você', 'ele', 'ela', 'nós', 'vou', 'vai', 'preciso', 
                                    'quero', 'nunca', 'sempre', 'não', 'sim', 'como', 'porque',
                                    'fazer', 'ser', 'ter', 'poder', 'dever', 'meu', 'seu']
                
                if not any(p in texto_lower for p in palavras_de_frase):
                    return None
                
                # Cria título
                if len(palavras) <= 8:
                    base_title = clean_text.upper()
                else:
                    power_starts = ["eu", "você", "ele", "ela", "nós", "vou", "vai", 
                                   "nunca", "sempre", "preciso", "quero", "isso", "não"]
                    best_start = 0
                    for i, w in enumerate(palavras[:5]):
                        if w.lower() in power_starts:
                            best_start = i
                            break
                    
                    end_idx = min(best_start + 7, len(palavras))
                    base_title = " ".join(palavras[best_start:end_idx]).upper()
                
                base_title = base_title.rstrip(".,;:!?\"'") + "!"
                
                # Verifica se não é repetido
                if base_title not in used_titles:
                    return base_title
                
                return None
            
            # ========== 1. TENTA DO TEXTO DIRETO ==========
            titulo_do_texto = tentar_criar_do_texto(text)
            if titulo_do_texto:
                used_titles.add(titulo_do_texto)
                logger.info(f"[TITULO v15.8] ✓ Da transcrição: '{titulo_do_texto}'")
                return titulo_do_texto
            
            # ========== 2. BUSCA TEXTO NO INTERVALO (SEM RECURSÃO) ==========
            if all_segs:
                for seg in all_segs:
                    seg_start = seg.get("start", 0)
                    seg_end = seg.get("end", 0)
                    
                    if seg_start >= start_time - 20 and seg_end <= start_time + 90:
                        seg_text = seg.get("text", "").strip()
                        titulo_do_seg = tentar_criar_do_texto(seg_text)
                        if titulo_do_seg:
                            used_titles.add(titulo_do_seg)
                            logger.info(f"[TITULO v15.8] ✓ Do segmento próximo: '{titulo_do_seg}'")
                            return titulo_do_seg
            
            # ========== 3. USA BIBLIOTECA ==========
            tipo_detectado = detectar_tipo_cena(text)
            titulo_criativo = escolher_da_biblioteca(tipo_detectado, index)
            used_titles.add(titulo_criativo)
            
            logger.info(f"[TITULO v15.8] ✓ Da biblioteca ({tipo_detectado}): '{titulo_criativo}'")
            return titulo_criativo
        
        # Processa cada momento importante
        for idx, moment in enumerate(filtered_moments):
            # Calcula tempos do corte
            moment_duration = moment["end"] - moment["start"]
            
            # Expande para atingir duração mínima
            if moment_duration < min_duration:
                expand = (min_duration - moment_duration) / 2
                start = max(0, moment["start"] - expand - 5)  # 5s de contexto antes
                end = min(video_duration, moment["end"] + expand + 5)  # 5s de contexto depois
            else:
                start = max(0, moment["start"] - 3)
                end = min(video_duration, moment["end"] + 3)
            
            # Ajusta para não exceder máximo
            if end - start > max_duration:
                center = (start + end) / 2
                start = center - max_duration / 2
                end = center + max_duration / 2
            
            # Garante que está dentro dos limites
            start = max(0, start)
            end = min(video_duration, end)
            cut_duration_actual = end - start
            
            # Verifica duração válida
            if cut_duration_actual < min_duration * 0.8:
                continue
            
            # Verifica se não é duplicado
            if is_range_used(start, end):
                continue
            
            # v15.6: Gera título ÚNICO usando nova função
            title = generate_unique_title_v158(
                moment.get("text", ""), 
                moment["type"], 
                len(cuts),
                anime_name,
                start,
                all_segments
            )
            
            # Adiciona corte
            cuts.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "title": title,
                "score": moment["score"],
                "type": moment["type"],
                "duration": round(cut_duration_actual, 2),
                "original_text": moment.get("text", "")[:100]
            })
            
            used_ranges.append((start, end))
            
            logger.info(f"  [CORTE {len(cuts)}] {start:.1f}s-{end:.1f}s ({cut_duration_actual:.1f}s) - \"{title[:30]}...\"")
        
        # ==================== ETAPA 10: FALLBACK SE POUCOS CORTES ====================
        if len(cuts) < 3:
            logger.info("[FALLBACK] Poucos cortes encontrados, adicionando cortes adicionais...")
            
            # Divide o vídeo em segmentos e adiciona cortes de partes não usadas
            segment_duration = max_duration
            num_possible_segments = int(video_duration / segment_duration)
            
            for i in range(num_possible_segments):
                if len(cuts) >= 10:  # Limite de segurança
                    break
                
                seg_start = i * segment_duration
                seg_end = min((i + 1) * segment_duration, video_duration)
                
                if not is_range_used(seg_start, seg_end):
                    # Busca diálogo neste segmento para gerar título
                    segment_text = ""
                    for seg in all_segments:
                        if seg_start <= seg["start"] < seg_end:
                            segment_text = seg["text"]
                            break
                    
                    # v15.6: Usa nova função de título único
                    title = generate_unique_title_v158(
                        segment_text, 
                        "segment", 
                        len(cuts),
                        anime_name,
                        seg_start,
                        all_segments
                    )
                    
                    cuts.append({
                        "start": round(seg_start, 2),
                        "end": round(seg_end, 2),
                        "title": title,
                        "score": 50,
                        "type": "segment",
                        "duration": round(seg_end - seg_start, 2)
                    })
                    
                    used_ranges.append((seg_start, seg_end))
                    logger.info(f"  [CORTE ADICIONAL {len(cuts)}] {seg_start:.1f}s-{seg_end:.1f}s - \"{title[:30]}...\"")
        
        # ==================== RESULTADO FINAL ====================
        logger.info("=" * 60)
        logger.info(f"[RESULTADO] {len(cuts)} CORTES ÚNICOS GERADOS")
        for i, cut in enumerate(cuts):
            logger.info(f"  [{i+1}] {cut['start']:.0f}s-{cut['end']:.0f}s: {cut['title'][:40]}")
        logger.info("=" * 60)
        
        return cuts
        
    except Exception as e:
        logger.error(f"[ERROR] Erro na análise: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return generate_smart_fallback_cuts(video_path, anime_name, min_duration, max_duration)
    
    finally:
        # Limpeza de arquivos temporários
        for audio_file in audio_files_to_clean:
            try:
                if audio_file.exists():
                    audio_file.unlink()
            except:
                pass
        gc.collect()


def generate_smart_fallback_cuts(video_path: str, anime_name: str, min_duration: int = 30, max_duration: int = 90) -> List[Dict]:
    """Gera cortes inteligentes quando IA não está disponível"""
    
    try:
        # Obtém duração
        if MOVIEPY_AVAILABLE:
            video = moviepy_imports['VideoFileClip'](video_path)
            duration = video.duration
            video.close()
        else:
            cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                   '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            duration = float(result.stdout.strip()) if result.returncode == 0 else 1200
        
        logger.info(f"[FALLBACK] Duração do vídeo: {duration:.1f}s")
        
        # Calcula número de cortes possíveis
        target_duration = (min_duration + max_duration) / 2
        num_cuts = max(3, int(duration / target_duration))
        
        cuts = []
        titles = [
            "MOMENTO INCRÍVEL!",
            "CENA ÉPICA!",
            "CONFRONTO!",
            "REVELAÇÃO!",
            "DESPERTAR!",
            "PODER MÁXIMO!",
            "HORA DA VERDADE!",
            "BATALHA DECISIVA!",
            "O INÍCIO!",
            "O CLÍMAX!"
        ]
        
        for i in range(num_cuts):
            start = i * target_duration
            end = min((i + 1) * target_duration, duration)
            
            if end - start >= min_duration * 0.8:
                cuts.append({
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "title": titles[i % len(titles)],
                    "score": 60,
                    "type": "fallback",
                    "duration": round(end - start, 2)
                })
        
        logger.info(f"[FALLBACK] {len(cuts)} cortes gerados")
        return cuts
        
    except Exception as e:
        logger.error(f"[FALLBACK ERROR] {e}")
        return [{
            "start": 0,
            "end": 60,
            "title": "MOMENTO ÉPICO!",
            "score": 50,
            "type": "emergency",
            "duration": 60
        }]
def generate_fallback_cuts(video_path: str, anime_name: str) -> List[Dict]:
    """Wrapper para manter compatibilidade - usa a nova função"""
    return generate_smart_fallback_cuts(video_path, anime_name, 30, 90)


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
    """Processa um corte individual OTIMIZADO PARA GPU com cleanup robusto
    
    v15.1: Títulos CHAMATIVOS e MAGNÉTICOS + Diagnóstico completo
    """
    
    # Lista para rastrear clips a serem fechados
    clips_to_close = []
    
    try:
        start = cut_data.get('start', 0)
        end = cut_data.get('end', start + 60)
        
        # ========== LOG DE DIAGNÓSTICO DO TITLE STYLE ==========
        title_style = config.get('titleStyle', {})
        logger.info("=" * 60)
        logger.info(f"[CORTE {num}] DIAGNÓSTICO DE ESTILO:")
        logger.info(f"  titleStyle recebido: {title_style}")
        logger.info(f"  fontSize: {title_style.get('fontSize')} (tipo: {type(title_style.get('fontSize')).__name__})")
        logger.info(f"  textColor: {title_style.get('textColor')}")
        logger.info(f"  strokeWidth: {title_style.get('strokeWidth')}")
        logger.info(f"  verticalPosition: {title_style.get('verticalPosition')}")
        logger.info("=" * 60)
        
        # v15.1: Título CHAMATIVO e MAGNÉTICO
        title = cut_data.get('title')
        
        # Títulos de emergência CHAMATIVOS (não genéricos)
        emergency_titles = [
            "A BATALHA QUE MUDOU TUDO!",
            "PODER ALÉM DO LIMITE!",
            "O MOMENTO DECISIVO!",
            "VOCÊ NÃO ESTÁ PREPARADO!",
            "EXPLOSÃO DE PODER!",
            "ISSO VAI TE EMOCIONAR!",
            "A HORA DA VERDADE!",
            "NUNCA DESISTA!",
            "O HERÓI DESPERTA!",
            "ARREPIOS GARANTIDOS!"
        ]
        
        # Se não houver título personalizado, gera um CHAMATIVO
        if not title or title == config.get('animeName', 'Anime') or len(title) < 5:
            original_text = cut_data.get('original_text', '')
            
            if original_text and len(original_text) > 10:
                # Processa o texto para criar título chamativo
                text = original_text.strip()
                words = text.split()
                
                # Palavras de impacto
                power_words = ["poder", "força", "nunca", "sempre", "vou", "matar", 
                              "proteger", "salvar", "destruir", "vencer", "impossível"]
                
                text_lower = text.lower()
                has_power = any(word in text_lower for word in power_words)
                
                if has_power and len(words) >= 3:
                    # Usa o texto com formatação impactante
                    title_words = words[:6]
                    title = " ".join(title_words).upper()
                    title = title.rstrip(".,;:!?") + "!"
                    
                    # Se ficou muito curto, adiciona contexto
                    if len(title) < 15:
                        title = f"QUANDO ELE DISSE: {title}"
                        
                    logger.info(f"[TITULO] Gerado do texto: '{title}'")
                else:
                    # Usa título de emergência chamativo
                    title = emergency_titles[num % len(emergency_titles)]
                    logger.info(f"[TITULO] Usando título chamativo: '{title}'")
            else:
                # Título de emergência
                title = emergency_titles[num % len(emergency_titles)]
                logger.info(f"[TITULO] Usando título de emergência: '{title}'")
        
        logger.info(f"[CUT {num}] Título: '{title}' ({start:.1f}s - {end:.1f}s)")
        
        # Valida tempos
        if start < 0 or end <= start:
            raise ValueError(f"Tempos inválidos: start={start}, end={end}")
        
        # Valida que o vídeo esteja em um path permitido
        if not is_safe_path(VOLUME_PATH, video_path) and not is_safe_path(TEMP_DIR, video_path) and not is_safe_path(CACHE_DIR, video_path):
            raise ValueError(f"Caminho de vídeo não permitido: {video_path}")

        # Carrega vídeo
        video = moviepy_imports['VideoFileClip'](video_path)
        clips_to_close.append(video)

        # Valida tempos contra duração do vídeo
        if end > video.duration:
            logger.warning(f"[WARNING] Ajustando fim de {end}s para {video.duration}s")
            end = video.duration

        if start >= video.duration:
            raise ValueError(f"Start {start}s excede duração do vídeo {video.duration}s")

        # Corta segmento
        clip = video.subclip(start, end)
        clips_to_close.append(clip)

        # Aplica anti-shadowban com opções v12.8
        antishadowban_config = config.get("antiShadowban", {})
        if isinstance(antishadowban_config, dict):
            if antishadowban_config.get("enabled", True):
                clip = apply_antishadowban(clip, antishadowban_config)
                clips_to_close.append(clip)
        elif antishadowban_config:  # Retrocompatibilidade: boolean True
            clip = apply_antishadowban(clip, None)
            clips_to_close.append(clip)
        
        # Configurações TikTok
        target_w, target_h = 1080, 1920
        
        # Background
        bg_path = config.get("background_path")
        bg_clip = None

        if bg_path and os.path.exists(bg_path) and PIL_AVAILABLE:
            try:
                # Valida background
                if not is_safe_path(VOLUME_PATH, bg_path) and not is_safe_path(TEMP_DIR, bg_path) and not is_safe_path(CACHE_DIR, bg_path):
                    raise ValueError(f"Caminho de background não permitido: {bg_path}")

                bg_img = Image.open(bg_path).convert('RGB')
                bg_img = bg_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                bg_clip = moviepy_imports['ImageClip'](np.array(bg_img)).set_duration(clip.duration)
                clips_to_close.append(bg_clip)
                logger.debug("[RENDER] Background carregado")
            except Exception as e:
                logger.warning(f"[WARNING] Background falhou: {e}")
                bg_clip = None

        if bg_clip is None:
            # Background sólido escuro
            bg_color = (15, 15, 30)
            bg_clip = moviepy_imports['ColorClip'](size=(target_w, target_h), color=bg_color)
            bg_clip = bg_clip.set_duration(clip.duration)
            clips_to_close.append(bg_clip)
            logger.debug("[RENDER] Background sólido")
        
        # ==================== ENQUADRAMENTO v12.8 ====================
        # frameMode: "letterbox" (moldura) ou "fill" (preencher com crop)
        frame_mode = config.get("frameMode", "letterbox")
        
        w, h = clip.w, clip.h
        clip_aspect = w / h
        target_aspect = target_w / target_h
        
        if frame_mode == "letterbox":
            # CORTE MOLDURA: Mantém vídeo 16:9 centralizado sobre fundo 9:16
            # Não faz crop - preserva todo o conteúdo
            
            # Calcula tamanho para caber na largura (1080px)
            video_width = target_w
            video_height = int(video_width / clip_aspect)
            
            # Se ficou maior que a altura disponível, ajusta pela altura
            if video_height > target_h:
                video_height = target_h
                video_width = int(video_height * clip_aspect)
            
            # Redimensiona mantendo proporção original
            clip_resized = clip.resize(width=video_width, height=video_height)
            clips_to_close.append(clip_resized)
            
            # Centraliza verticalmente e horizontalmente
            x_pos = (target_w - video_width) // 2
            y_pos = (target_h - video_height) // 2
            clip_pos = clip_resized.set_position((x_pos, y_pos))
            clips_to_close.append(clip_pos)
            
            logger.info(f"[RENDER] Corte moldura: {video_width}x{video_height} centralizado em {target_w}x{target_h}")
        
        else:
            # MODO FILL: Crop para preencher (comportamento anterior)
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
            
            clips_to_close.append(clip_cropped)
            
            clip_resized = clip_cropped.resize(width=target_w)
            clips_to_close.append(clip_resized)
            
            clip_pos = clip_resized.set_position(('center', 'center'))
            clips_to_close.append(clip_pos)
            
            logger.info(f"[RENDER] Modo fill: crop para {target_w}x{target_h}")
        
        # Camadas
        layers = [bg_clip, clip_pos]

        # Título
        title_clip = None
        if config.get("generateTitles", True) and title and PIL_AVAILABLE:
            logger.info(f"[TITULO] Gerando título: '{title[:40]}...'")
            title_style = config.get("titleStyle", {})
            safe_title_text = sanitize_input(str(title).upper(), max_len=80)
            
            # Posição vertical: converte de % para fração (15 -> 0.15)
            vertical_pos = title_style.get("verticalPosition", 15)
            if isinstance(vertical_pos, (int, float)) and vertical_pos > 1:
                vertical_pos = vertical_pos / 100.0  # Converte de % para fração
            
            title_clip = criar_titulo_simples(
                texto=safe_title_text,
                largura_video=target_w,
                altura_video=target_h,
                duracao=clip.duration,
                font_size=title_style.get("fontSize", 70),
                text_color=title_style.get("textColor", "#FFD700"),
                stroke_color=title_style.get("strokeColor", "#000000"),
                stroke_width=title_style.get("strokeWidth", 6),
                pos_vertical=vertical_pos,
                font_family=title_style.get("fontFamily")
            )

            if title_clip:
                layers.append(title_clip)
                clips_to_close.append(title_clip)
                logger.info("[TITULO] Adicionado à composição")
            else:
                logger.warning("[TITULO] Falha ao criar título - verifique fontes instaladas")
        else:
            if not config.get("generateTitles", True):
                logger.info("[TITULO] Geração de títulos desativada")
            elif not title:
                logger.info("[TITULO] Nenhum título fornecido")
            elif not PIL_AVAILABLE:
                logger.warning("[TITULO] PIL não disponível")

        # Composição final
        final = moviepy_imports['CompositeVideoClip'](layers, size=(target_w, target_h))
        clips_to_close.append(final)

        # Nome do arquivo de saída
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_'))[:30]
        output_filename = f"cut_{num}_{safe_title}_{uuid.uuid4().hex[:6]}.mp4"
        output_path = OUTPUT_DIR / output_filename
        
        # ==================== ENCODING v15.9.5 - SIMPLIFICADO ====================
        # Problema: Métodos complexos falham silenciosamente
        # Solução: Um único método robusto com NVENC
        
        logger.info("=" * 60)
        logger.info("[ENCODING v15.9.5] NVENC SIMPLIFICADO")
        logger.info("=" * 60)
        
        start_encode = time.time()
        cut_duration = end - start
        logger.info(f"[CORTE] Duração: {cut_duration:.1f}s")
        
        encoding_success = False
        
        # Verifica NVENC
        nvenc_available = False
        try:
            result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], 
                                   capture_output=True, text=True, timeout=10)
            nvenc_available = 'h264_nvenc' in result.stdout
            logger.info(f"[GPU] NVENC: {'✓ DISPONÍVEL' if nvenc_available else '✗ NÃO'}")
        except:
            pass
        
        # Dimensões
        w, h = clip.w, clip.h
        clip_aspect = w / h
        
        if clip_aspect > (target_w / target_h):
            vid_w = target_w
            vid_h = int(target_w / clip_aspect)
        else:
            vid_h = target_h
            vid_w = int(target_h * clip_aspect)
        
        vid_w = vid_w - (vid_w % 2)
        vid_h = vid_h - (vid_h % 2)
        x_offset = (target_w - vid_w) // 2
        y_offset = (target_h - vid_h) // 2
        
        logger.info(f"[LAYOUT] Vídeo: {vid_w}x{vid_h} em ({x_offset},{y_offset})")
        
        # ==================== CRIA PNGs UMA VEZ ====================
        bg_png_path = None
        title_png_path = None
        
        # 1. Background PNG
        if bg_clip:
            try:
                bg_png_path = TEMP_DIR / f"bg_{num}_{uuid.uuid4().hex[:6]}.png"
                bg_frame = bg_clip.get_frame(0)
                if bg_frame.dtype != np.uint8:
                    bg_frame = np.clip(bg_frame, 0, 255).astype(np.uint8)
                Image.fromarray(bg_frame).save(str(bg_png_path), 'PNG')
                logger.info(f"[BG] ✓ PNG criado: {bg_png_path}")
            except Exception as e:
                logger.warning(f"[BG] ✗ Falha ao criar PNG: {e}")
                bg_png_path = None
        else:
            logger.warning("[BG] ✗ bg_clip não existe")
        
        # 2. Título PNG com transparência
        if title and config.get("generateTitles", True):
            try:
                title_png_path = TEMP_DIR / f"title_{num}_{uuid.uuid4().hex[:6]}.png"
                title_style = config.get("titleStyle", {})
                pos_vertical = title_style.get("verticalPosition", 25) / 100.0
                safe_title = sanitize_input(str(title).upper(), max_len=80)
                
                result = criar_titulo_png_transparente(
                    texto=safe_title,
                    largura_video=target_w,
                    altura_video=target_h,
                    pos_vertical=pos_vertical,
                    title_style=title_style,
                    output_path=title_png_path
                )
                
                if result and title_png_path.exists():
                    logger.info(f"[TITULO] ✓ PNG criado: {title_png_path}")
                else:
                    logger.warning("[TITULO] ✗ Falha ao criar PNG")
                    title_png_path = None
            except Exception as e:
                logger.warning(f"[TITULO] ✗ Erro: {e}")
                title_png_path = None
        
        # Log do que temos
        logger.info(f"[RECURSOS] BG: {'✓' if bg_png_path else '✗'} | Título: {'✓' if title_png_path else '✗'}")
        
        # ==================== MÉTODO ÚNICO: NVENC COM OVERLAY ====================
        if nvenc_available:
            try:
                logger.info("[ENCODING] NVENC com overlay completo")
                
                # Monta filter_complex e inputs baseado no que temos
                inputs = ['-ss', str(start), '-t', str(cut_duration), '-i', video_path]
                
                if bg_png_path and bg_png_path.exists() and title_png_path and title_png_path.exists():
                    # Tem TUDO: background + vídeo + título
                    inputs.extend(['-loop', '1', '-t', str(cut_duration), '-i', str(bg_png_path)])
                    inputs.extend(['-loop', '1', '-t', str(cut_duration), '-i', str(title_png_path)])
                    
                    filter_complex = (
                        f"[1:v]scale={target_w}:{target_h}[bg];"
                        f"[0:v]scale={vid_w}:{vid_h}[vid];"
                        f"[bg][vid]overlay={x_offset}:{y_offset}[withvid];"
                        f"[withvid][2:v]overlay=0:0[final]"
                    )
                    map_stream = '[final]'
                    logger.info("[FILTER] BG + Vídeo + Título")
                    
                elif bg_png_path and bg_png_path.exists():
                    # Só background + vídeo (sem título)
                    inputs.extend(['-loop', '1', '-t', str(cut_duration), '-i', str(bg_png_path)])
                    
                    filter_complex = (
                        f"[1:v]scale={target_w}:{target_h}[bg];"
                        f"[0:v]scale={vid_w}:{vid_h}[vid];"
                        f"[bg][vid]overlay={x_offset}:{y_offset}[final]"
                    )
                    map_stream = '[final]'
                    logger.info("[FILTER] BG + Vídeo (sem título)")
                    
                elif title_png_path and title_png_path.exists():
                    # Só título (fundo preto)
                    inputs.extend(['-loop', '1', '-t', str(cut_duration), '-i', str(title_png_path)])
                    
                    filter_complex = (
                        f"color=c=black:s={target_w}x{target_h}:d={cut_duration}[bg];"
                        f"[0:v]scale={vid_w}:{vid_h}[vid];"
                        f"[bg][vid]overlay={x_offset}:{y_offset}[withvid];"
                        f"[withvid][1:v]overlay=0:0[final]"
                    )
                    map_stream = '[final]'
                    logger.info("[FILTER] Fundo preto + Vídeo + Título")
                    
                else:
                    # Nada - só vídeo centralizado
                    filter_complex = (
                        f"color=c=black:s={target_w}x{target_h}:d={cut_duration}[bg];"
                        f"[0:v]scale={vid_w}:{vid_h}[vid];"
                        f"[bg][vid]overlay={x_offset}:{y_offset}[final]"
                    )
                    map_stream = '[final]'
                    logger.info("[FILTER] Fundo preto + Vídeo (sem nada)")
                
                # Encoder NVENC
                encoder = ['-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '23', '-b:v', '6M']
                
                cmd = ['ffmpeg', '-y'] + inputs + [
                    '-filter_complex', filter_complex,
                    '-map', map_stream,
                    '-map', '0:a?',
                ] + encoder + [
                    '-c:a', 'aac', '-b:a', '128k',
                    '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart',
                    str(output_path)
                ]
                
                logger.info(f"[NVENC] Executando... (timeout: 180s)")
                
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                
                if proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 50000:
                    encoding_success = True
                    encode_time = time.time() - start_encode
                    file_size = output_path.stat().st_size / 1e6
                    speed = cut_duration / encode_time if encode_time > 0 else 0
                    
                    logger.info("=" * 60)
                    logger.info(f"[✓ SUCCESS] Corte {num} - NVENC!")
                    logger.info(f"    Arquivo: {file_size:.1f} MB | Tempo: {encode_time:.1f}s")
                    logger.info(f"    Velocidade: {speed:.2f}x realtime")
                    logger.info("=" * 60)
                else:
                    err_msg = proc.stderr[-500:] if proc.stderr else 'sem detalhes'
                    logger.warning(f"[NVENC] Falhou: {err_msg}")
                    
            except subprocess.TimeoutExpired:
                logger.warning("[NVENC] Timeout!")
            except Exception as e:
                logger.warning(f"[NVENC] Erro: {str(e)[:200]}")
        
        # ==================== FALLBACK: CPU ====================
        if not encoding_success:
            logger.warning("[ENCODING] Fallback CPU")
            
            try:
                # Reconstroi com libx264
                inputs = ['-ss', str(start), '-t', str(cut_duration), '-i', video_path]
                
                if bg_png_path and bg_png_path.exists() and title_png_path and title_png_path.exists():
                    inputs.extend(['-loop', '1', '-t', str(cut_duration), '-i', str(bg_png_path)])
                    inputs.extend(['-loop', '1', '-t', str(cut_duration), '-i', str(title_png_path)])
                    filter_complex = (
                        f"[1:v]scale={target_w}:{target_h}[bg];"
                        f"[0:v]scale={vid_w}:{vid_h}[vid];"
                        f"[bg][vid]overlay={x_offset}:{y_offset}[withvid];"
                        f"[withvid][2:v]overlay=0:0[final]"
                    )
                    map_stream = '[final]'
                elif bg_png_path and bg_png_path.exists():
                    inputs.extend(['-loop', '1', '-t', str(cut_duration), '-i', str(bg_png_path)])
                    filter_complex = (
                        f"[1:v]scale={target_w}:{target_h}[bg];"
                        f"[0:v]scale={vid_w}:{vid_h}[vid];"
                        f"[bg][vid]overlay={x_offset}:{y_offset}[final]"
                    )
                    map_stream = '[final]'
                else:
                    filter_complex = (
                        f"color=c=black:s={target_w}x{target_h}:d={cut_duration}[bg];"
                        f"[0:v]scale={vid_w}:{vid_h}[vid];"
                        f"[bg][vid]overlay={x_offset}:{y_offset}[final]"
                    )
                    map_stream = '[final]'
                
                cmd = ['ffmpeg', '-y'] + inputs + [
                    '-filter_complex', filter_complex,
                    '-map', map_stream,
                    '-map', '0:a?',
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                    '-c:a', 'aac', '-b:a', '128k',
                    '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart',
                    str(output_path)
                ]
                
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                
                if proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 10000:
                    encoding_success = True
                    encode_time = time.time() - start_encode
                    
                    logger.info("=" * 60)
                    logger.info(f"[✓ SUCCESS] Corte {num} - CPU")
                    logger.info(f"    Tempo: {encode_time:.1f}s")
                    logger.info("=" * 60)
                    
            except Exception as e:
                logger.error(f"[CPU] Falhou: {str(e)[:100]}")
        
        # ==================== ÚLTIMO RECURSO: CORTE SIMPLES ====================
        if not encoding_success:
            logger.error("[ENCODING] Último recurso: corte simples")
            
            try:
                cmd = [
                    'ffmpeg', '-y',
                    '-ss', str(start),
                    '-t', str(cut_duration),
                    '-i', video_path,
                    '-vf', f'scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black',
                    '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '26',
                    '-c:a', 'aac', '-b:a', '128k',
                    '-pix_fmt', 'yuv420p',
                    str(output_path)
                ]
                
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if proc.returncode == 0 and output_path.exists():
                    encoding_success = True
                    logger.info(f"[✓] Corte {num} - Simples (sem BG/título)")
                    
            except Exception as e:
                logger.error(f"[SIMPLES] Falhou: {e}")
        
        # Limpa PNGs temporários
        for f in [bg_png_path, title_png_path]:
            if f and f.exists():
                try: f.unlink()
                except: pass
        
        if not encoding_success:
            raise Exception("Encoding falhou com todos os codecs")

        return str(output_path)
        
    except Exception as e:
        logger.error(f"[ERROR GPU] Erro no corte {num}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
        
    finally:
        # CORREÇÃO v12.4: Cleanup robusto de todos os clips
        for clip_obj in reversed(clips_to_close):
            try:
                if clip_obj is not None and hasattr(clip_obj, 'close'):
                    clip_obj.close()
            except Exception as e:
                logger.debug(f"[CLEANUP] Erro ao fechar clip: {e}")
        
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
    logger.info(f"ANIMECUT v12.7.3 - NOVA REQUISIÇÃO [ID: {request_id}]")
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

        # ==================== DIAGNÓSTICO v15.1 - LOG COMPLETO DO INPUT ====================
        logger.info("=" * 70)
        logger.info("🔍 [DIAGNÓSTICO v15.1] DADOS RECEBIDOS DO WEBAPP")
        logger.info("=" * 70)
        
        # Log de TODOS os campos recebidos
        logger.info("[RAW INPUT] Campos presentes no input_data:")
        for key, value in input_data.items():
            if key == "video_url":
                logger.info(f"  {key}: {str(value)[:50]}...")
            elif key == "background_url":
                logger.info(f"  {key}: {value}")  # Log COMPLETO da URL
            elif key == "titleStyle":
                logger.info(f"  {key}: {value}")  # Log COMPLETO do titleStyle
            else:
                logger.info(f"  {key}: {value}")
        
        # Verifica especificamente o titleStyle
        logger.info("-" * 70)
        logger.info("[TITLE STYLE] Análise detalhada:")
        ts_raw = input_data.get("titleStyle")
        logger.info(f"  Valor bruto: {ts_raw}")
        logger.info(f"  Tipo: {type(ts_raw)}")
        
        if ts_raw:
            if isinstance(ts_raw, dict):
                logger.info(f"  fontSize bruto: {ts_raw.get('fontSize')} (tipo: {type(ts_raw.get('fontSize')).__name__})")
                logger.info(f"  textColor: {ts_raw.get('textColor')}")
                logger.info(f"  strokeColor: {ts_raw.get('strokeColor')}")
                logger.info(f"  strokeWidth: {ts_raw.get('strokeWidth')} (tipo: {type(ts_raw.get('strokeWidth')).__name__})")
                logger.info(f"  verticalPosition: {ts_raw.get('verticalPosition')}")
                logger.info(f"  fontFamily: {ts_raw.get('fontFamily')}")
            else:
                logger.warning(f"  ⚠️ titleStyle NÃO é um dict! É {type(ts_raw)}")
        else:
            logger.warning("  ⚠️ titleStyle está VAZIO ou None!")
        
        # Verifica background_url
        logger.info("-" * 70)
        logger.info("[BACKGROUND URL] Análise detalhada:")
        bg_url_raw = input_data.get("background_url")
        logger.info(f"  Valor bruto: {bg_url_raw}")
        logger.info(f"  Tipo: {type(bg_url_raw)}")
        logger.info(f"  É string?: {isinstance(bg_url_raw, str)}")
        logger.info(f"  Está vazio?: {not bg_url_raw or bg_url_raw == ''}")
        if bg_url_raw:
            logger.info(f"  Começa com https?: {str(bg_url_raw).startswith('https://')}")
            logger.info(f"  Contém backblazeb2?: {'backblazeb2' in str(bg_url_raw).lower()}")
            logger.info(f"  Comprimento: {len(str(bg_url_raw))} caracteres")
        logger.info("=" * 70)

        # 1. Download com retry
        logger.info("[STEP 1/4] Download de vídeo...")
        video_path = download_video(video_url)

        # Background (opcional) - não escapar HTML em URLs
        bg_url = input_data.get("background_url")
        bg_url = sanitize_input(bg_url, escape_html=False) if bg_url else None
        
        logger.info("=" * 70)
        logger.info("[BACKGROUND DOWNLOAD] Iniciando processo de download")
        logger.info(f"  URL após sanitize: {bg_url}")
        logger.info("=" * 70)
        
        if bg_url:
            logger.info(f"[BACKGROUND] URL recebida: {bg_url}")
            bg_path = download_background(bg_url)
            if bg_path:
                logger.info(f"[BACKGROUND] ✓ Baixado com sucesso: {bg_path}")
                # Verifica se o arquivo existe e tem tamanho
                try:
                    bg_file = Path(bg_path)  # Path já importado no início do arquivo
                    if bg_file.exists():
                        logger.info(f"[BACKGROUND] ✓ Arquivo existe, tamanho: {bg_file.stat().st_size} bytes")
                    else:
                        logger.warning(f"[BACKGROUND] ⚠️ Arquivo NÃO existe no path: {bg_path}")
                except Exception as e:
                    logger.warning(f"[BACKGROUND] Erro ao verificar arquivo: {e}")
            else:
                logger.warning("[BACKGROUND] ✗ Falha ao baixar background - download_background retornou None")
        else:
            bg_path = None
            logger.warning("[BACKGROUND] ⚠️ Nenhuma URL de background fornecida (bg_url é None ou vazio)")
        
        # Configuração v12.8 - COMPLETA
        # Anti-shadowban pode ser boolean (retrocompatível) ou objeto detalhado
        antishadowban_input = input_data.get("antiShadowban", True)
        if isinstance(antishadowban_input, dict):
            antishadowban_config = {
                "enabled": antishadowban_input.get("enabled", True),
                "mirror": antishadowban_input.get("mirror", True),
                "colorGrading": antishadowban_input.get("colorGrading", True),
                "microZoom": antishadowban_input.get("microZoom", False),
                "filmGrain": antishadowban_input.get("filmGrain", False)
            }
        else:
            # Retrocompatibilidade: boolean simples
            antishadowban_config = {
                "enabled": antishadowban_input == True,
                "mirror": True,
                "colorGrading": True,
                "microZoom": False,
                "filmGrain": False
            }
        
        # TitleStyle com novos parâmetros
        default_title_style = {
            "fontSize": 70,
            "textColor": "#FFD700",
            "strokeColor": "#000000",
            "strokeWidth": 6,
            "verticalPosition": 15,  # % do topo
            "fontFamily": None  # Usa fonte padrão
        }
        title_style_input = input_data.get("titleStyle", {})
        title_style = {**default_title_style, **title_style_input}
        
        # Configuração de duração dos cortes
        cut_duration_input = input_data.get("cutDuration", {})
        cut_duration = {
            "min": cut_duration_input.get("min", 15),
            "max": cut_duration_input.get("max", 60)
        }
        
        # Configuração de áudio
        audio_input = input_data.get("audio", {})
        audio_config = {
            "isolateVoice": audio_input.get("isolateVoice", False)
        }
        
        config = {
            "animeName": anime_name,
            "generateTitles": input_data.get("generateTitles", True),
            "titleStyle": title_style,
            "background_path": bg_path,
            
            # v14.1 - Parâmetros completos
            "frameMode": input_data.get("frameMode", "letterbox"),
            "scenePreference": input_data.get("scenePreference", "balanced"),
            "cutDuration": cut_duration,
            "antiShadowban": antishadowban_config,
            "audio": audio_config,
            
            # v14.1: Número de cortes (0 = sem limite, IA decide)
            "numberOfCuts": input_data.get("numberOfCuts", 0),
            
            # Smart Crop
            "smartCrop": input_data.get("smartCrop", {
                "enabled": False,
                "mode": "fixed",
                "zoom": 1.0
            }),
            
            # Resolução de saída
            "outputResolution": input_data.get("outputResolution", "1080p")
        }
        
        # Log detalhado da configuração
        logger.info("=" * 60)
        logger.info("[CONFIG v14.1] PARÂMETROS RECEBIDOS:")
        logger.info(f"  animeName: {anime_name}")
        logger.info(f"  generateTitles: {config['generateTitles']}")
        logger.info(f"  frameMode: {config['frameMode']}")
        logger.info(f"  scenePreference: {config['scenePreference']}")
        logger.info(f"  cutDuration: {cut_duration}")
        logger.info(f"  numberOfCuts: {config['numberOfCuts']} (0 = sem limite)")
        logger.info(f"  background_path: {bg_path}")
        logger.info(f"  titleStyle: fontSize={title_style.get('fontSize')}, color={title_style.get('textColor')}, font={title_style.get('fontFamily')}")
        logger.info(f"  antiShadowban: {antishadowban_config}")
        logger.info("=" * 60)
        
        # 2. Análise e definição de cortes
        logger.info("[STEP 2/4] Análise de conteúdo...")
        
        cuts = []
        cut_type = input_data.get("cutType", "auto")
        
        if cut_type == "auto" and AI_AVAILABLE:
            logger.info(f"[MODE] Automático com IA (foco: {config['scenePreference']})")
            cuts = analyze_video_content_gpu(
                video_path, 
                anime_name,
                scene_preference=config['scenePreference'],
                cut_duration=config['cutDuration']
            )
        elif cut_type == "manual":
            manual_cuts = input_data.get("cuts", [])
            if manual_cuts:
                cuts = manual_cuts
                logger.info(f"[MODE] Manual: {len(cuts)} cortes")
            else:
                logger.info("[MODE] Manual sem cortes, usando automático")
                cuts = analyze_video_content_gpu(
                    video_path, 
                    anime_name,
                    scene_preference=config['scenePreference'],
                    cut_duration=config['cutDuration']
                )
        else:
            logger.info("[MODE] Fallback")
            cuts = generate_fallback_cuts(video_path, anime_name)
        
        # Validação de cortes (SEM LIMITE - v14.0)
        cuts = [c for c in cuts if c.get("start", 0) >= 0 and c.get("end", 0) > c.get("start", 0)]
        # v14.0: REMOVIDO LIMITE DE 3 CORTES - Gera quantos cortes a IA identificar
        
        if not cuts:
            logger.warning("[WARNING] Nenhum corte válido, usando fallback de emergência")
            cuts = generate_fallback_cuts(video_path, anime_name)
        
        logger.info(f"[CUTS v14] {len(cuts)} cortes para processar (SEM LIMITE)")
        
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
        
        # Log detalhado dos resultados para debug
        logger.info("=" * 50)
        logger.info("[RESULTADO DETALHADO]")
        for r in results:
            logger.info(f"  Cut {r['id']}: {r['title'][:40]}")
            logger.info(f"    URL: {r['url'][:80] if r['url'] else 'SEM URL'}...")
            logger.info(f"    Duração: {r['duration']:.1f}s")
        logger.info("=" * 50)
        
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
                "timestamp": datetime.now().isoformat(),
                "version": "13.0",
                "frame_mode": config.get("frameMode", "letterbox"),
                "scene_preference": config.get("scenePreference", "balanced")
            },
            # URLs diretas para facilitar acesso pelo webapp
            "download_urls": [r["url"] for r in results if r.get("url")]
        }
        
        logger.info("=" * 70)
        logger.info(f"REQUISIÇÃO FINALIZADA [ID: {request_id}]")
        logger.info(f"URLs de download: {len(response['download_urls'])}")
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
        # Banner com versão detalhada
        print("\n" + "="*70)
        print("╔═══════════════════════════════════════════════════════════════════╗")
        print("║   ANIMECUT SERVERLESS v15.9.7 - BUILD 2025-12-22 21:00           ║")
        print("║   🚀 FFMPEG COM NVENC - GPU ENCODING REAL                        ║")
        print("╚═══════════════════════════════════════════════════════════════════╝")
        print("Correções v15.9.7:")
        print("  ✓ FFMPEG: Compilado do source com NVENC")
        print("  ✓ NVENC: h264_nvenc habilitado")
        print("  ✓ GPU: Encoding 2-3x mais rápido")
        print("  ✓ PATH: Erro corrigido")
        print(f"Volume: {VOLUME_BASE}")
        print(f"Cache: {CACHE_DIR}")
        print(f"B2 Bucket: {B2_BUCKET if B2_BUCKET else 'NÃO CONFIGURADO'}")
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
