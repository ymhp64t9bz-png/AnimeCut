#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
✂️ AnimeCut Serverless v10.0 FINAL (MoviePy V2 + Qwen/Whisper AI RESTORED)
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

# PATHS
sys.path.append("/runpod-volume/site-packages") 
sys.path.append("/runpod-volume")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AnimeCutPro")

TEMP_DIR = Path("/tmp/animecut")
OUTPUT_DIR = Path("/runpod-volume/output")
TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VOLUME_PATH = Path("/runpod-volume")
MODELS_PATH = VOLUME_PATH / "models"
QWEN_MODEL_PATH = MODELS_PATH / "Qwen2.5-7B-Instruct"
FONT_PATH = VOLUME_PATH / "fonts" / "impact.ttf"
FONT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ==================== IMPORTS IA ====================
try:
    import torch
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
    from ultralytics import YOLO
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    logger.warning("⚠️ IA libs ausentes.")

# ==================== MOVIEPY V2 FIX ====================
try:
    from moviepy import *
    from moviepy.video.fx import MirrorX, GammaCorr
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logger.error("❌ MoviePy não encontrado.")

# ==================== B2 BACKBLAZE ====================
try:
    import boto3
    from botocore.client import Config
    B2_KEY_ID = os.environ.get("B2_KEY_ID")
    B2_APP_KEY = os.environ.get("B2_APPLICATION_KEY")
    B2_ENDPOINT = os.environ.get("B2_ENDPOINT", "https://s3.us-east-005.backblazeb2.com")
    B2_BUCKET = os.environ.get("B2_BUCKET_NAME", "KortexAI")
    
    if B2_KEY_ID and B2_APP_KEY:
        s3_client = boto3.client("s3", endpoint_url=B2_ENDPOINT,
            aws_access_key_id=B2_KEY_ID, aws_secret_access_key=B2_APP_KEY,
            config=Config(signature_version="s3v4"))
        B2_AVAILABLE = True
    else:
        B2_AVAILABLE = False
except: B2_AVAILABLE = False

# ==================== LOADERS IA ====================
whisper_pipe = None
qwen_model = None
qwen_tokenizer = None

def load_models():
    global whisper_pipe, qwen_model, qwen_tokenizer
    if not AI_AVAILABLE: return

    # Whisper Turbo
    if whisper_pipe is None:
        logger.info("🚀 Carregando Whisper...")
        try:
            whisper_pipe = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3",
                torch_dtype=torch.float16,
                device="cuda:0",
                model_kwargs={"attn_implementation": "flash_attention_2"}
            )
        except:
            # Fallback se Flash Attn falhar
            whisper_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device="cuda:0")

    # Qwen
    if qwen_model is None:
        logger.info("🧠 Carregando Qwen...")
        try:
            model_p = str(QWEN_MODEL_PATH) if QWEN_MODEL_PATH.exists() else "Qwen/Qwen2.5-7B-Instruct"
            qwen_tokenizer = AutoTokenizer.from_pretrained(model_p, trust_remote_code=True)
            qwen_model = AutoModelForCausalLM.from_pretrained(model_p, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
        except Exception as e:
            logger.error(f"Erro Qwen: {e}")

# ==================== ANALISE DE CONTEÚDO ====================
def analyze_video(video_path, anime_name):
    load_models()
    if not whisper_pipe or not qwen_model: return []
    
    logger.info("🎤 Transcrevendo...")
    # Extrai audio
    audio_path = str(TEMP_DIR / "temp.wav")
    subprocess.run(['ffmpeg', '-i', video_path, '-vn', '-ar', '16000', '-ac', '1', audio_path, '-y', '-loglevel', 'error'])
    
    # Transcreve
    result = whisper_pipe(audio_path, chunk_length_s=30, batch_size=24, return_timestamps=True)
    text_log = "\n".join([f"[{c['timestamp'][0]}-{c['timestamp'][1]}] {c['text']}" for c in result['chunks']])
    
    logger.info("🧠 Gerando Cortes com LLM...")
    prompt = f"""
    Anime: {anime_name}
    Transcript:
    {text_log[:15000]}
    
    Identifique 2 cenas virais (40s a 60s). Retorne JSON puro:
    [ {{"start": 10.0, "end": 50.0, "title": "TITULO IMPACTANTE"}} ]
    """
    
    inputs = qwen_tokenizer([prompt], return_tensors="pt").to("cuda")
    out = qwen_model.generate(**inputs, max_new_tokens=200)
    resp = qwen_tokenizer.decode(out[0], skip_special_tokens=True)
    
    # Parser simples de JSON (pode precisar de regex em prod)
    try:
        start_idx = resp.find("[")
        end_idx = resp.rfind("]") + 1
        return json.loads(resp[start_idx:end_idx])
    except:
        logger.error("Erro parse JSON Qwen")
        return []

# ==================== EDIÇÃO & RENDER ====================
def processar_corte(video_path, cut, num, config):
    try:
        video = VideoFileClip(video_path)
        clip = video.subclipped(cut['start'], cut['end']) # SINTAXE V2 CORRETA
        
        # Anti-Shadowban
        if config.get("antiShadowban"):
            if random.choice([True, False]): clip = clip.with_effects([MirrorX()])
            
        # Formato 9:16
        w, h = 1080, 1920
        clip_resized = clip.resized(width=w) # SINTAXE V2 CORRETA
        clip_final = clip_resized.with_position("center") # SINTAXE V2 CORRETA
        
        final = CompositeVideoClip([
            ColorClip(size=(w, h), color=(10,10,10)).with_duration(clip.duration),
            clip_final
        ], size=(w, h))
        
        out_path = OUTPUT_DIR / f"cut_{num}.mp4"
        
        # NVENC
        final.write_videofile(
            str(out_path), 
            codec="h264_nvenc" if torch.cuda.is_available() else "libx264",
            preset="fast",
            ffmpeg_params=["-cq", "23"] if torch.cuda.is_available() else [],
            audio_codec="aac",
            threads=4, logger=None
        )
        final.close()
        video.close()
        return str(out_path)
        
    except Exception as e:
        logger.error(f"Erro Render: {e}")
        raise

# ==================== HANDLER ====================
def handler(event):
    gc.collect()
    torch.cuda.empty_cache()
    inp = event.get("input", {})
    
    try:
        url = inp.get("video_url")
        if not url: return {"error": "No URL provided"}
        
        # Download
        r = requests.get(url, stream=True)
        vid_path = str(TEMP_DIR / "input.mp4")
        with open(vid_path, 'wb') as f:
            for c in r.iter_content(8192): f.write(c)
            
        # IA ou Manual
        if inp.get("cutType") == "auto":
            cuts = analyze_video(vid_path, inp.get("animeName", "Anime"))
            if not cuts: cuts = [{"start": 0, "end": 30, "title": "FALLBACK IA FALHOU"}]
        else:
            cuts = [{"start": 30, "end": 60, "title": "TESTE MANUAL"}]
            
        res = []
        for i, cut in enumerate(cuts):
            out = processar_corte(vid_path, cut, i, inp)
            # Upload B2
            if B2_AVAILABLE:
                key = f"animecut/{os.path.basename(out)}"
                s3_client.upload_file(out, B2_BUCKET, key)
                link = s3_client.generate_presigned_url('get_object', Params={'Bucket': B2_BUCKET, 'Key': key})
                res.append({"url": link, "title": cut.get("title")})
            else:
                res.append({"path": out, "error": "B2 unavailable"})
                
        return {"status": "success", "data": res}
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
