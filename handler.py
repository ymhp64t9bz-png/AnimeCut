#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AnimeCut Serverless - Handler Ultra Simples
Versão mínima para teste
"""

import runpod
import sys
import os

print("=" * 50)
print("🚀 AnimeCut Handler Iniciando...")
print("=" * 50)

def handler(event):
    """Handler ultra simples para teste"""
    print(f"📦 Event recebido: {event}")
    
    try:
        # Retorna sucesso sempre
        result = {
            "status": "success",
            "message": "AnimeCut worker is ALIVE!",
            "python_version": sys.version,
            "event_received": event
        }
        
        print(f"✅ Retornando: {result}")
        return result
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    print("🎬 Starting RunPod Worker...")
    runpod.serverless.start({"handler": handler})
    print("✅ Worker started!")
