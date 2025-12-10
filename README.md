# 💎 AnimeCut Serverless v6.0

Sistema completo de processamento de vídeo com IA para RunPod Serverless.

## 🚀 Funcionalidades

- ✅ **AI Scene Detection** (Qwen 2.5)
- ✅ **Viral Title Generation**
- ✅ **Audio Transcription** (Whisper)
- ✅ **GPU Rendering** (NVENC) + CPU fallback
- ✅ **Backblaze B2 Integration**
- ✅ **Anti-Shadowban Features**

## 📦 Deploy no RunPod

### **1. Build da Imagem**

```bash
docker build -t animecut-serverless:v6 .
docker tag animecut-serverless:v6 seu-usuario/animecut-serverless:v6
docker push seu-usuario/animecut-serverless:v6
```

### **2. Configurar Endpoint no RunPod**

1. Acesse RunPod Console
2. Crie novo Serverless Endpoint
3. Configure:
   - **Container Image:** `seu-usuario/animecut-serverless:v6`
   - **Container Disk:** 20 GB
   - **GPU:** RTX 4090 ou A100 (recomendado)

### **3. Variáveis de Ambiente**

```bash
B2_KEY_ID=68702c2cbfc6
B2_APP_KEY=00506496bc1450b6722b672d9a43d00605f17eadd7
B2_ENDPOINT=https://s3.us-east-005.backblazeb2.com
B2_BUCKET_NAME=autocortes-storage
```

### **4. Volume para Modelos de IA (IMPORTANTE)**

⚠️ **Os modelos de IA são grandes (5-10 GB) e devem ser armazenados em um Volume persistente do RunPod.**

#### **Criar Volume:**

1. No RunPod Console, vá em **Storage** → **Network Volumes**
2. Crie um novo volume: `animecut-models` (20 GB)
3. Monte o volume em: `/app/models`

#### **Baixar Modelos no Volume:**

Execute uma vez para popular o volume:

```python
# Qwen 2.5 (para títulos virais)
# Baixar de: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF
# Salvar em: /app/models/qwen2.5-7b-instruct-q4_k_m.gguf

# Whisper Medium (para transcrição)
# Será baixado automaticamente na primeira execução
```

#### **Configurar Volume no Endpoint:**

No RunPod Endpoint, adicione:
- **Volume Name:** `animecut-models`
- **Mount Path:** `/app/models`

Isso garante que os modelos sejam carregados rapidamente sem precisar baixar a cada execução.

## 📊 Performance

| Etapa | Tempo Médio (GPU) |
|-------|-------------------|
| Scene Detection | 30-60s |
| Transcription | 10-20s |
| Title Generation | 5-10s |
| Rendering (NVENC) | 20-40s |
| Upload B2 | 10-30s |
| **Total** | **75-160s** |

## 🔧 Dependências Principais

- **llama-cpp-python** (com GGML_CUDA para GPU)
- **openai-whisper**
- **moviepy**
- **opencv-python-headless**
- **boto3** (Backblaze B2)

## 📝 Notas

- O Dockerfile usa `GGML_CUDA` (não `LLAMA_CUBLAS` que está deprecated)
- Requer GPU com CUDA 11.8+
- Modelos de IA devem estar em volume persistente
- Build time: ~15-20 minutos
- Imagem final: ~8-10 GB

## 🆘 Suporte

Para problemas ou dúvidas, consulte a documentação completa no repositório.

---

**Versão:** 6.0  
**Status:** ✅ Production Ready  
**Última atualização:** 10/12/2024
