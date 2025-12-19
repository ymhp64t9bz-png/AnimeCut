# 🚀 DEPLOY v15.6 - NVENC FORÇADO + TÍTULOS ÚNICOS 

## RESUMO DOS 3 PROBLEMAS RESOLVIDOS

| # | Problema | Causa Raiz | Solução v15.6 |
|---|----------|------------|---------------|
| 1 | **Preview sem background** | Webapp não carrega imagem | `TitlePreview.jsx` componente React |
| 2 | **Títulos repetidos** | Função não evitava duplicação | `generate_unique_title_v156()` com tracking |
| 3 | **Fallback CPU** | NVENC não era prioridade | NVENC forçado com 3 tentativas |

---

## 📊 PERFORMANCE ESPERADA v15.6

| Métrica | Antes | v15.6 |
|---------|-------|-------|
| Encoding | CPU (5-6 min) | **GPU NVENC (~60s)** |
| Títulos | Repetidos genéricos | **100% únicos** |
| 9 cortes | 45-54 min | **~10 min** |

---

## 🔧 MUDANÇAS TÉCNICAS

### 1. TÍTULOS ÚNICOS (generate_unique_title_v156)

```python
# ANTES (v15.5):
def extract_title_from_text(text, moment_type, index):
    # Podia repetir títulos
    titles = fallback_by_type.get(moment_type, ...)
    return titles[index % len(titles)]  # REPETE!

# DEPOIS (v15.6):
used_titles = set()  # Tracking global

def generate_unique_title_v156(text, moment_type, index, anime_name, start_time, all_segs):
    # 1. Tenta usar transcrição
    # 2. Busca texto no intervalo de tempo
    # 3. Gera título criativo com anime_name e index
    # 4. NUNCA repete - verifica em used_titles
    
    if final_title in used_titles:
        # Adiciona variação
        final_title = f"{base_title} #{index+1}!"
    
    used_titles.add(final_title)
    return final_title
```

### 2. NVENC FORÇADO (3 tentativas)

```python
# MÉTODO 1: NVENC otimizado para RTX 4090
final.write_videofile(
    codec='h264_nvenc',
    ffmpeg_params=[
        '-preset', 'p4',      # Balanceado
        '-tune', 'hq',        # Alta qualidade
        '-rc', 'vbr',         # Variable bitrate
        '-cq', '23',          # Qualidade constante
        '-b:v', '6M',         # Bitrate alvo
        '-gpu', '0'           # Força GPU 0
    ]
)

# MÉTODO 2: NVENC conservador (se método 1 falhar)
final.write_videofile(
    codec='h264_nvenc',
    ffmpeg_params=['-preset', 'p2', '-b:v', '5M']
)

# MÉTODO 3: CPU (ÚLTIMO RECURSO APENAS)
final.write_videofile(codec='libx264', preset='ultrafast')
```

---

## 📁 ARQUIVOS PARA DEPLOY

### Backend (RunPod)
```
├── handler.py          # v15.6 (3563 linhas)
├── Dockerfile          # v15.6 NVENC + TÍTULOS
└── fontes/             # 16 fontes (JÁ EXISTE no repo)
```

### Frontend (Webapp)
```
└── TitlePreview.jsx    # Componente React para preview
```

---

## 🎨 COMPONENTE REACT - TitlePreview.jsx

### O que faz:
1. Carrega a imagem do background (URL do B2)
2. Renderiza no canvas
3. Desenha o título com fonte, cor, borda
4. Mostra loading e erro

### Como usar:

```jsx
import TitlePreview from './TitlePreview';

<TitlePreview 
  backgroundUrl="https://f005.backblazeb2.com/file/KortexClipAI2/..."
  titleText="VOCÊ PRECISA VER ISSO!"
  titleStyle={{
    fontSize: 80,
    textColor: '#FFFFFF',
    strokeColor: '#000000',
    strokeWidth: 2,
    verticalPosition: 25,
    fontFamily: 'Heinan'
  }}
/>
```

---

## 🧪 VERIFICAÇÃO PÓS-DEPLOY

### Logs Corretos v15.6:
```
╔═══════════════════════════════════════════════════════════════════╗
║   ANIMECUT SERVERLESS v15.6 - BUILD 2025-12-18 10:00            ║
║   🚀 NVENC FORÇADO + TÍTULOS ÚNICOS + FONTES                    ║
╚═══════════════════════════════════════════════════════════════════╝

[GPU] NVENC: ✓ DISPONÍVEL
[GPU] Encoders: ['h264_nvenc', 'hevc_nvenc']
[GPU] Status: NVIDIA GeForce RTX 4090, 23000 MiB

[TITULO v15.6] ✓ Gerado da TRANSCRIÇÃO: 'EU VOU PROTEGER MEUS AMIGOS!'
[TITULO v15.6] ✓ Gerado da TRANSCRIÇÃO: 'NUNCA VOU DESISTIR!'

[ENCODING v15.6] NVENC FORÇADO - RTX 4090
[ENCODING] Método 1: NVENC h264_nvenc (GPU)
[✓ SUCCESS] Corte 1 - NVENC GPU!
    Arquivo: 18.2 MB
    Tempo: 45.3s
    Velocidade: 2.21x realtime
    Codec: h264_nvenc (RTX 4090)
```

### Se ainda usar CPU (PROBLEMA):
```
[ENCODING] Método 3: CPU libx264 (fallback)
⚠ ATENÇÃO: Usando CPU, verifique NVENC
```

---

## ✅ CHECKLIST DE DEPLOY

### Backend
- [ ] Baixar `handler.py` v15.6
- [ ] Baixar `Dockerfile` v15.6
- [ ] Verificar pasta `fontes/` existe no repo
- [ ] Fazer commit no GitHub
- [ ] Build no RunPod
- [ ] Verificar banner mostra v15.6
- [ ] Verificar NVENC disponível nos logs
- [ ] Testar com um job
- [ ] Confirmar títulos são únicos
- [ ] Confirmar encoding usa NVENC

### Frontend (Agent 3)
- [ ] Adicionar `TitlePreview.jsx` ao projeto
- [ ] Importar no componente de edição
- [ ] Passar backgroundUrl, titleText, titleStyle
- [ ] Testar preview mostra background
- [ ] Testar ajuste de verticalPosition funciona

---

## 🔍 TROUBLESHOOTING

### NVENC não funciona
```bash
# Verificar no container:
ffmpeg -encoders | grep nvenc
nvidia-smi
```

### Títulos ainda repetidos
- Verificar se v15.6 está rodando (banner)
- Verificar logs mostram `[TITULO v15.6]`

### Preview ainda preto
- Verificar backgroundUrl não está vazio
- Verificar CORS está habilitado no B2
- Abrir URL no navegador para testar
