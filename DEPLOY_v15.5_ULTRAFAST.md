# 🚀 DEPLOY v15.5 - ULTRARRÁPIDO + TÍTULOS + FONTES

## RESUMO DAS CORREÇÕES

| Problema | Causa | Solução v15.5 |
|----------|-------|---------------|
| **Lentidão (6min/corte)** | MoviePy renderiza frame a frame lento | FFmpeg pipe + NVENC (~60s/corte) |
| **Títulos genéricos** | Cenas de ação não tinham texto | Busca transcrição no intervalo do corte |
| **Fontes não encontradas** | Diretório /workspace/fonts vazio | 16 fontes incluídas no Docker |
| **Background 401** | Bucket B2 não público | Download via S3 API |

---

## 📊 PERFORMANCE ESPERADA

| Métrica | v15.4 (atual) | v15.5 (novo) |
|---------|---------------|--------------|
| Tempo/corte | 5-6 minutos | **~60 segundos** |
| 9 cortes | 45-54 minutos | **~10 minutos** |
| Custo/job | ~$0.50 | **~$0.10** |
| Títulos | Genéricos | **Da transcrição** |
| Fontes | 0 customizadas | **16 fontes** |

---

## 📁 ESTRUTURA DO DEPLOY

```
deploy/
├── handler.py          # v15.5 (3514 linhas)
├── Dockerfile          # Atualizado com fontes
└── fonts/              # 16 fontes customizadas
    ├── AmoreChristmas.otf
    ├── BarberChop.otf
    ├── Basketball.otf
    ├── BearDays.otf
    ├── Blustrue.otf
    ├── ChocolateAdventure.ttf
    ├── Heinan.otf
    ├── HeinanOutline.otf
    ├── HeroesLegend.ttf
    ├── HeroesLegendHollow.ttf
    ├── Karina.ttf
    ├── Kotton.otf
    ├── MilkDays.otf
    ├── PersonaAura.otf
    ├── SuperCrawler.ttf
    └── SuperSquadItalic.otf
```

---

## 🔧 MUDANÇAS TÉCNICAS

### 1. ENCODING ULTRARRÁPIDO (FFmpeg Pipe)

**Antes (v15.4):**
```python
# MoviePy renderiza frame a frame = LENTO
final.write_videofile(output, codec='libx264', preset='ultrafast')
# Tempo: 5-6 minutos para 100s de vídeo
```

**Depois (v15.5):**
```python
# FFmpeg recebe frames via pipe + NVENC = RÁPIDO
ffmpeg_proc = subprocess.Popen(['ffmpeg', '-f', 'rawvideo', ...])
for frame in final.iter_frames():
    ffmpeg_proc.stdin.write(frame.tobytes())
# Tempo: ~60 segundos para 100s de vídeo
```

### 2. TÍTULOS DA TRANSCRIÇÃO

**Antes:**
```python
# Cenas de ação usavam texto genérico
"text": "[CENA DE AÇÃO INTENSA]"  # Sempre fallback
```

**Depois:**
```python
# Busca melhor frase da transcrição no intervalo
action_text = get_transcription_for_range(start, end, all_segments)
# Retorna: "EU VOU PROTEGER MEUS AMIGOS!" (texto real)
```

### 3. FONTES CUSTOMIZADAS

**Antes:**
```
[TITULO] Fontes disponíveis: []
[TITULO] ✗ Fonte 'Heinan' NÃO encontrada
```

**Depois:**
```
[TITULO] Fontes disponíveis: ['Heinan.otf', 'HeroesLegend.ttf', ...]
[TITULO] ✓ Fonte encontrada: /workspace/fonts/Heinan.otf
```

---

## 📋 PASSOS PARA DEPLOY

### 1. Estrutura de Arquivos
Certifique-se de ter:
```
├── Dockerfile
├── handler.py
└── fonts/
    ├── (16 arquivos .otf e .ttf)
```

### 2. Build no RunPod
```bash
# O CACHEBUST força rebuild completo
docker build -t animecut:v15.5 .
```

### 3. Verificar Build
O log deve mostrar:
```
=== BUILD COMPLETO v15.5 ===
Handler: 15.5_20251218_0800_ULTRAFAST_FONTS
Novidades: FFmpeg pipe NVENC, títulos da transcrição, 16 fontes
Fontes disponíveis:
-rw-r--r-- 1 root root 75480 Dec 18 ... Heinan.otf
-rw-r--r-- 1 root root 26012 Dec 18 ... HeroesLegend.ttf
...
```

---

## 🧪 VERIFICAÇÃO PÓS-DEPLOY

### Logs Corretos (v15.5):
```
╔═══════════════════════════════════════════════════════════════════╗
║   ANIMECUT SERVERLESS v15.5 - BUILD 2025-12-18 08:00             ║
║   🚀 ULTRARRÁPIDO + TÍTULOS DA TRANSCRIÇÃO + FONTES              ║
╚═══════════════════════════════════════════════════════════════════╝

[TITULO v15.2] Gerado do TEXTO: 'EU VOU PROTEGER!' (original: 'Eu vou proteger...')
[TITULO] ✓ Fonte encontrada: /workspace/fonts/Heinan.otf

[ENCODING v15.5] ULTRARRÁPIDO - FFMPEG PIPE
[NVENC] Disponível: ✓ SIM
[ENCODING] Método 1: FFmpeg via pipe...
[ENCODING] Usando NVENC (p4)
[✓ SUCCESS] Corte 1 finalizado!
    Arquivo: 15.2 MB
    Tempo: 58.3s
    Velocidade: 1.71x realtime
    Método: FFmpeg Pipe + NVENC
```

### Logs Errados (versão antiga):
```
[TITULO v15.2] ⚠ Usando fallback (sem texto): 'ESSA CENA VAI TE SURPREENDER!'
[TITULO] ✗ Fonte 'Heinan' NÃO encontrada

[ENCODING v15.4] OTIMIZADO E CONFIÁVEL
    Tempo: 367.6s
    Velocidade: 0.27x realtime
```

---

## ⚠️ PARA O WEBAPP (Agent 3)

### Problema: Prévia sem Background
O webapp mostra tela preta na prévia ao invés do background.

### Solução Sugerida:
```javascript
// No componente de prévia do webapp
const PreviewCanvas = ({ backgroundUrl, titleStyle, titleText }) => {
  const canvasRef = useRef(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // 1. Carrega background
    if (backgroundUrl) {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        drawTitle(ctx, titleText, titleStyle);
      };
      img.onerror = () => {
        console.error('Erro ao carregar background');
        // Fallback: cor sólida
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        drawTitle(ctx, titleText, titleStyle);
      };
      img.src = backgroundUrl;
    }
  }, [backgroundUrl, titleStyle, titleText]);
  
  const drawTitle = (ctx, text, style) => {
    const y = (style.verticalPosition / 100) * canvas.height;
    ctx.fillStyle = style.textColor;
    ctx.strokeStyle = style.strokeColor;
    ctx.lineWidth = style.strokeWidth;
    ctx.font = `bold ${style.fontSize}px ${style.fontFamily}`;
    ctx.textAlign = 'center';
    ctx.strokeText(text, canvas.width / 2, y);
    ctx.fillText(text, canvas.width / 2, y);
  };
  
  return <canvas ref={canvasRef} width={1080} height={1920} />;
};
```

---

## 📊 LISTA DE FONTES DISPONÍVEIS

| Nome do Arquivo | Para usar no webapp |
|-----------------|---------------------|
| AmoreChristmas.otf | `fontFamily: 'AmoreChristmas'` |
| BarberChop.otf | `fontFamily: 'BarberChop'` |
| Basketball.otf | `fontFamily: 'Basketball'` |
| BearDays.otf | `fontFamily: 'BearDays'` |
| Blustrue.otf | `fontFamily: 'Blustrue'` |
| ChocolateAdventure.ttf | `fontFamily: 'ChocolateAdventure'` |
| Heinan.otf | `fontFamily: 'Heinan'` |
| HeinanOutline.otf | `fontFamily: 'HeinanOutline'` |
| HeroesLegend.ttf | `fontFamily: 'HeroesLegend'` |
| HeroesLegendHollow.ttf | `fontFamily: 'HeroesLegendHollow'` |
| Karina.ttf | `fontFamily: 'Karina'` |
| Kotton.otf | `fontFamily: 'Kotton'` |
| MilkDays.otf | `fontFamily: 'MilkDays'` |
| PersonaAura.otf | `fontFamily: 'PersonaAura'` |
| SuperCrawler.ttf | `fontFamily: 'SuperCrawler'` |
| SuperSquadItalic.otf | `fontFamily: 'SuperSquadItalic'` |

---

## ✅ CHECKLIST DE DEPLOY

- [ ] Copiar `handler.py` para pasta do Docker
- [ ] Copiar `Dockerfile` para pasta do Docker
- [ ] Criar pasta `fonts/` e copiar as 16 fontes
- [ ] Executar `docker build`
- [ ] Verificar que o build mostra as fontes
- [ ] Deploy no RunPod
- [ ] Testar com um job e verificar logs
- [ ] Confirmar tempo < 2 minutos por corte
- [ ] Confirmar títulos vêm da transcrição
- [ ] Confirmar fonte customizada foi usada
