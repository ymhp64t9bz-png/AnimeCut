# 🔍 GUIA DE DEBUG v13.0 - AnimeCut

## ❌ PROBLEMAS RELATADOS E SOLUÇÕES

### PROBLEMA 1: Vídeo redimensionado ao invés de moldura
**Causa:** O parâmetro `frameMode` não está sendo enviado ou está como `"fill"`
**Solução:** Garantir que o webapp envia `"frameMode": "letterbox"`

### PROBLEMA 2: Títulos genéricos (não personalizados)
**Causa v12.x:** O sistema usava apenas `anime_name + número`
**Solução v13.0:** Nova função `generate_smart_title()` analisa a transcrição e gera títulos baseados no conteúdo

### PROBLEMA 3: Cortes duplicados (mesma cena)
**Causa v12.x:** O algoritmo não verificava sobreposição
**Solução v13.0:** Divide o vídeo em 3 terços e garante 1 corte de cada parte, com verificação de sobreposição

### PROBLEMA 4: Parâmetros não aplicados
**Causa:** O webapp pode estar enviando JSON mal formatado
**Solução:** Ver seção DEBUG abaixo

### PROBLEMA 5: Background não baixado
**Causa:** URL inválida ou bloqueio de rede
**Solução:** Verificar logs para erro específico

### PROBLEMA 6: Vídeos não disponíveis para download
**Causa:** Resposta da API não está sendo parseada corretamente pelo webapp
**Solução:** Ver estrutura de resposta abaixo

---

## 📋 ESTRUTURA DA REQUISIÇÃO (O QUE O WEBAPP DEVE ENVIAR)

```json
{
  "input": {
    "video_url": "https://...",
    "animeName": "Nome do Anime - Episódio X",
    "cutType": "auto",
    "frameMode": "letterbox",
    "scenePreference": "balanced",
    "generateTitles": true,
    "background_url": "https://...",
    "antiShadowban": {
      "enabled": true,
      "mirror": true,
      "colorGrading": true
    },
    "titleStyle": {
      "fontSize": 70,
      "textColor": "#FFD700",
      "strokeColor": "#000000",
      "strokeWidth": 6,
      "verticalPosition": 15
    },
    "cutDuration": {
      "min": 40,
      "max": 90
    }
  }
}
```

---

## 📤 ESTRUTURA DA RESPOSTA (O QUE O WEBAPP VAI RECEBER)

```json
{
  "status": "success",
  "request_id": "abc123",
  "cuts": [
    {
      "id": 1,
      "path": "/app/output/cut_1_xxx.mp4",
      "url": "https://s3.us-east-005.backblazeb2.com/...",
      "title": "TITULO GERADO PELA IA",
      "start": 60.0,
      "end": 105.0,
      "duration": 45.0,
      "score": 85,
      "type": "action",
      "processing_time": 12.5,
      "file_size_mb": 8.2,
      "gpu_encoded": true
    },
    {
      "id": 2,
      "url": "https://...",
      "title": "OUTRO TITULO GERADO",
      ...
    },
    {
      "id": 3,
      "url": "https://...",
      "title": "TERCEIRO TITULO",
      ...
    }
  ],
  "metadata": {
    "anime_name": "Nome do Anime",
    "total_cuts_requested": 3,
    "successful_cuts": 3,
    "processing_time": 45.2,
    "gpu_used": true,
    "timestamp": "2025-12-17T04:00:00Z"
  }
}
```

---

## 🔧 DEBUG NO WEBAPP (Agent 3)

### Antes de enviar a requisição:
```javascript
console.log("========== DEBUG ANIMECUT v13 ==========");
console.log("REQUISIÇÃO COMPLETA:", JSON.stringify(requestBody, null, 2));
console.log("video_url presente?", !!requestBody.input?.video_url);
console.log("animeName:", requestBody.input?.animeName);
console.log("frameMode:", requestBody.input?.frameMode);
console.log("scenePreference:", requestBody.input?.scenePreference);
console.log("background_url:", requestBody.input?.background_url);
console.log("antiShadowban:", JSON.stringify(requestBody.input?.antiShadowban));
console.log("titleStyle:", JSON.stringify(requestBody.input?.titleStyle));
console.log("=========================================");
```

### Ao receber a resposta:
```javascript
console.log("========== RESPOSTA ANIMECUT v13 ==========");
console.log("status:", response.status);
console.log("cuts:", response.cuts?.length);

if (response.cuts) {
  response.cuts.forEach((cut, i) => {
    console.log(`CUT ${i+1}:`);
    console.log("  - title:", cut.title);
    console.log("  - url:", cut.url);
    console.log("  - duration:", cut.duration);
  });
}

console.log("metadata:", JSON.stringify(response.metadata, null, 2));
console.log("============================================");
```

---

## 📊 COMO O WEBAPP DEVE EXIBIR OS DOWNLOADS

```javascript
// Após receber a resposta
if (response.status === "success" && response.cuts) {
  // Para cada corte, criar botão de download
  response.cuts.forEach((cut, index) => {
    const downloadUrl = cut.url;  // URL presigned do B2
    const title = cut.title;
    const duration = cut.duration;
    
    // Criar elemento de download
    const downloadBtn = document.createElement('a');
    downloadBtn.href = downloadUrl;
    downloadBtn.download = `${title.replace(/[^a-zA-Z0-9]/g, '_')}.mp4`;
    downloadBtn.textContent = `Baixar Corte ${index + 1}: ${title}`;
    downloadBtn.className = 'download-button';
    
    // Adicionar ao container de downloads
    downloadsContainer.appendChild(downloadBtn);
  });
}
```

---

## ⚠️ TIPOS DE DADOS IMPORTANTES

| Campo | Tipo Correto | Tipo Errado |
|-------|-------------|-------------|
| `generateTitles` | `true` | `"true"` |
| `antiShadowban.enabled` | `true` | `"true"` |
| `fontSize` | `70` | `"70"` |
| `verticalPosition` | `15` | `"15"` |
| `cutDuration.min` | `40` | `"40"` |

---

## 🚨 CHECKLIST PARA TESTAR

### No Webapp:
- [ ] Todos os campos do formulário são coletados?
- [ ] Os valores são convertidos para o tipo correto (Number, Boolean)?
- [ ] Os parâmetros estão dentro de `"input": {}`?
- [ ] A URL do vídeo está correta?
- [ ] A URL do background está correta?
- [ ] O console mostra a requisição completa antes de enviar?

### Na Resposta:
- [ ] `response.status === "success"`?
- [ ] `response.cuts` é um array?
- [ ] Cada `cut.url` é uma URL válida?
- [ ] Os títulos (`cut.title`) são diferentes para cada corte?

---

## 📝 LOGS DO SISTEMA (O QUE PROCURAR)

### Sucesso esperado:
```
[ANALYSIS v13] Preferência: balanced, Duração: 40-90s
[ANALYSIS v13] Anime: Nome do Anime
[TRANSCRIPTION] 45 segmentos de diálogo
[ACTION ANALYSIS] 12 cenas de ação detectadas
[CUT 1] 60.0s-105.0s (45.0s) - 'TITULO INTELIGENTE...'
[CUT 2] 320.0s-365.0s (45.0s) - 'OUTRO TITULO...'
[CUT 3] 580.0s-625.0s (45.0s) - 'TERCEIRO TITULO...'
[ANALYSIS v13] 3 cortes DISTINTOS gerados
[RENDER] Corte moldura: 1080x607 centralizado em 1080x1920
```

### Erros a procurar:
```
[BACKGROUND] Falha ao baixar background
[ERROR] Whisper GPU não pode ser carregado
[WARNING] URL inválida
[ERROR] Erro na análise v13
```

---

## 🔄 FLUXO COMPLETO

1. **Webapp coleta dados do formulário**
2. **Webapp monta JSON com estrutura correta**
3. **Webapp envia para RunPod API**
4. **AnimeCut recebe e loga:** `[CONFIG] {...}`
5. **AnimeCut baixa vídeo e background**
6. **AnimeCut analisa (Whisper + Ação)**
7. **AnimeCut gera 3 cortes DISTINTOS com títulos IA**
8. **AnimeCut renderiza com MOLDURA**
9. **AnimeCut faz upload para B2**
10. **AnimeCut retorna resposta com URLs**
11. **Webapp parseia resposta**
12. **Webapp exibe botões de download**
