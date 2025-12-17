# 📦 ANIMECUT v13.0 - CHANGELOG

## 🚀 DATA: 2025-12-17 04:00

---

## 🎯 PROBLEMAS RESOLVIDOS

### ❌ ANTES (v12.x)
1. Vídeo redimensionado em vez de moldura
2. Títulos genéricos ("Anime - AÇÃO 1")
3. Cortes duplicados (mesma cena 3x)
4. Parâmetros do webapp ignorados
5. Background não baixado
6. Downloads não disponíveis no webapp

### ✅ AGORA (v13.0)
1. **CORTE MOLDURA REAL**: Vídeo 16:9 centralizado sobre 9:16
2. **TÍTULOS INTELIGENTES**: Gerados pela IA baseados na transcrição
3. **CORTES DISTINTOS**: Garantia de partes diferentes do vídeo
4. **PARÂMETROS APLICADOS**: Todos os campos do webapp processados
5. **LOGS DETALHADOS**: Debug completo para identificar problemas
6. **RESPOSTA ESTRUTURADA**: URLs de download claramente organizadas

---

## ✨ NOVAS FUNCIONALIDADES

### 1. GERAÇÃO DE TÍTULOS COM IA

O sistema agora analisa a transcrição do Whisper para gerar títulos:

```python
# Função generate_smart_title() analisa:
# - Diálogos próximos à cena
# - Palavras-chave de impacto (poder, força, luta, proteger...)
# - Contexto da ação
```

**Exemplo de resultado:**
- ANTES: "Gachiakuta - AÇÃO 1"
- AGORA: "EU VOU PROTEGER VOCÊ!"

### 2. CORTES DE PARTES DIFERENTES

O vídeo é dividido em 3 terços e o sistema garante 1 corte de cada parte:

```
Terço 1: 0% - 33% do vídeo    → Corte 1
Terço 2: 33% - 66% do vídeo   → Corte 2
Terço 3: 66% - 100% do vídeo  → Corte 3
```

Também verifica sobreposição para garantir que não há cortes duplicados.

### 3. RESPOSTA MELHORADA

A resposta agora inclui:
- `download_urls`: Array com todas as URLs prontas
- `metadata.version`: "13.0"
- `metadata.frame_mode`: Confirma qual modo foi usado
- `metadata.scene_preference`: Confirma preferência de cena

---

## 📁 ARQUIVOS ATUALIZADOS

- **handler.py**: ~3000 linhas, v13.0
- **Dockerfile**: v13.0, CACHEBUST novo
- **DEBUG_GUIDE_v13.md**: Guia de debug completo

---

## 🔧 DEPLOY

```bash
git add handler.py Dockerfile
git commit -m "v13.0 - Títulos IA + Cortes Distintos + Moldura Real"
git push
```

---

## 📋 ESTRUTURA DA REQUISIÇÃO

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

## 📤 ESTRUTURA DA RESPOSTA

```json
{
  "status": "success",
  "cuts": [
    {
      "id": 1,
      "url": "https://s3.../...",
      "title": "TITULO GERADO PELA IA",
      "start": 60.0,
      "end": 105.0,
      "duration": 45.0
    },
    ...
  ],
  "download_urls": [
    "https://...",
    "https://...",
    "https://..."
  ],
  "metadata": {
    "version": "13.0",
    "frame_mode": "letterbox",
    "scene_preference": "balanced",
    ...
  }
}
```

---

## ✅ VERIFICAÇÃO NO LOG

Após deploy, procure por:

```
╔═══════════════════════════════════════════════════════════════════╗
║   ANIMECUT SERVERLESS v13.0 - BUILD 2025-12-17 04:00             ║
║   TÍTULOS IA + CORTES DISTINTOS + MOLDURA                        ║
╚═══════════════════════════════════════════════════════════════════╝

[ANALYSIS v13] 3 cortes DISTINTOS gerados
  [1] 60.0s-105.0s: TITULO INTELIGENTE...
  [2] 320.0s-365.0s: OUTRO TITULO...
  [3] 580.0s-625.0s: TERCEIRO TITULO...

[RENDER] Corte moldura: 1080x607 centralizado em 1080x1920

[RESULTADO DETALHADO]
  Cut 1: TITULO INTELIGENTE
    URL: https://s3.us-east-005.backblazeb2.com/...
    Duração: 45.0s
```

---

## ⚠️ PARA O AGENT 3 (WEBAPP)

O webapp DEVE:

1. **Enviar JSON correto** com todos os parâmetros
2. **Usar tipos corretos**: `true` não `"true"`, `70` não `"70"`
3. **Parsear a resposta** e usar `response.cuts[].url` para downloads
4. **Mostrar botões de download** para cada URL
5. **Adicionar logs de debug** para verificar o que está sendo enviado/recebido

Ver arquivo `DEBUG_GUIDE_v13.md` para instruções detalhadas.
