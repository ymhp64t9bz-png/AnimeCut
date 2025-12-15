#!/bin/bash
# ============================================================
# AnimeCut v12.6 - Setup Inicial do Volume
# Execute este script UMA VEZ ao configurar o pod
#
# O que este script faz:
#   1. Instala fontes do GitHub em /workspace/fonts
#   2. Cria estrutura de diretórios necessária
#   3. Configura permissões
#
# USO:
#   chmod +x setup_volume.sh
#   ./setup_volume.sh
# ============================================================

set -e

echo "=============================================="
echo "AnimeCut v12.6 - Setup Inicial do Volume"
echo "=============================================="
echo ""

# Configurações
WORKSPACE="/workspace"
FONTS_DIR="$WORKSPACE/fonts"
CACHE_DIR="$WORKSPACE/cache"
MODELS_DIR="$WORKSPACE/models"
TEMP_DIR="$WORKSPACE/temp"
OUTPUT_DIR="$WORKSPACE/output"

GITHUB_TOKEN="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"
GITHUB_REPO="ymhp64t9bz-png/AnimeCut"

# ==================== ESTRUTURA DE DIRETÓRIOS ====================
echo "[1/3] Criando estrutura de diretórios..."

mkdir -p "$FONTS_DIR"
mkdir -p "$CACHE_DIR/backgrounds"
mkdir -p "$CACHE_DIR/fonts"
mkdir -p "$CACHE_DIR/videos"
mkdir -p "$MODELS_DIR"
mkdir -p "$TEMP_DIR"
mkdir -p "$OUTPUT_DIR"

echo "  ✓ $FONTS_DIR"
echo "  ✓ $CACHE_DIR"
echo "  ✓ $MODELS_DIR"
echo "  ✓ $TEMP_DIR"
echo "  ✓ $OUTPUT_DIR"
echo ""

# ==================== INSTALAÇÃO DE FONTES ====================
echo "[2/3] Instalando fontes do GitHub..."

# Verifica se já tem fontes instaladas
EXISTING_FONTS=$(ls -1 "$FONTS_DIR"/*.ttf "$FONTS_DIR"/*.otf 2>/dev/null | wc -l || echo "0")

if [ "$EXISTING_FONTS" -gt 5 ]; then
    echo "  ℹ️  Já existem $EXISTING_FONTS fontes instaladas."
    echo "  Pulando download (use --force para reinstalar)"
else
    # Clone do repositório
    TEMP_CLONE="/tmp/animecut_setup_$$"
    mkdir -p "$TEMP_CLONE"
    cd "$TEMP_CLONE"
    
    echo "  Clonando repositório..."
    REPO_URL="https://${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git"
    
    if git clone --depth 1 "$REPO_URL" repo 2>/dev/null; then
        cd repo
        
        # Copia fontes
        FONT_COUNT=0
        while IFS= read -r font; do
            cp "$font" "$FONTS_DIR/" 2>/dev/null && ((FONT_COUNT++)) || true
        done < <(find . -type f \( -iname "*.ttf" -o -iname "*.otf" \))
        
        echo "  ✓ $FONT_COUNT fontes instaladas"
    else
        echo "  ⚠️  Erro ao clonar repositório"
    fi
    
    # Limpa
    rm -rf "$TEMP_CLONE"
fi

echo ""

# ==================== VERIFICAÇÃO FINAL ====================
echo "[3/3] Verificação final..."
echo ""

# Conta recursos
FONTS=$(ls -1 "$FONTS_DIR"/*.ttf "$FONTS_DIR"/*.otf 2>/dev/null | wc -l || echo "0")
CACHED_BG=$(ls -1 "$CACHE_DIR/backgrounds"/* 2>/dev/null | wc -l || echo "0")
CACHED_VIDEOS=$(ls -1 "$CACHE_DIR/videos"/* 2>/dev/null | wc -l || echo "0")

echo "=============================================="
echo "RESUMO DO VOLUME:"
echo "=============================================="
echo "  Fontes instaladas:    $FONTS"
echo "  Backgrounds em cache: $CACHED_BG"
echo "  Vídeos em cache:      $CACHED_VIDEOS"
echo ""
echo "Estrutura:"
echo "  /workspace/"
echo "  ├── fonts/        <- Fontes personalizadas"
echo "  ├── cache/        <- Cache de arquivos"
echo "  ├── models/       <- Modelos de IA"
echo "  ├── temp/         <- Arquivos temporários"
echo "  └── output/       <- Vídeos gerados"
echo "=============================================="
echo ""

if [ "$FONTS" -gt 0 ]; then
    echo "✅ Setup concluído com sucesso!"
    echo ""
    echo "Fontes disponíveis:"
    ls -1 "$FONTS_DIR"/*.ttf "$FONTS_DIR"/*.otf 2>/dev/null | xargs -I {} basename {} | head -10
    TOTAL_FONTS=$(ls -1 "$FONTS_DIR"/*.ttf "$FONTS_DIR"/*.otf 2>/dev/null | wc -l)
    if [ "$TOTAL_FONTS" -gt 10 ]; then
        echo "  ... e mais $((TOTAL_FONTS - 10)) fontes"
    fi
else
    echo "⚠️  Nenhuma fonte foi instalada."
    echo "Você pode adicionar fontes manualmente em: $FONTS_DIR"
fi

echo ""
