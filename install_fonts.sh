#!/bin/bash
# ============================================================
# AnimeCut v12.6 - Instalador de Fontes do GitHub
# Instala fontes em /workspace/fonts (Volume persistente)
# 
# USO:
#   chmod +x install_fonts.sh
#   ./install_fonts.sh
# ============================================================

set -e

# Configurações
GITHUB_REPO="https://github.com/ymhp64t9bz-png/AnimeCut.git"
GITHUB_TOKEN="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"
FONTS_DIR="/workspace/fonts"
TEMP_DIR="/tmp/animecut_fonts_$$"

echo "========================================"
echo "AnimeCut - Instalador de Fontes v12.6"
echo "========================================"
echo ""

# Cria diretórios
mkdir -p "$FONTS_DIR"
mkdir -p "$TEMP_DIR"

# Função de limpeza
cleanup() {
    echo "[CLEANUP] Removendo arquivos temporários..."
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Clone do repositório
echo "[1/4] Clonando repositório do GitHub..."
cd "$TEMP_DIR"

# Usa token para autenticação (repositório privado)
REPO_URL="https://${GITHUB_TOKEN}@github.com/ymhp64t9bz-png/AnimeCut.git"
git clone --depth 1 "$REPO_URL" repo 2>&1 | grep -v "ghp_" || {
    echo "[INFO] Tentando clone público..."
    git clone --depth 1 "$GITHUB_REPO" repo
}

echo "[2/4] Procurando fontes no repositório..."

# Entra no repositório
cd repo

# Mostra fontes encontradas
echo ""
echo "Fontes encontradas:"
find . -type f \( -iname "*.ttf" -o -iname "*.otf" -o -iname "*.TTF" -o -iname "*.OTF" \) -exec basename {} \; | sort | uniq | while read font; do
    echo "  ✓ $font"
done

echo ""
echo "[3/4] Copiando fontes para $FONTS_DIR..."

# Copia todas as fontes encontradas
COPIED=0
while IFS= read -r font; do
    cp "$font" "$FONTS_DIR/" 2>/dev/null && ((COPIED++)) || true
done < <(find . -type f \( -iname "*.ttf" -o -iname "*.otf" -o -iname "*.TTF" -o -iname "*.OTF" \))

# Ajusta permissões
chmod 644 "$FONTS_DIR"/*.ttf 2>/dev/null || true
chmod 644 "$FONTS_DIR"/*.otf 2>/dev/null || true

echo ""
echo "[4/4] Verificando instalação..."
echo ""

# Lista fontes instaladas
echo "========================================"
echo "Fontes instaladas em $FONTS_DIR:"
echo "========================================"

if ls "$FONTS_DIR"/*.ttf "$FONTS_DIR"/*.otf 2>/dev/null | head -20; then
    echo ""
else
    echo "(nenhuma fonte encontrada)"
fi

TTF_COUNT=$(ls -1 "$FONTS_DIR"/*.ttf 2>/dev/null | wc -l || echo "0")
OTF_COUNT=$(ls -1 "$FONTS_DIR"/*.otf 2>/dev/null | wc -l || echo "0")
TOTAL=$((TTF_COUNT + OTF_COUNT))

echo ""
echo "========================================"
echo "RESUMO:"
echo "  Fontes .ttf: $TTF_COUNT"
echo "  Fontes .otf: $OTF_COUNT"
echo "  Total: $TOTAL fontes"
echo "========================================"
echo ""

if [ "$TOTAL" -gt 0 ]; then
    echo "✅ Instalação concluída com sucesso!"
    echo ""
    echo "As fontes estão em: $FONTS_DIR"
    echo "O AnimeCut detectará automaticamente na próxima execução."
else
    echo "⚠️  Nenhuma fonte foi encontrada no repositório."
    echo ""
    echo "Verifique se há arquivos .ttf ou .otf em:"
    echo "  $GITHUB_REPO"
fi

echo ""
