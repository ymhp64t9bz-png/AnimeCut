#!/bin/bash
# ============================================================
# Script para instalar fontes no Volume do RunPod
# AnimeCut v12.6
# ============================================================

FONTS_DIR="/workspace/fonts"
GITHUB_REPO="https://github.com/SEU_USUARIO/SEU_REPO/archive/refs/heads/main.zip"

echo "========================================"
echo "AnimeCut - Instalador de Fontes"
echo "========================================"

# Cria diretório de fontes
mkdir -p "$FONTS_DIR"

# Método 1: Se você tem as fontes em um repositório GitHub
# Descomente e ajuste a URL abaixo:
# echo "[1/3] Baixando fontes do GitHub..."
# cd /tmp
# wget -q "$GITHUB_REPO" -O fonts.zip
# unzip -q fonts.zip -d fonts_temp
# cp fonts_temp/*/fonts/*.ttf "$FONTS_DIR/" 2>/dev/null || true
# cp fonts_temp/*/fonts/*.otf "$FONTS_DIR/" 2>/dev/null || true
# rm -rf fonts.zip fonts_temp

# Método 2: Copiar fontes do sistema (fontes gratuitas)
echo "[1/3] Copiando fontes do sistema..."
cp /usr/share/fonts/truetype/dejavu/*.ttf "$FONTS_DIR/" 2>/dev/null || true
cp /usr/share/fonts/truetype/liberation/*.ttf "$FONTS_DIR/" 2>/dev/null || true

# Método 3: Baixar fontes populares gratuitas
echo "[2/3] Baixando fontes populares..."

# Roboto (Google Fonts - Licença Apache 2.0)
if [ ! -f "$FONTS_DIR/Roboto-Bold.ttf" ]; then
    echo "  - Baixando Roboto..."
    wget -q "https://github.com/googlefonts/roboto/releases/download/v2.138/roboto-android.zip" -O /tmp/roboto.zip
    unzip -q /tmp/roboto.zip -d /tmp/roboto
    cp /tmp/roboto/*.ttf "$FONTS_DIR/" 2>/dev/null || true
    rm -rf /tmp/roboto.zip /tmp/roboto
fi

# Open Sans (Google Fonts - Licença Apache 2.0)  
if [ ! -f "$FONTS_DIR/OpenSans-Bold.ttf" ]; then
    echo "  - Baixando Open Sans..."
    wget -q "https://fonts.google.com/download?family=Open%20Sans" -O /tmp/opensans.zip 2>/dev/null
    if [ -f /tmp/opensans.zip ]; then
        unzip -q /tmp/opensans.zip -d /tmp/opensans
        cp /tmp/opensans/static/*.ttf "$FONTS_DIR/" 2>/dev/null || true
        rm -rf /tmp/opensans.zip /tmp/opensans
    fi
fi

# Montserrat (Google Fonts - OFL License)
if [ ! -f "$FONTS_DIR/Montserrat-Bold.ttf" ]; then
    echo "  - Baixando Montserrat..."
    wget -q "https://fonts.google.com/download?family=Montserrat" -O /tmp/montserrat.zip 2>/dev/null
    if [ -f /tmp/montserrat.zip ]; then
        unzip -q /tmp/montserrat.zip -d /tmp/montserrat
        cp /tmp/montserrat/static/*.ttf "$FONTS_DIR/" 2>/dev/null || true
        rm -rf /tmp/montserrat.zip /tmp/montserrat
    fi
fi

# Verificação
echo "[3/3] Verificando fontes instaladas..."
echo ""
echo "Fontes em $FONTS_DIR:"
ls -la "$FONTS_DIR"/*.ttf 2>/dev/null | wc -l | xargs echo "Total de fontes .ttf:"
ls -la "$FONTS_DIR"/*.otf 2>/dev/null | wc -l | xargs echo "Total de fontes .otf:"

echo ""
echo "========================================"
echo "Instalação concluída!"
echo ""
echo "Para usar fontes personalizadas:"
echo "1. Faça upload das suas fontes .ttf ou .otf para:"
echo "   $FONTS_DIR"
echo ""
echo "2. Reinicie o pod para que as fontes sejam detectadas"
echo "========================================"
