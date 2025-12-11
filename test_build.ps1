# Teste de Build Local - AnimeCut

# Simula o que o RunPod faz
Write-Host "🔧 Testando instalação das dependências..." -ForegroundColor Cyan

# Cria ambiente virtual
python -m venv test_env
.\test_env\Scripts\Activate.ps1

# Atualiza pip
Write-Host "`n📦 Atualizando pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

# Tenta instalar requirements.txt
Write-Host "`n📥 Instalando requirements.txt..." -ForegroundColor Yellow
pip install -r "C:\AutoCortes\Animecut-Serverless-Clean\requirements.txt"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ SUCESSO! Todas as dependências instaladas!" -ForegroundColor Green
    
    # Testa imports
    Write-Host "`n🧪 Testando imports..." -ForegroundColor Cyan
    python -c "import runpod; print('✅ runpod OK')"
    python -c "import moviepy; print('✅ moviepy OK')"
    python -c "import whisper; print('✅ whisper OK')"
    python -c "import boto3; print('✅ boto3 OK')"
    python -c "from PIL import Image; print('✅ PIL OK')"
} else {
    Write-Host "`n❌ ERRO na instalação!" -ForegroundColor Red
    Write-Host "O mesmo erro deve estar acontecendo no RunPod!" -ForegroundColor Red
}

# Desativa ambiente
deactivate

Write-Host "`n✅ Teste concluído!" -ForegroundColor Green
Write-Host "Pressione qualquer tecla para sair..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
