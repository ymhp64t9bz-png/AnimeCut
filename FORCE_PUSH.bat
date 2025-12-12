@echo off
echo ========================================================
echo ENVIANDO ATUALIZACOES PARA O GITHUB (AnimeCut)
echo ========================================================

cd /d "C:\AutoCortes\Animecut-Serverless-Clean"

echo 1. Adicionando arquivos...
git add Dockerfile handler.py requirements.txt

echo 2. Verificando status...
git status

echo 3. Commitando alteracoes ANTI-CACHE...
git commit -m "fix(runpod): force rebuild v8 with anti-cache and full handler"

echo 4. Enviando para o GitHub (Push)...
git push origin master

echo ========================================================
echo CONCLUIDO! AGORA FACA REBUILD NO RUNPOD.
echo ========================================================
pause
