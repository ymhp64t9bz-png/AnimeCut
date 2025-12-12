import boto3
from botocore.client import Config
import os

# Credenciais fornecidas
KEY_ID = "68702c2cbfc6"
SECRET_KEY = "00506496bc1450b6722b672d9a43d00605f17eadd7"
ENDPOINT = "https://s3.us-east-005.backblazeb2.com"

# Nomes possíveis de bucket
BUCKETS_TO_TRY = ["KortexClipAICriada", "KortexClipAI", "kortexclipai"]

print(f"Testando conexao B2 com KeyID: {KEY_ID}")

s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=KEY_ID,
    aws_secret_access_key=SECRET_KEY,
    config=Config(signature_version="s3v4")
)

try:
    print("Listando buckets para verificar autenticacao...")
    response = s3.list_buckets()
    print("✅ Autenticacao SUCESSO!")
    print("Buckets disponiveis:")
    found = False
    for bucket in response['Buckets']:
        print(f" - {bucket['Name']}")
        if bucket['Name'] in BUCKETS_TO_TRY:
            print(f"   -> ENCONTRADO ALVO: {bucket['Name']}")
            found = True
            
    if not found:
        print("⚠️ Nenhum dos nomes suspeitos foi encontrado na conta.")
        
except Exception as e:
    print(f"❌ FALHA na Autenticacao: {e}")
    if "InvalidAccessKeyId" in str(e):
        # Tentar adicionar prefixo '005' ao ID se for App Key
        print("\nTentando com prefixo '005' no ID...")
        try:
            s3_retry = boto3.client(
                "s3",
                endpoint_url=ENDPOINT,
                aws_access_key_id="005" + KEY_ID,
                aws_secret_access_key=SECRET_KEY,
                config=Config(signature_version="s3v4")
            )
            resp = s3_retry.list_buckets()
            print("✅ SUCESSO com prefixo 005!")
            print(f"🔑 USER O KEY_ID REAL É: 005{KEY_ID}")
            for bucket in resp['Buckets']:
                print(f" - {bucket['Name']}")
        except Exception as e2:
             print(f"❌ Falha tambem com prefixo: {e2}")
