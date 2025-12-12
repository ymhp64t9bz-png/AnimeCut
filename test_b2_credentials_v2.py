import boto3
from botocore.client import Config
import os

# Dados originais
RAW_ID = "68702c2cbfc6"
SECRET_KEY = "00506496bc1450b6722b672d9a43d00605f17eadd7"
ENDPOINT = "https://s3.us-east-005.backblazeb2.com"

# Variantes para testar
ids_to_try = [
    RAW_ID,                    # 68702c2cbfc6
    "005" + RAW_ID,           # 00568702c2cbfc6 (Padrao app key)
    "00" + RAW_ID,            # As vezes apenas 2 zeros
]

print("INICIANDO TESTE B2...")

for key_id in ids_to_try:
    print(f"\n[TESTE] Tentando Key ID: {key_id}")
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=ENDPOINT,
            aws_access_key_id=key_id,
            aws_secret_access_key=SECRET_KEY,
            config=Config(signature_version="s3v4")
        )
        
        # Tenta listar buckets
        response = s3.list_buckets()
        print(">>> SUCESSO! Conexao estabelecida.")
        print(f">>> O Key ID CORRETO E: {key_id}")
        
        print("Buckets na conta:")
        for b in response['Buckets']:
            print(f" - {b['Name']}")
            
        break # Parar se funcionar
        
    except Exception as e:
        print(f"XXX Falhou: {e}") 
