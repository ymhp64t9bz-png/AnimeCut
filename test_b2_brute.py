import boto3
from botocore.client import Config

SECRET_KEY = "00506496bc1450b6722b672d9a43d00605f17eadd7"
ENDPOINT = "https://s3.us-east-005.backblazeb2.com"
ACCOUNT_ID = "68702c2cbfc6"

print("BRUTE FORCE KEY ID...")

# Padrao B2 App Key ID: <AccountID (12)> + <00000000> + <Index (2 chars hex)> ? 
# Na verdade é: <Region/Cluster (3 chars?)> não... 
# App Keys geralmente sao: K005... ou 005...
# Mas se for baseada no Account ID: 
# Exemplo real: 00249823482340000000001 (Account 4982..., Cluster 002)

# Se o secret comeca com 005, o KeyID DEVE comecar com 005. 
# O AccountID 68702c2cbfc6 nao tem o 005. Entao o KeyID completo contem o AccountID.
# Tente: 005 + 68702c2cbfc6 + 0000000001

base = f"005{ACCOUNT_ID}"

for i in range(1, 10):
    suffix = f"{i:08d}" # 8 digitos? ou 9?
    # B2 key IDs sao 25 chars.
    # len("005" + "68702c2cbfc6") = 3 + 12 = 15.
    # Faltam 10 chars. 
    # Tentar padrao 0000000001
    
    key_id = f"{base}{i:010d}" # 10 digitos sufixo?
    print(f"Tentando: {key_id} (len={len(key_id)})")
    
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=ENDPOINT,
            aws_access_key_id=key_id,
            aws_secret_access_key=SECRET_KEY,
            config=Config(signature_version="s3v4", connect_timeout=1, read_timeout=1)
        )
        s3.list_buckets()
        print(f"!!! ACHAMOS !!! Key ID: {key_id}")
        break
    except Exception as e:
        if "InvalidAccessKeyId" not in str(e):
            print(f"Erro diferente: {e}")
        pass
