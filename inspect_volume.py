import boto3
from botocore.client import Config

# ================= CONFIGURAÇÃO =================
# Substitua pela sua RunPod API Key
RUNPOD_API_KEY = "SUA_API_KEY_AQUI" 

# ID do Volume (do seu comando)
VOLUME_ID = "ra9k6gxtma" 
REGION = "us-il-1"
ENDPOINT = f"https://s3api-{REGION}.runpod.io"

# ================================================

def list_volume_contents():
    print(f"📡 Conectando ao Volume RunPod: {VOLUME_ID}")
    print(f"📍 Endpoint: {ENDPOINT}")
    
    try:
        s3 = boto3.client(
            's3',
            endpoint_url=ENDPOINT,
            aws_access_key_id=RUNPOD_API_KEY,      # Na RunPod S3, o Access Key é a API Key
            aws_secret_access_key=RUNPOD_API_KEY,  # E o Secret Key TAMBÉM é a API Key
            config=Config(signature_version='s3v4')
        )
        
        # O "Bucket" no caso é o ID do volume
        response = s3.list_objects_v2(Bucket=VOLUME_ID)
        
        if 'Contents' in response:
            print(f"\n✅ Arquivos encontrados no volume '{VOLUME_ID}':")
            print("="*60)
            for obj in response['Contents']:
                size_mb = obj['Size'] / (1024*1024)
                print(f"📄 {obj['Key']:<50} | {size_mb:>8.2f} MB")
            print("="*60)
        else:
            print("\n⚠️ O volume está vazio ou não conseguimos listar os objetos.")
            
    except Exception as e:
        print(f"\n❌ Erro ao acessar o volume:")
        print(str(e))
        print("\nDICA: Verifique se sua API Key tem permissão de 'Network Volume' e se a região está correta.")

if __name__ == "__main__":
    list_volume_contents()
