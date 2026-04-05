from PIL import Image
from pathlib import Path
import os

path = Path('dataset_local')
arquivos = list(path.rglob('*.png'))

print(f"🔍 Verificando integridade de {len(arquivos)} imagens...")
deletados = 0

for f in arquivos:
    try:
        with Image.open(f) as img:
            img.verify() # Tenta ler o cabeçalho da imagem
    except:
        print(f"❌ Arquivo corrompido encontrado: {f}")
        os.remove(f) # Deleta o arquivo ruim
        deletados += 1

print(f"\n✅ Faxina concluída! {deletados} arquivos corrompidos foram removidos.")