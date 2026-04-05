import os
from PIL import Image
import random
from pathlib import Path

# 1. Caminho para a pasta não urbana
path_nao_urbano = Path('dataset_local/nao_urbano')
fotos_originais = [f for f in os.listdir(path_nao_urbano) if f.lower().endswith(('.png', '.jpg')) and not f.startswith('aug_')]

if not fotos_originais:
    print("❌ Nenhuma foto original encontrada em nao_urbano!")
    exit()

print(f"🌿 Gerando variações para as {len(fotos_originais)} fotos de vegetação/solo...")

# Vamos gerar 3 variações para cada (168 x 3 = 504 fotos no total)
for img_name in fotos_originais:
    try:
        img = Image.open(path_nao_urbano/img_name)
        
        for i in range(3): 
            # Rotação aleatória
            out = img.rotate(random.randint(0, 360))
            
            # Espelhamento aleatório
            escolha_flip = random.choice([Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, "nada"])
            if escolha_flip != "nada":
                out = out.transpose(escolha_flip)
                
            # Salva com o prefixo 'aug_nu_'
            out.save(path_nao_urbano/f"aug_nu_{i}_{img_name}")
    except Exception as e:
        print(f"⚠️ Erro no arquivo {img_name}: {e}")

print(f"SUCESSO! O dataset 'nao_urbano' agora deve ter mais de 500 imagens.")