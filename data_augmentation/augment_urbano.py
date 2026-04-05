import os
from PIL import Image
import random

# 1. Caminho para a pasta urbana
pasta_urbano = 'dataset_local/urbano'
fotos_originais = [f for f in os.listdir(pasta_urbano) if f.endswith(('.png', '.jpg')) and not f.startswith('aug_')]

if not fotos_originais:
    print("❌ Nenhuma foto original encontrada na pasta urbano!")
    exit()

print(f"🏙️ Gerando variações para as {len(fotos_originais)} fotos urbanas...")

# Vamos gerar 10 variações para cada uma (23 x 10 = 230 fotos novas)
for img_name in fotos_originais:
    try:
        img = Image.open(os.path.join(pasta_urbano, img_name))
        
        for i in range(3): 
            # Rotação aleatória (importante para satélite)
            out = img.rotate(random.randint(0, 360))
            
            # Espelhamento aleatório
            escolha_flip = random.choice([Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, "nada"])
            if escolha_flip != "nada":
                out = out.transpose(escolha_flip)
                
            # Salva com o prefixo 'aug_u_' para diferenciar
            out.save(os.path.join(pasta_urbano, f"aug_u_{i}_{img_name}"))
    except:
        print(f"Pulei o arquivo {img_name} (provavelmente corrompido)")

print(f"SUCESSO! Agora sua classe 'urbano' está equilibrada com as outras.")