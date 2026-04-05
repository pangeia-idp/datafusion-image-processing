import os
from PIL import Image
import random

# 1. Caminhos
pasta_costa = 'dataset_local/costa'
fotos_originais = [f for f in os.listdir(pasta_costa) if f.endswith(('.png', '.jpg'))]

if not fotos_originais:
    print("❌ Nenhuma foto encontrada na pasta costa!")
    exit()

print(f"🔄 Gerando variações para as {len(fotos_originais)} fotos de costa...")

for img_name in fotos_originais:
    img = Image.open(os.path.join(pasta_costa, img_name))
    
    # Gerar 30 variações para cada uma (9 x 30 = 270 fotos)
    # Isso vai equilibrar bem com as 150 de minas
    for i in range(30): 
        # Rotação aleatória
        out = img.rotate(random.randint(0, 360))
        
        # Escolha da inversão (espelhamento)
        # Removi o None para evitar o TypeError e usei uma lógica de IF
        escolha_flip = random.choice([Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, "nada"])
        
        if escolha_flip != "nada":
            out = out.transpose(escolha_flip)
            
        # Salva a nova imagem com prefixo aug_
        out.save(os.path.join(pasta_costa, f"aug_{i}_{img_name}"))

print("✅ SUCESSO! Agora a sua pasta 'costa' tem dados suficientes para a IA aprender.")