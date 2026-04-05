import os
import shutil
import random

def organizar_dataset_para_treino(pasta_origem="crops_final", split=0.8):
    classes = os.listdir(pasta_origem)
    
    for cls in classes:
        caminho_classe = os.path.join(pasta_origem, cls)
        if not os.path.isdir(caminho_classe): continue
        
        # Lista todas as imagens da classe
        imagens = [f for f in os.listdir(caminho_classe) if f.endswith('.png')]
        random.shuffle(imagens) # Embaralha para não viciar o treino
        
        # Define o ponto de corte (80% treino, 20% validação)
        limite = int(len(imagens) * split)
        treino = imagens[:limite]
        validacao = imagens[limite:]
        
        # Cria as pastas de destino
        for fase in ['train', 'val']:
            os.makedirs(f"dataset_ia/{fase}/{cls}", exist_ok=True)
            
        # Move os arquivos
        for img in treino:
            shutil.copy(os.path.join(caminho_classe, img), f"dataset_ia/train/{cls}/{img}")
        for img in validacao:
            shutil.copy(os.path.join(caminho_classe, img), f"dataset_ia/val/{cls}/{img}")
            
    print("✅ Dataset organizado em 'dataset_ia/'! Pronto para o Deep Learning.")

# Executar
organizar_dataset_para_treino()