import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import os

def seletor_mineracao_matplotlib(url_imagem, classe_alvo, id_imagem):
    print(f"\n--- Processando {id_imagem} ---")
    print(f"Alvo: {classe_alvo}")
    
    # 1. Download da imagem
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url_imagem, headers=headers, timeout=15)
        response.raise_for_status()
        img_pil = Image.open(BytesIO(response.content)).convert('RGB')
        img_np = np.array(img_pil)
    except Exception as e:
        print(f"Erro ao aceder a {id_imagem}: {e}")
        return

    # 2. Interface de Seleção
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img_np)
    ax.set_title(f"IMAGEM: {id_imagem}\nClique no centro das áreas de MINERAÇÃO e prima ENTER")
    
    print("Aguardando cliques... Selecione as cavas/escavações e feche a janela ou prima ENTER.")
    # Permite cliques ilimitados até que a janela seja fechada ou Enter seja premido
    pontos = plt.ginput(n=-1, timeout=0) 
    plt.close()

    # 3. Salvar os Recortes (Crops)
    if pontos:
        pasta = f"crops_final/{classe_alvo}"
        if not os.path.exists(pasta):
            os.makedirs(pasta)

        # Tamanho do recorte (120x120 pixels para capturar bem a estrutura da mina)
        tamanho = 60 
        
        for i, (px, py) in enumerate(pontos):
            x, y = int(px), int(py)
            
            # Cálculo dos limites do crop
            y1, y2 = max(0, y-tamanho), min(img_np.shape[0], y+tamanho)
            x1, x2 = max(0, x-tamanho), min(img_np.shape[1], x+tamanho)
            crop = img_np[y1:y2, x1:x2]
            
            # Conversão para BGR para salvar corretamente com OpenCV
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            nome = f"{pasta}/{id_imagem}_mina_{i}.png"
            cv2.imwrite(nome, crop_bgr)
            print(f"Recorte salvo: {nome}")
    else:
        print(f"Nenhum ponto selecionado para {id_imagem}.")

# --- LISTA ATUALIZADA COM AS 4 IMAGENS ---
links_mineracao = [
    ("https://capella-open-data.s3.amazonaws.com/data/2024/4/22/CAPELLA_C10_SP_GEO_HH_20240422145502_20240422145529/CAPELLA_C10_SP_GEO_HH_20240422145502_20240422145529_thumb.png", "MINERACAO", "C10_ABR_24"),
    ("https://capella-open-data.s3.amazonaws.com/data/2024/5/4/CAPELLA_C09_SP_GEO_HH_20240504093925_20240504093954/CAPELLA_C09_SP_GEO_HH_20240504093925_20240504093954_thumb.png", "MINERACAO", "C09_MAI_24"),
    ("https://capella-open-data.s3.amazonaws.com/data/2025/1/28/CAPELLA_C13_SP_GEO_HH_20250128083814_20250128083824/CAPELLA_C13_SP_GEO_HH_20250128083814_20250128083824_thumb.png", "MINERACAO", "C13_JAN_25_A"),
    ("https://capella-open-data.s3.amazonaws.com/data/2025/1/6/CAPELLA_C13_SP_GEO_HH_20250106201856_20250106201859/CAPELLA_C13_SP_GEO_HH_20250106201856_20250106201859_thumb.png", "MINERACAO", "C13_JAN_25_B"),
    ("https://capella-open-data.s3.amazonaws.com/data/2024/7/19/CAPELLA_C14_SP_GEO_HH_20240719130720_20240719130730/CAPELLA_C14_SP_GEO_HH_20240719130720_20240719130730_thumb.png", "MINERACAO", "C14_JUL_24")
]

# Execução em loop
for url, classe, id_img in links_mineracao:
    seletor_mineracao_matplotlib(url, classe, id_img)