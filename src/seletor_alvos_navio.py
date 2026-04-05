import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import os

def capturar_navios_matplotlib(url, id_imagem):
    print(f"\n--- Iniciando Captura de Navios: {id_imagem} ---")
    
    # 1. Download e Preparação
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        img_pil = Image.open(BytesIO(response.content)).convert('RGB')
        img_np = np.array(img_pil)
    except Exception as e:
        print(f"Erro ao baixar imagem: {e}")
        return

    # 2. Interface de Clique
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img_np)
    ax.set_title(f"DETECÇÃO DE NAVIOS - {id_imagem}\nClique no centro de cada ponto brilhante e aperte ENTER")
    
    print("Aguardando cliques nos navios... Use o zoom da janela se precisar de precisão.")
    pontos = plt.ginput(n=-1, timeout=0) 
    plt.close()

    # 3. Processamento dos Crops
    if pontos:
        pasta = "crops_final/COSTA_NAVIO"
        if not os.path.exists(pasta):
            os.makedirs(pasta)

        tamanho = 40 # Quadrado de 80x80 pixels em volta do navio
        
        for i, (px, py) in enumerate(pontos):
            x, y = int(px), int(py)
            
            # Cálculo dos limites
            y1, y2 = max(0, y-tamanho), min(img_np.shape[0], y+tamanho)
            x1, x2 = max(0, x-tamanho), min(img_np.shape[1], x+tamanho)
            
            crop = img_np[y1:y2, x1:x2]
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            
            nome_arq = f"{pasta}/{id_imagem}_navio_{i}.png"
            cv2.imwrite(nome_arq, crop_bgr)
            print(f"Navio capturado: {nome_arq}")
            
        print(f"\nSucesso! {len(pontos)} navios salvos em {pasta}.")
    else:
        print("Nenhum alvo selecionado.")

# --- LINK DA IMAGEM DE 2021 ---
url_2021 = "https://capella-open-data.s3.amazonaws.com/data/2021/1/13/CAPELLA_C02_SP_GEO_HH_20210113212333_20210113212357/CAPELLA_C02_SP_GEO_HH_20210113212333_20210113212357_thumb.png"

capturar_navios_matplotlib(url_2021, "C02_JAN_2021")