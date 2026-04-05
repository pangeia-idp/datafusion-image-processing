import pandas as pd
import requests
import os
from tqdm import tqdm
from pathlib import Path

# 1. Carregar e FILTRAR o CSV
df = pd.read_csv('treino.csv')

# --- A MÁGICA ACONTECE AQUI ---
# Filtra o DataFrame para manter apenas as linhas onde a classe é 'costa'
df = df[df['super_classe'] == 'urbano'] 
# ------------------------------

base_path = 'dataset_local'

print(f"📥 Baixando {len(df)} imagens apenas da classe COSTA...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    classe = row['super_classe']
    stac_id = row['stac_id']
    
    # Criar pasta da classe
    os.makedirs(f"{base_path}/{classe}", exist_ok=True)
    
    caminho_arquivo = f"{base_path}/{classe}/{stac_id}.png"
    
    # DICA DE ENGENHEIRA: Só baixa se o arquivo ainda não existir
    if os.path.exists(caminho_arquivo):
        continue

    # Limpar link e pegar thumbnail
    link_json = row['stac_browser_link'].split('.json')[0] + '.json'
    url_json = "https://" + link_json.split('external/')[1] if 'external/' in link_json else link_json
    
    try:
        data = requests.get(url_json, timeout=5).json()
        img_url = data['assets']['thumbnail']['href']
        img_data = requests.get(img_url, timeout=5).content
        
        with open(caminho_arquivo, 'wb') as f:
            f.write(img_data)
    except:
        continue

print(f"✅ Download da classe 'costa' finalizado em {base_path}/costa")