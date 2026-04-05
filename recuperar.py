import pandas as pd
import requests
import os
from tqdm import tqdm

df = pd.read_csv('treino.csv')
base_path = 'dataset_local'

print("📥 Baixando imagens para treino local...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    # Criar pasta da classe
    classe = row['super_classe']
    os.makedirs(f"{base_path}/{classe}", exist_ok=True)
    
    # Limpar link e pegar thumbnail
    link_json = row['stac_browser_link'].split('.json')[0] + '.json'
    url_json = "https://" + link_json.split('external/')[1] if 'external/' in link_json else link_json
    
    try:
        data = requests.get(url_json, timeout=5).json()
        img_url = data['assets']['thumbnail']['href']
        img_data = requests.get(img_url, timeout=5).content
        
        with open(f"{base_path}/{classe}/{row['stac_id']}.png", 'wb') as f:
            f.write(img_data)
    except:
        continue # Ignora os links que falharem na primeira vez