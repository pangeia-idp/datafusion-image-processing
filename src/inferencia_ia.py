import pandas as pd
from fastai.vision.all import *
import requests
from io import BytesIO
from tqdm import tqdm

# 1. Carregar a IA (ajustado para o caminho que deu certo no seu PC)
try:
    learn = load_learner('models/meu_modelo_sar.pkl')
    print("✅ Cérebro da IA carregado com sucesso!")
except Exception as e:
    print(f"❌ Erro ao carregar o modelo: {e}")
    exit()

df = pd.read_csv('teste.csv')

# 2. Lógica de URL (Removendo zeros à esquerda para o S3 da Capella)
def construir_url_correta(stac_id):
    try:
        partes = stac_id.split('_')
        data_str = partes[-2] # Ex: '20240527001929'
        
        year = data_str[:4]
        month = str(int(data_str[4:6])) # '05' -> '5'
        day = str(int(data_str[6:8]))   # '27' -> '27'
        
        return f"https://capella-open-data.s3.amazonaws.com/data/{year}/{month}/{day}/{stac_id}/{stac_id}_thumb.png"
    except:
        return None

def predizer_ia_seguro(url):
    if not url: return "Erro_URL", 0.0
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return "Nao_Disponivel", 0.0
        
        img = PILImage.create(BytesIO(r.content))
        classe, _, probs = learn.predict(img)
        return classe, probs.max().item()
    except:
        return "Erro_Download", 0.0

# 3. Execução com Barra de Progresso
print(f"🚀 Analisando {len(df)} imagens de satélite...")

# Criar a coluna de URL
df['url_ia'] = df['stac_id'].apply(construir_url_correta)

# Inicializa as colunas novas
ia_classes = []
ia_confiancas = []

# Loop com barra de progresso
for url in tqdm(df['url_ia'], desc="Processando"):
    res_classe, res_conf = predizer_ia_seguro(url)
    ia_classes.append(res_classe)
    ia_confiancas.append(res_conf)

df['ia_super_classe'] = ia_classes
df['ia_confianca'] = ia_confiancas

# 4. Salvar o resultado
df.to_csv('resultado_pesquisa_idp.csv', index=False)
print("\n✅ Finalizado! O arquivo 'resultado_pesquisa_idp.csv' está pronto.")