import pandas as pd
from download import baixar_imagem
from processing import processar_imagem_sar

df = pd.read_excel("../data/resultados_editado.xlsx", sheet_name="Dados_Completos")

amostras = df.groupby("KMeans_Cluster").first().reset_index()

print(f"Iniciando pipeline SAR com {len(amostras)} imagens (1 por cluster)...")

for _, row in amostras.iterrows():
    stac_id = row["stac_id"]
    cluster = row["KMeans_Cluster"]
    data = str(row["datetime"])[:10]  

    print(f"\nCluster: {cluster} | Data: {data} | ID: {stac_id[:40]}...")


    partes = stac_id.split("_")
    produto = partes[3] 

    if produto != "GEO":
        print(f"Pulando {produto} (só processamos GEO por enquanto)")
        continue

    ano = data[:4]
    mes = data[:7]

    caminho_s3 = (
        f"data/capella-open-data-by-datetime/"
        f"capella-open-data-{ano}/capella-open-data-{mes}/"
        f"capella-open-data-{data}/"
        f"{stac_id}/{stac_id}.tif"
    )

    try:
        caminho_local = baixar_imagem(caminho_s3)
        processar_imagem_sar(caminho_local)
    except Exception as e:
        print(f"Erro: {e}")

print("\nPipeline concluído!")