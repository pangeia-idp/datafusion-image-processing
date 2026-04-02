# src/ingestion/download.py
import pandas as pd
import boto3
import requests
import os
import yaml
from botocore import UNSIGNED
from botocore.config import Config

# Carrega config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# S3 público sem autenticação
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
BUCKET = "capella-open-data"

CSV_URL = "https://www.grss-ieee.org/wp-content/uploads/2026/02/Capella_IEEE_DataContest_2026.csv"
PRODUTOS = ["GEO", "SLC"]  # os dois produtos que você precisa


def construir_stac_url(stac_id: str) -> str:
    """Monta a URL do JSON STAC a partir do stac_id."""
    partes = stac_id.split("_")
    # Ex: CAPELLA_C13_SP_GEO_HH_20251112022441_20251112022453
    data = partes[5]
    ano  = data[:4]
    mes  = data[4:6]
    dia  = data[6:8]

    return (
        f"https://capella-open-data.s3.amazonaws.com/stac/"
        f"capella-open-data-by-datetime/"
        f"capella-open-data-{ano}/"
        f"capella-open-data-{ano}-{mes}/"
        f"capella-open-data-{ano}-{mes}-{dia}/"
        f"{stac_id}/{stac_id}.json"
    )


def baixar_assets(collect_id: str, produtos: list, destino: str):
    """Baixa os assets de um collect_id."""
    stac_url = construir_stac_url(collect_id)

    resp = requests.get(stac_url)
    if resp.status_code != 200:
        print(f"  ✗ STAC não encontrado: {collect_id}")
        return

    item = resp.json()
    os.makedirs(destino, exist_ok=True)

    for asset_name, asset in item["assets"].items():
        # Pula preview e thumbnail, baixa HH e metadata
        if asset_name in ["preview", "thumbnail"]:
            continue

        href = asset["href"]
        filename = href.split("/")[-1]
        caminho = os.path.join(destino, filename)

        if os.path.exists(caminho):
            print(f"  já existe: {filename}")
            continue

        print(f"  ↓ {filename}")
        s3_key = href.replace(f"https://{BUCKET}.s3.amazonaws.com/", "")
        s3.download_file(BUCKET, s3_key, caminho)

    print(f"  ✓ {collect_id}")


def baixar_amostra(n: int = 5):
    """Baixa os primeiros n pares do contest para teste."""
    print("Baixando CSV do contest...")
    df = pd.read_csv(cfg["paths"]["csv_contest"])
    print(f"Total de pares disponíveis: {len(df)}")

    # Pega os primeiros n collect_ids únicos de referência
    ids_teste = df["stac_id"].unique()[:n]
    print(f"\nBaixando {n} imagens de teste...\n")

    destino = cfg["paths"]["raw"]

    for i, collect_id in enumerate(ids_teste):
        print(f"[{i+1}/{n}] {collect_id}")
        baixar_assets(collect_id, PRODUTOS, destino)
        print()

    print("Download de teste concluído!")
    print(f"Arquivos salvos em: {destino}")


if __name__ == "__main__":
    baixar_amostra(n=5)  # mude para baixar mais ou menos