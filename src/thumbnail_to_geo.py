# src/thumbnail_to_geo.py
#
# Abordagem 1 — Proporção Direta (Redimensionamento Espacial)
#
# Mapeia uma região selecionada na thumbnail para a imagem GEO original
# usando escala proporcional de pixels, depois lê o recorte via range
# request HTTP (sem baixar o arquivo inteiro).
#
# Uso:
#   from src.thumbnail_to_geo import recortar_de_thumbnail
#
#   recorte = recortar_de_thumbnail(
#       stac_id="CAPELLA_C13_SP_GEO_HH_20250306144636_20250306144707",
#       thumb_x1=284, thumb_y1=283,
#       thumb_x2=854, thumb_y2=851,
#   )
#   # recorte.data é um numpy array com os pixels SAR em alta resolução

# OTIMIZAR TEMPO DE RESPOSTA (OUTPUT)
import io
import requests
import numpy as np
import rasterio
from rasterio.windows import Window
from dataclasses import dataclass
from PIL import Image
import os


@dataclass
class Recorte:
    data: np.ndarray   # array 2D com os pixels SAR do recorte
    bbox_original: tuple  # (x_min, y_min, x_max, y_max) na imagem original


def _buscar_stac(stac_id: str) -> dict:
    """Busca o JSON STAC a partir do stac_id."""
    partes = stac_id.split("_")
    data = partes[5]
    ano, mes, dia = data[:4], data[4:6], data[6:8]

    url = (
        f"https://capella-open-data.s3.amazonaws.com/stac/"
        f"capella-open-data-by-datetime/"
        f"capella-open-data-{ano}/"
        f"capella-open-data-{ano}-{mes}/"
        f"capella-open-data-{ano}-{mes}-{dia}/"
        f"{stac_id}/{stac_id}.json"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()


def mapear_regiao_proporcional(
    bbox_thumb: tuple,
    dim_thumb: tuple,
    dim_original: tuple,
) -> tuple:
    """
    Converte coordenadas de pixels da thumbnail para pixels da imagem original.

    Parâmetros:
        bbox_thumb   : (x_min, y_min, x_max, y_max) na thumbnail
        dim_thumb    : (largura, altura) da thumbnail em pixels
        dim_original : (largura, altura) da imagem original em pixels

    Retorna:
        (x_min, y_min, x_max, y_max) na imagem original
    """
    x_min_t, y_min_t, x_max_t, y_max_t = bbox_thumb
    w_thumb, h_thumb = dim_thumb
    w_orig, h_orig = dim_original

    scale_x = w_orig / w_thumb
    scale_y = h_orig / h_thumb

    x_min_o = int(x_min_t * scale_x)
    y_min_o = int(y_min_t * scale_y)
    x_max_o = int(x_max_t * scale_x)
    y_max_o = int(y_max_t * scale_y)

    # Garante que não ultrapassa os limites
    x_min_o = max(0, min(x_min_o, w_orig - 1))
    y_min_o = max(0, min(y_min_o, h_orig - 1))
    x_max_o = max(0, min(x_max_o, w_orig))
    y_max_o = max(0, min(y_max_o, h_orig))

    return (x_min_o, y_min_o, x_max_o, y_max_o)


def recortar_de_thumbnail(
    stac_id: str,
    thumb_x1: int,
    thumb_y1: int,
    thumb_x2: int,
    thumb_y2: int,
) -> Recorte:
    """
    Dado um retângulo selecionado na thumbnail, retorna o recorte
    correspondente na imagem GEO original em alta resolução.

    Parâmetros:
        stac_id      : ID da imagem (ex: CAPELLA_C13_SP_GEO_HH_20250306...)
        thumb_x1/y1  : canto superior-esquerdo do retângulo na thumbnail (pixels)
        thumb_x2/y2  : canto inferior-direito do retângulo na thumbnail (pixels)

    Retorna:
        Recorte com .data (numpy array 2D) e .bbox_original (pixels na GEO)
    """
    print(f"Buscando metadados STAC para {stac_id}...")
    stac = _buscar_stac(stac_id)

    # Dimensões da imagem original via STAC (proj:shape = [linhas, colunas])
    orig_rows, orig_cols = stac["properties"]["proj:shape"]
    geo_asset = next(
        (v for v in stac["assets"].values() if v.get("type") == "image/tiff; application=geotiff"),
        None
    )
    if geo_asset is None:
        raise ValueError(f"Nenhum asset GeoTIFF encontrado para {stac_id}")
    geo_url = geo_asset["href"]

    # Dimensões da thumbnail — lidas diretamente do arquivo PNG
    thumb_url = stac["assets"]["thumbnail"]["href"]
    r = requests.get(thumb_url, timeout=15)
    img = Image.open(io.BytesIO(r.content))
    thumb_w, thumb_h = img.size  # (largura=colunas, altura=linhas)
    print(f"Thumbnail: {thumb_w}×{thumb_h} px  |  Original: {orig_cols}×{orig_rows} px")

    # Abordagem 1: proporção direta
    bbox_original = mapear_regiao_proporcional(
        bbox_thumb=(thumb_x1, thumb_y1, thumb_x2, thumb_y2),
        dim_thumb=(thumb_w, thumb_h),
        dim_original=(orig_cols, orig_rows),
    )
    x_min, y_min, x_max, y_max = bbox_original
    print(f"Bbox na original: x {x_min}→{x_max}, y {y_min}→{y_max}")

    # Lê só a janela via range request HTTP (não baixa o arquivo inteiro)
    print(f"Fazendo range request para {geo_url.split('/')[-1]}...")
    window = Window(col_off=x_min, row_off=y_min, width=x_max - x_min, height=y_max - y_min)

    with rasterio.open(geo_url) as src:
        data = src.read(1, window=window)

    print(f"Recorte concluído: array {data.shape}")
    return Recorte(data=data, bbox_original=bbox_original)


# --- Exemplo de uso direto ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    stac_id = "CAPELLA_C13_SP_GEO_HH_20250306144636_20250306144707"

    # Seleciona o quarto central da thumbnail (região de exemplo)
    # Thumbnail é ~1130x1138 — pegamos o centro
    recorte = recortar_de_thumbnail(
        stac_id=stac_id,
        thumb_x1=284, thumb_y1=283,
        thumb_x2=854, thumb_y2=851,
    )

    print(f"\nBbox na imagem original: {recorte.bbox_original}")

    plt.figure(figsize=(8, 8))
    plt.imshow(recorte.data, cmap="gray",
               vmin=np.percentile(recorte.data, 2),
               vmax=np.percentile(recorte.data, 98))
    plt.title(f"Recorte SAR — {stac_id}")
    plt.colorbar(label="Backscatter (DN)")
    plt.tight_layout()
    
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/recorte_exemplo.png", dpi=150)
    print("Salvo em output/recorte_exemplo.png")
