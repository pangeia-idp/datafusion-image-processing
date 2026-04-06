# src/thumbnail_to_geo.py
#
# Approach 1 — Direct Proportion (Spatial Rescaling)
#
# Maps a region selected on the thumbnail to the original GEO image
# using proportional pixel scaling, then reads the crop via HTTP
# range request (without downloading the entire file).
#
# Usage:
#   from src.geo_crop_extractor import extract_geo_crop
#
#   crop = extract_geo_crop(
#       stac_id="CAPELLA_C13_SP_GEO_HH_20250306144636_20250306144707",
#       thumb_x1=284, thumb_y1=283,
#       thumb_x2=854, thumb_y2=851,
#   )
#   # crop.data is a 2D numpy array with high-resolution SAR pixels

import io
import os
import requests
import numpy as np
import rasterio
from rasterio.env import Env
from rasterio.windows import Window
from dataclasses import dataclass
from PIL import Image
import os


@dataclass
class Crop:
    data: np.ndarray      # 2D array with SAR pixels of the crop
    bbox_original: tuple  # (x_min, y_min, x_max, y_max) in the original image


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


def extract_geo_crop(
    stac_id: str,
    thumb_x1: int,
    thumb_y1: int,
    thumb_x2: int,
    thumb_y2: int,
) -> Crop:
    """
    Given a rectangle selected on the thumbnail, returns the corresponding
    crop from the original high-resolution GEO image.

    Parameters:
        stac_id      : image ID (e.g. CAPELLA_C13_SP_GEO_HH_20250306...)
        thumb_x1/y1  : top-left corner of the rectangle on the thumbnail (pixels)
        thumb_x2/y2  : bottom-right corner of the rectangle on the thumbnail (pixels)

    Returns:
        Crop with .data (2D numpy array) and .bbox_original (pixels in the GEO)
    """
    print(f"Fetching STAC metadata for {stac_id}...")
    stac = _buscar_stac(stac_id)

    # Original image dimensions from STAC (proj:shape = [rows, cols])
    orig_rows, orig_cols = stac["properties"]["proj:shape"]
    geo_asset = next(
        (v for v in stac["assets"].values() if v.get("type") == "image/tiff; application=geotiff"),
        None
    )
    if geo_asset is None:
        raise ValueError(f"No GeoTIFF asset found for {stac_id}")
    geo_url = geo_asset["href"]

    # Thumbnail dimensions — read directly from the PNG file
    thumb_url = stac["assets"]["thumbnail"]["href"]
    r = requests.get(thumb_url, timeout=15)
    img = Image.open(io.BytesIO(r.content))
    thumb_w, thumb_h = img.size  # (width=cols, height=rows)
    print(f"Thumbnail: {thumb_w}×{thumb_h} px  |  Original: {orig_cols}×{orig_rows} px")

    # Approach 1: direct proportion
    bbox_original = mapear_regiao_proporcional(
        bbox_thumb=(thumb_x1, thumb_y1, thumb_x2, thumb_y2),
        dim_thumb=(thumb_w, thumb_h),
        dim_original=(orig_cols, orig_rows),
    )
    x_min, y_min, x_max, y_max = bbox_original
    print(f"Bbox in original: x {x_min}→{x_max}, y {y_min}→{y_max}")

    # Read only the window via HTTP range request (no full-file download)
    # GDAL env vars optimized for COG: avoids dozens of HTTP roundtrips
    print(f"Range request for {geo_url.split('/')[-1]}...")
    window = Window(col_off=x_min, row_off=y_min, width=x_max - x_min, height=y_max - y_min)

    gdal_env = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "GDAL_HTTP_MULTIPLEX": "YES",
        "GDAL_HTTP_VERSION": "2",
        "VSI_CACHE": "TRUE",
        "VSI_CACHE_SIZE": "10000000",
    }
    with Env(**gdal_env):
        with rasterio.open(geo_url) as src:
            data = src.read(1, window=window)

    print(f"Crop complete: array {data.shape}")
    return Crop(data=data, bbox_original=bbox_original)


# --- Direct usage example ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    stac_id = "CAPELLA_C13_SP_GEO_HH_20250306144636_20250306144707"

    # Select the central quarter of the thumbnail (example region)
    # Thumbnail is ~1130x1138 — grab the center
    crop = extract_geo_crop(
        stac_id=stac_id,
        thumb_x1=284, thumb_y1=283,
        thumb_x2=854, thumb_y2=851,
    )

    print(f"\nBbox in original image: {crop.bbox_original}")

    plt.figure(figsize=(8, 8))
    plt.imshow(crop.data, cmap="gray",
               vmin=np.percentile(crop.data, 2),
               vmax=np.percentile(crop.data, 98))
    plt.title(f"SAR Crop — {stac_id}")
    plt.colorbar(label="Backscatter (DN)")
    plt.tight_layout()

    os.makedirs("output", exist_ok=True)
    plt.savefig("output/crop_example.png", dpi=150)
    print("Saved to output/crop_example.png")
