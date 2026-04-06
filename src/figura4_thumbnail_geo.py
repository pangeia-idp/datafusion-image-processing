# src/figura4_thumbnail_geo.py
#
# Figure 4 — Thumbnail → GEO Mapping (3 coastal examples)
# Generates a figure with 3 rows: [original thumbnail | high-resolution GEO crop]
#
# Usage:
#   cd data-fusion-image-processing
#   python src/plot_geo_crops.py

import os
import sys
import io
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from thumbnail_to_geo import extract_geo_crop, _buscar_stac

# 3 exemplos costeiros (stac_ids das thumbnails em support/costa/)
EXEMPLOS = [
    {
        "stac_id": "CAPELLA_C03_SP_GEO_HH_20220107205605_20220107205621",
        "label": "Costa — Jan 2022 (HH)",
    },
    {
        "stac_id": "CAPELLA_C10_SP_GEO_HH_20240509150102_20240509150132",
        "label": "Costa — Mai 2024 (HH)",
    },
    {
        "stac_id": "CAPELLA_C13_SP_GEO_HH_20241112185032_20241112185042",
        "label": "Costa — Nov 2024 (HH)",
    },
]

# Região selecionada na thumbnail: centro (60% da imagem)
# Será recalculado dinamicamente com base no tamanho real de cada thumbnail
MARGEM = 0.20  # ignora 20% das bordas em cada lado
MAX_GEO_PX = 2000  # tamanho máximo do recorte GEO em pixels (cada dimensão)


def carregar_thumbnail_remota(stac_id: str) -> np.ndarray:
    """Baixa a thumbnail PNG do S3 e retorna como array numpy."""
    stac = _buscar_stac(stac_id)
    thumb_url = stac["assets"]["thumbnail"]["href"]
    r = requests.get(thumb_url, timeout=15)
    img = Image.open(io.BytesIO(r.content)).convert("L")
    return np.array(img), img.size  # array + (largura, altura)


def calcular_bbox_central(w: int, h: int, margem: float = MARGEM):
    """Retorna bbox central com margem percentual em cada lado."""
    x1 = int(w * margem)
    y1 = int(h * margem)
    x2 = int(w * (1 - margem))
    y2 = int(h * (1 - margem))
    return x1, y1, x2, y2


def main():
    os.makedirs("output", exist_ok=True)

    fig, axes = plt.subplots(len(EXEMPLOS), 2, figsize=(14, 6 * len(EXEMPLOS)))
    fig.suptitle("Figure 4 — Thumbnail → GEO Mapping (Coastal Examples)",
                 fontsize=14, fontweight="bold", y=1.01)

    for i, ex in enumerate(EXEMPLOS):
        stac_id = ex["stac_id"]
        label = ex["label"]
        print(f"\n[{i+1}/{len(EXEMPLOS)}] Processing: {label}")

        # --- Thumbnail ---
        print("  Downloading thumbnail...")
        thumb_arr, (w, h) = carregar_thumbnail_remota(stac_id)
        x1, y1, x2, y2 = calcular_bbox_central(w, h)
        print(f"  Selected bbox on thumbnail: ({x1},{y1}) → ({x2},{y2})")

        # Limit crop to MAX_GEO_PX px per dimension (centered)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        half = MAX_GEO_PX // (2 * 20)  # ~20x scale thumb→GEO; crop on thumb
        tx1 = max(0, cx - half)
        ty1 = max(0, cy - half)
        tx2 = min(w, cx + half)
        ty2 = min(h, cy + half)

        ax_thumb = axes[i][0]
        ax_thumb.imshow(thumb_arr, cmap="gray")
        # Rectangle shows the effectively cropped region (limited by MAX_GEO_PX)
        rect = patches.Rectangle(
            (tx1, ty1), tx2 - tx1, ty2 - ty1,
            linewidth=2, edgecolor="red", facecolor="none"
        )
        ax_thumb.add_patch(rect)
        ax_thumb.set_title(f"{label}\nThumbnail ({w}×{h} px) — red box = mapped region",
                           fontsize=9)
        ax_thumb.axis("off")

        # --- GEO Crop ---
        print("  Fetching GEO crop (HTTP range request — may take a while)...")
        try:
            crop = extract_geo_crop(
                stac_id=stac_id,
                thumb_x1=tx1, thumb_y1=ty1,
                thumb_x2=tx2, thumb_y2=ty2,
            )
            geo_data = crop.data
            x_min, y_min, x_max, y_max = crop.bbox_original

            ax_geo = axes[i][1]
            ax_geo.imshow(
                geo_data, cmap="gray",
                vmin=np.percentile(geo_data, 2),
                vmax=np.percentile(geo_data, 98),
            )
            ax_geo.set_title(
                f"GEO Crop — 0.5 m/pixel\n"
                f"Region: x {x_min}→{x_max}, y {y_min}→{y_max}  "
                f"({geo_data.shape[1]}×{geo_data.shape[0]} px)",
                fontsize=9,
            )
            ax_geo.axis("off")
            print(f"  OK — array {geo_data.shape}")

        except Exception as e:
            axes[i][1].text(0.5, 0.5, f"Error:\n{e}",
                            ha="center", va="center", transform=axes[i][1].transAxes,
                            color="red", fontsize=9)
            axes[i][1].axis("off")
            print(f"  ERROR: {e}")

    plt.tight_layout()
    output_path = "output/geo_crops_coastal.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path}")


if __name__ == "__main__":
    main()
