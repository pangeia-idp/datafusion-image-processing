# src/salvar_geo_crops.py
#
# Saves GEO crops for specific pairs from figura5_change_multi.py.
# Uses the exact same logic (log-ratio bbox) to ensure identical crops.
#
# Usage:
#   cd data-fusion-image-processing
#   python src/salvar_geo_crops.py

import io
import sys
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from thumbnail_to_geo import extract_geo_crop
from figura5_change_multi import (
    carregar_thumb_local, log_ratio, bbox_max_mudanca, HALF_THUMB
)

THUMBS = Path(__file__).parent.parent / "output/thumbs_aoi"
OUT    = Path(__file__).parent.parent / "output/geo_crops"

PAIRS = [
    {
        "label": "Military_San_Jose_5months",
        "stac_t1": "CAPELLA_C13_SP_GEO_HH_20250418105739_20250418105748",
        "stac_t2": "CAPELLA_C13_SP_GEO_HH_20250910071741_20250910071750",
        "folder": THUMBS / "militar_san_jose",
    },
    {
        "label": "Mining_WAngelas_WA_30days",
        "stac_t1": "CAPELLA_C14_SP_GEO_HH_20240701201753_20240701201804",
        "stac_t2": "CAPELLA_C14_SP_GEO_HH_20240731082017_20240731082028",
        "folder": THUMBS / "mineracao_wa_angelas",
    },
    {
        "label": "Military_Gibraltar_3days",
        "stac_t1": "CAPELLA_C13_SP_GEO_HH_20250923192508_20250923192516",
        "stac_t2": "CAPELLA_C13_SP_GEO_HH_20250926182152_20250926182200",
        "folder": THUMBS / "militar_gibraltar",
    },
    {
        "label": "Military_Maryland_~2months",
        "stac_t1": "CAPELLA_C08_SP_GEO_VV_20221207151056_20221207151120",
        "stac_t2": "CAPELLA_C08_SP_GEO_VV_20230224151157_20230224151220",
        "folder": THUMBS / "militar_maryland",
    },
    {
        "label": "Military_Virginia_8months",
        "stac_t1": "CAPELLA_C02_SP_GEO_HH_20210904014246_20210904014312",
        "stac_t2": "CAPELLA_C02_SP_GEO_HH_20220518133200_20220518133225",
        "folder": THUMBS / "militar_virginia",
    },
]


def salvar(geo_data: np.ndarray, path: Path):
    vmin = np.percentile(geo_data, 2)
    vmax = np.percentile(geo_data, 98)
    clipped = np.clip((geo_data - vmin) / (vmax - vmin + 1e-8), 0, 1)
    img = Image.fromarray((clipped * 255).astype(np.uint8))
    img.save(path, dpi=(300, 300))
    print(f"  saved: {path}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    for pair in PAIRS:
        label   = pair["label"]
        stac_t1 = pair["stac_t1"]
        stac_t2 = pair["stac_t2"]
        folder  = pair["folder"]
        print(f"\n[{label}]")

        arr_t1, (w1, h1), _ = carregar_thumb_local(stac_t1, folder)
        arr_t2, (w2, h2), _ = carregar_thumb_local(stac_t2, folder)

        if (w2, h2) != (w1, h1):
            img_t2 = Image.fromarray(arr_t2.astype(np.uint8)).resize((w1, h1), Image.BILINEAR)
            arr_t2 = np.array(img_t2, dtype=np.float32)

        mapa = log_ratio(arr_t1, arr_t2)
        x1, y1, x2, y2 = bbox_max_mudanca(mapa, arr_t1, arr_t2, HALF_THUMB, w1, h1)
        print(f"  bbox: ({x1},{y1})→({x2},{y2})")

        print("  Fetching T1 GEO crop...")
        try:
            crop_t1 = extract_geo_crop(stac_t1, x1, y1, x2, y2)
            salvar(crop_t1.data, OUT / f"{label}_T1_{stac_t1[24:32]}.png")
        except Exception as e:
            print(f"  ERROR T1: {e}")

        print("  Fetching T2 GEO crop...")
        try:
            crop_t2 = extract_geo_crop(stac_t2, x1, y1, x2, y2)
            salvar(crop_t2.data, OUT / f"{label}_T2_{stac_t2[24:32]}.png")
        except Exception as e:
            print(f"  ERROR T2: {e}")

    print(f"\nDone. Crops saved to: {OUT}")


if __name__ == "__main__":
    main()
