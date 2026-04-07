# src/baixar_thumbs_aoi.py
#
# Downloads thumbnails for specific AOIs from S3 to form temporal pairs.
# Reads stac_ids directly from resultados_editado.xlsx (sheet Dados_Completos).
#
# Usage:
#   cd data-fusion-image-processing
#   python src/baixar_thumbs_aoi.py

import io
import requests
import pandas as pd
from pathlib import Path
from PIL import Image

XLSX = Path(__file__).parent.parent.parent / "data-fusion-agent/resultados_editado.xlsx"
OUT  = Path(__file__).parent.parent / "output/thumbs_aoi"

# Target AOIs: (folder_name, center_lat, center_lon, tolerance_degrees)
AOIS = [
    ("mineracao_wa_c13",    -21.8, 122.2, 0.3),
    ("urbano_tasmania_c13", -43.0, 147.3, 0.3),
    ("costa_nc",             35.3, -75.5, 0.3),
    ("vulcao_hawaii",        19.4, -155.3, 0.3),
    ("mineracao_ca",         34.8, -118.1, 0.3),
    ("mineracao_wa_angelas", -23.2, 118.8, 0.3),
    ("militar_san_jose",     37.3, -121.9, 0.3),
    ("porto_barcelona",      41.3,   2.2,  0.3),
    ("costa_sf",             37.8, -122.5, 0.3),
]

MAX_PER_AOI = 6  # thumbnails distributed over time


def _buscar_stac(stac_id: str) -> dict:
    p = stac_id.split("_")
    d = p[5]
    ano, mes, dia = d[:4], d[4:6], d[6:8]
    url = (
        f"https://capella-open-data.s3.amazonaws.com/stac/"
        f"capella-open-data-by-datetime/"
        f"capella-open-data-{ano}/capella-open-data-{ano}-{mes}/"
        f"capella-open-data-{ano}-{mes}-{dia}/{stac_id}/{stac_id}.json"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


def download_thumbnail(stac_id: str, dest: Path) -> bool:
    if dest.exists():
        print(f"  already exists: {dest.name}")
        return True
    try:
        stac = _buscar_stac(stac_id)
        thumb_url = stac["assets"]["thumbnail"]["href"]
        r = requests.get(thumb_url, timeout=15)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        sz = Image.open(io.BytesIO(r.content)).size
        print(f"  downloaded: {dest.name}  {sz}")
        return True
    except Exception as e:
        print(f"  ERROR {stac_id}: {e}")
        return False


def select_distributed(ids: list, n: int) -> list:
    ids = sorted(set(ids))
    if len(ids) <= n:
        return ids
    step = len(ids) / n
    return [ids[int(i * step)] for i in range(n)]


def main():
    print("Reading Excel...")
    df = pd.read_excel(XLSX, sheet_name="Dados_Completos")
    # Keep only GEO (not SLC)
    df = df[df["stac_id"].str.contains("_GEO_")]

    for folder_name, lat, lon, tol in AOIS:
        sub = df[
            df["center_lat"].between(lat - tol, lat + tol) &
            df["center_lon"].between(lon - tol, lon + tol)
        ].sort_values("datetime")

        print(f"\n[{folder_name}]  {len(sub)} GEOs available")
        ids = select_distributed(sub["stac_id"].tolist(), MAX_PER_AOI)
        print(f"  Selected ({len(ids)}): {[s[24:32] for s in ids]}")

        folder = OUT / folder_name
        for sid in ids:
            download_thumbnail(sid, folder / f"{sid}.png")

    print(f"\nDone. Thumbnails saved to: {OUT}")


if __name__ == "__main__":
    main()
