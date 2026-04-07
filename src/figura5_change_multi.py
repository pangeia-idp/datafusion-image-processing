# src/figura5_change_multi.py
#
# Figure 5 — Multi-Region / Multi-Interval Change Detection
#
# For each pair (T1, T2):
#   1. Computes log-ratio on thumbnails
#   2. Locates the region of greatest change on the thumbnail
#   3. Extracts the corresponding GEO crop via HTTP range request
#   4. Displays: T1 thumb | T2 thumb | log-ratio | GEO crop | size comparison
#   5. Shows size comparison: thumbnail vs full GEO vs extracted crop
#
# Usage:
#   cd data-fusion-image-processing
#   python src/plot_change_detection.py

import io
import os
import sys
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from thumbnail_to_geo import extract_geo_crop, _buscar_stac

# ─── Image pairs ─────────────────────────────────────────────────────────────
THUMBS = Path(__file__).parent.parent / "output/thumbs_aoi"

PAIRS = [
    # ── Mining WA — Newman (C13, HH, ascending ~29°) ─────────────────────────
    {
        "label": "Mining WA — 30 days",
        "stac_t1": "CAPELLA_C13_SP_GEO_HH_20241123120750_20241123120759",
        "stac_t2": "CAPELLA_C13_SP_GEO_HH_20241223013514_20241223013517",
        "folder": THUMBS / "mineracao_wa_c13",
    },
    {
        "label": "Mining WA — 2 months",
        "stac_t1": "CAPELLA_C13_SP_GEO_HH_20241223013514_20241223013517",
        "stac_t2": "CAPELLA_C13_SP_GEO_HH_20250226022256_20250226022305",
        "folder": THUMBS / "mineracao_wa_c13",
    },
    {
        "label": "Mining WA — ~6 months",
        "stac_t1": "CAPELLA_C13_SP_GEO_HH_20241123120750_20241123120759",
        "stac_t2": "CAPELLA_C13_SP_GEO_HH_20250508010500_20250508010509",
        "folder": THUMBS / "mineracao_wa_c13",
    },
    # ── Urban Tasmania (C13, descending ~39.6°) ───────────────────────────────
    {
        "label": "Urban Tasmania — 3 days",
        "stac_t1": "CAPELLA_C13_SP_GEO_HH_20250306144636_20250306144707",
        "stac_t2": "CAPELLA_C13_SP_GEO_HH_20250309134329_20250309134400",
        "folder": THUMBS / "urbano_tasmania_c13",
    },
    {
        "label": "Urban Tasmania — 9 days",
        "stac_t1": "CAPELLA_C13_SP_GEO_HH_20250306144636_20250306144707",
        "stac_t2": "CAPELLA_C13_SP_GEO_HH_20250315113656_20250315113727",
        "folder": THUMBS / "urbano_tasmania_c13",
    },
    # ── NC Coast ──────────────────────────────────────────────────────────────
    {
        "label": "NC Coast — ~1 month",
        "stac_t1": "CAPELLA_C10_SP_GEO_HH_20240509150102_20240509150132",
        "stac_t2": "CAPELLA_C09_SP_GEO_HH_20240605032146_20240605032214",
        "folder": THUMBS / "costa_nc",
    },
    # ── Volcano Hawaii (C13, HH) ──────────────────────────────────────────────
    {
        "label": "Volcano Hawaii — ~2 months",
        "stac_t1": "CAPELLA_C13_SP_GEO_HH_20250212132904_20250212132943",
        "stac_t2": "CAPELLA_C13_SP_GEO_HH_20250413061949_20250413062001",
        "folder": THUMBS / "vulcao_hawaii",
    },
    {
        "label": "Volcano Hawaii — ~4 months",
        "stac_t1": "CAPELLA_C13_SP_GEO_HH_20250413061949_20250413062001",
        "stac_t2": "CAPELLA_C13_SP_GEO_HH_20250823165702_20250823165715",
        "folder": THUMBS / "vulcao_hawaii",
    },
    # ── Mining California (C13, HH) ───────────────────────────────────────────
    {
        "label": "Mining CA — ~2 months",
        "stac_t1": "CAPELLA_C13_SP_GEO_HH_20241127064647_20241127064656",
        "stac_t2": "CAPELLA_C13_SP_GEO_HH_20250131073458_20250131073508",
        "folder": THUMBS / "mineracao_ca",
    },
    {
        "label": "Mining CA — ~4.5 months",
        "stac_t1": "CAPELLA_C13_SP_GEO_HH_20250131073458_20250131073508",
        "stac_t2": "CAPELLA_C13_SP_GEO_HH_20250616070447_20250616070457",
        "folder": THUMBS / "mineracao_ca",
    },
    # ── Mining West Angelas WA (C10/C14, HH) ─────────────────────────────────
    {
        "label": "Mining W.Angelas WA — ~7 months",
        "stac_t1": "CAPELLA_C10_SP_GEO_HH_20230507180103_20230507180133",
        "stac_t2": "CAPELLA_C10_SP_GEO_HH_20231225171518_20231225171547",
        "folder": THUMBS / "mineracao_wa_angelas",
    },
    {
        "label": "Mining W.Angelas WA — 30 days",
        "stac_t1": "CAPELLA_C14_SP_GEO_HH_20240701201753_20240701201804",
        "stac_t2": "CAPELLA_C14_SP_GEO_HH_20240731082017_20240731082028",
        "folder": THUMBS / "mineracao_wa_angelas",
    },
    # ── Military San Jose CA (C13, HH) ────────────────────────────────────────
    {
        "label": "Military San Jose — ~3 months",
        "stac_t1": "CAPELLA_C13_SP_GEO_HH_20250110214529_20250110214532",
        "stac_t2": "CAPELLA_C13_SP_GEO_HH_20250418105739_20250418105748",
        "folder": THUMBS / "militar_san_jose",
    },
    {
        "label": "Military San Jose — ~5 months",
        "stac_t1": "CAPELLA_C13_SP_GEO_HH_20250418105739_20250418105748",
        "stac_t2": "CAPELLA_C13_SP_GEO_HH_20250910071741_20250910071750",
        "folder": THUMBS / "militar_san_jose",
    },
    # ── Port Barcelona (C02, HH) ──────────────────────────────────────────────
    {
        "label": "Port Barcelona — ~8 months",
        "stac_t1": "CAPELLA_C02_SP_GEO_HH_20210113212333_20210113212357",
        "stac_t2": "CAPELLA_C02_SP_GEO_HH_20210927092320_20210927092345",
        "folder": THUMBS / "porto_barcelona",
    },
    {
        "label": "Port Barcelona — ~2 months",
        "stac_t1": "CAPELLA_C03_SP_GEO_HH_20220201211613_20220201211630",
        "stac_t2": "CAPELLA_C03_SP_GEO_HH_20220408211608_20220408211625",
        "folder": THUMBS / "porto_barcelona",
    },
    # ── SF Coast (C13, HH) ────────────────────────────────────────────────────
    {
        "label": "SF Coast — 9 days",
        "stac_t1": "CAPELLA_C13_SP_GEO_HH_20241112185032_20241112185042",
        "stac_t2": "CAPELLA_C13_SP_GEO_HH_20241121154044_20241121154053",
        "folder": THUMBS / "costa_sf",
    },
    {
        "label": "SF Coast — 3 days",
        "stac_t1": "CAPELLA_C13_SP_GEO_HH_20241121154044_20241121154053",
        "stac_t2": "CAPELLA_C13_SP_GEO_HH_20241124143728_20241124143738",
        "folder": THUMBS / "costa_sf",
    },
]

HALF_THUMB = 50   # px on thumbnail in each direction (→ ~1000 px on GEO)


# ─── Utilities ───────────────────────────────────────────────────────────────

def carregar_thumb_local(stac_id: str, pasta: Path):
    """Carrega thumbnail já baixada. Retorna (array, (w,h), bytes)."""
    path = pasta / f"{stac_id}.png"
    with open(path, "rb") as f:
        raw = f.read()
    img = Image.open(io.BytesIO(raw)).convert("L")
    return np.array(img, dtype=np.float32), img.size, len(raw)


def log_ratio(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Log-ratio normalizado com máscara de borda."""
    mapa = np.abs(np.log10(arr1 + 1e-6) - np.log10(arr2 + 1e-6))
    mascara = (arr1 > 5) & (arr2 > 5)
    if mascara.sum() > 0:
        validos = mapa[mascara]
        mapa = (mapa - validos.min()) / (validos.max() - validos.min() + 1e-8)
    mapa = np.clip(mapa, 0, 1)
    mapa[~mascara] = 0
    return mapa


def bbox_max_mudanca(mapa: np.ndarray, arr_t1: np.ndarray, arr_t2: np.ndarray,
                     half: int, w: int, h: int):
    """Encontra centroide dos pixels de maior mudança (top 5%).
    Exclui pixels onde T1 ou T2 têm valor baixo (fundo preto / borda da imagem SAR)."""
    # Máscara: ambas as imagens devem ter sinal real (não fundo)
    mascara_valida = (arr_t1 > 10) & (arr_t2 > 10)
    mapa_filtrado = mapa.copy()
    mapa_filtrado[~mascara_valida] = 0

    if mapa_filtrado.max() == 0:
        cy, cx = h // 2, w // 2
    else:
        # Blur gaussiano para encontrar o maior CLUSTER de mudança (não pixel isolado)
        from scipy.ndimage import gaussian_filter
        suavizado = gaussian_filter(mapa_filtrado, sigma=15)
        cy, cx = np.unravel_index(np.argmax(suavizado), suavizado.shape)
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)
    return x1, y1, x2, y2


def tamanho_geo_original(stac_id: str) -> tuple:
    """Retorna (linhas, colunas, bytes_s3) da imagem GEO original via STAC + HEAD."""
    stac = _buscar_stac(stac_id)
    rows, cols = stac["properties"]["proj:shape"]
    geo_asset = next(
        (v for v in stac["assets"].values()
         if v.get("type") == "image/tiff; application=geotiff"), None
    )
    bytes_s3 = 0
    if geo_asset:
        try:
            resp = requests.head(geo_asset["href"], timeout=10)
            bytes_s3 = int(resp.headers.get("Content-Length", 0))
        except Exception:
            pass
    return rows, cols, bytes_s3


def fmt_bytes(n: int) -> str:
    if n == 0:
        return "?"
    if n >= 1e9:
        return f"{n/1e9:.1f} GB"
    if n >= 1e6:
        return f"{n/1e6:.1f} MB"
    return f"{n/1e3:.0f} KB"


# ─── Main ────────────────────────────────────────────────────────────────────

def salvar_geo_crop(geo_data: np.ndarray, stac_id: str, label: str):
    """Salva o GEO crop como PNG em output/geo_crops/."""
    out_dir = Path("output/geo_crops")
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = label.replace(" ", "_").replace("/", "-").replace("~", "").replace(".", "")
    fname = out_dir / f"{slug}_{stac_id[24:32]}_geo_crop.png"
    vmin = np.percentile(geo_data, 2)
    vmax = np.percentile(geo_data, 98)
    clipped = np.clip((geo_data - vmin) / (vmax - vmin + 1e-8), 0, 1)
    img = Image.fromarray((clipped * 255).astype(np.uint8))
    img.save(fname)
    print(f"  GEO crop saved: {fname}")


def main():
    os.makedirs("output", exist_ok=True)

    n = len(PAIRS)
    fig, axes = plt.subplots(n, 5, figsize=(25, 5 * n))
    fig.suptitle(
        "Figure 5 — Multi-Region / Multi-Interval Change Detection\n"
        "Log-Ratio on Thumbnails → High-Resolution GEO Crop",
        fontsize=13, fontweight="bold", y=1.01
    )

    for i, pair in enumerate(PAIRS):
        label   = pair["label"]
        stac_t1 = pair["stac_t1"]
        stac_t2 = pair["stac_t2"]
        folder  = pair["folder"]
        print(f"\n[{i+1}/{n}] {label}")

        # 1. Thumbnails
        print("  Loading thumbnails...")
        arr_t1, (w1, h1), sz_t1 = carregar_thumb_local(stac_t1, folder)
        arr_t2, (w2, h2), sz_t2 = carregar_thumb_local(stac_t2, folder)

        # Ensure same size for log-ratio (resize T2 → T1 size)
        if (w2, h2) != (w1, h1):
            img_t2 = Image.fromarray(arr_t2.astype(np.uint8)).resize((w1, h1), Image.BILINEAR)
            arr_t2 = np.array(img_t2, dtype=np.float32)
            w2, h2 = w1, h1

        # 2. Log-ratio
        mapa = log_ratio(arr_t1, arr_t2)

        # 3. Bbox of greatest change
        x1, y1, x2, y2 = bbox_max_mudanca(mapa, arr_t1, arr_t2, HALF_THUMB, w1, h1)
        print(f"  Greatest change region on thumb: ({x1},{y1})→({x2},{y2})")

        # 4. GEO crop
        print("  Fetching GEO crop (HTTP range request)...")
        try:
            crop = extract_geo_crop(stac_t1, x1, y1, x2, y2)
            geo_data = crop.data
            geo_crop_bytes = geo_data.nbytes
            print(f"  OK — array {geo_data.shape}  ({fmt_bytes(geo_crop_bytes)})")
            salvar_geo_crop(geo_data, stac_t1, label)
        except Exception as e:
            geo_data = None
            geo_crop_bytes = 0
            print(f"  ERROR fetching GEO crop: {e}")

        # 5. Original GEO file size (HEAD request)
        print("  Querying original GEO file size...")
        try:
            geo_rows, geo_cols, geo_s3_bytes = tamanho_geo_original(stac_t1)
        except Exception:
            geo_rows, geo_cols, geo_s3_bytes = 0, 0, 0

        geo_uncomp = geo_rows * geo_cols * 2  # uint16 = 2 bytes/pixel

        # ── Coluna 0: T1 thumbnail ──────────────────────────────────────────
        ax = axes[i][0]
        ax.imshow(arr_t1, cmap="gray")
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="cyan", facecolor="none"
        )
        ax.add_patch(rect)
        ax.set_title(
            f"T1 — {stac_t1[24:32]}\n"
            f"Thumbnail  {w1}×{h1} px  |  {fmt_bytes(sz_t1)}",
            fontsize=8
        )
        ax.axis("off")

        # ── Coluna 1: T2 thumbnail ──────────────────────────────────────────
        ax = axes[i][1]
        ax.imshow(arr_t2, cmap="gray")
        ax.set_title(
            f"T2 — {stac_t2[24:32]}\n"
            f"Thumbnail  {w2}×{h2} px  |  {fmt_bytes(sz_t2)}",
            fontsize=8
        )
        ax.axis("off")

        # ── Coluna 2: log-ratio com bbox ───────────────────────────────────
        ax = axes[i][2]
        im = ax.imshow(mapa, cmap="hot", vmin=0, vmax=1)
        rect2 = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="cyan", facecolor="none"
        )
        ax.add_patch(rect2)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(
            f"{label}\nLog-Ratio  (cyan = extracted region)",
            fontsize=8
        )
        ax.axis("off")

        # ── Coluna 3: recorte GEO ───────────────────────────────────────────
        ax = axes[i][3]
        if geo_data is not None:
            ax.imshow(
                geo_data, cmap="gray",
                vmin=np.percentile(geo_data, 2),
                vmax=np.percentile(geo_data, 98),
            )
            ax.set_title(
                f"GEO Crop — 0.5 m/pixel\n"
                f"{geo_data.shape[1]}×{geo_data.shape[0]} px  |  {fmt_bytes(geo_crop_bytes)}",
                fontsize=8
            )
        else:
            ax.text(0.5, 0.5, "GEO crop error",
                    ha="center", va="center", transform=ax.transAxes,
                    color="red", fontsize=9)
            ax.set_title("GEO Crop — error", fontsize=8)
        ax.axis("off")

        # ── Coluna 4: comparativo de tamanhos (barras) ─────────────────────
        ax = axes[i][4]
        ax.set_facecolor("#1a1a1a")
        fig.patch.set_facecolor("white")

        labels  = ["Analyzed\nThumbnail", "GEO original\n(compressed)", "GEO original\n(uncompressed)", "Extracted\nCrop"]
        valores = [sz_t1, geo_s3_bytes if geo_s3_bytes > 0 else 1,
                   geo_uncomp if geo_uncomp > 0 else 1,
                   geo_crop_bytes if geo_crop_bytes > 0 else 1]
        cores   = ["#4fc3f7", "#ef5350", "#ff8a65", "#66bb6a"]

        bars = ax.barh(labels, valores, color=cores, edgecolor="white", height=0.5)

        # Anotação com o valor legível em cada barra
        for bar, val in zip(bars, valores):
            ax.text(
                bar.get_width() * 1.02, bar.get_y() + bar.get_height() / 2,
                fmt_bytes(val), va="center", ha="left",
                fontsize=9, fontweight="bold", color="white"
            )

        ax.set_xscale("log")
        ax.set_xlim(right=max(valores) * 5)
        ax.set_xlabel("Size (log scale)", fontsize=8)
        ax.set_title(f"Size Comparison\n{label}", fontsize=8, fontweight="bold")
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.xaxis.set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    plt.tight_layout()
    output_path = "output/change_detection_multi.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path}")


if __name__ == "__main__":
    main()
