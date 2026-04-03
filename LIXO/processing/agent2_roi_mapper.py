"""
agent2_roi_mapper.py
Mapeia ROI detectada no thumbnail para a imagem GEO original.
Usa proporção direta conforme documento Rastreio_de_coordenadas.docx
"""

import rasterio
from rasterio.windows import Window
import numpy as np
import pandas as pd
import json
from pathlib import Path
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

RAW        = Path(cfg["paths"]["raw"])
OUTPUT_DIR = Path("output/recortes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def mapear_roi_proporcional(roi_pct: dict,
                             dim_thumb: tuple,
                             dim_geo: tuple) -> dict:
    """
    Converte ROI em % do thumbnail para pixels na GEO original.

    roi_pct:   {"x_min": 0-100, "y_min": 0-100, ...}
    dim_thumb: (largura, altura) do thumbnail em pixels
    dim_geo:   (largura, altura) da GEO em pixels
    """
    w_thumb, h_thumb = dim_thumb
    w_geo,   h_geo   = dim_geo

    # Converte % → pixels no thumbnail
    x_min_t = int(roi_pct["x_min"] / 100 * w_thumb)
    y_min_t = int(roi_pct["y_min"] / 100 * h_thumb)
    x_max_t = int(roi_pct["x_max"] / 100 * w_thumb)
    y_max_t = int(roi_pct["y_max"] / 100 * h_thumb)

    # Fator de escala thumb → GEO
    scale_x = w_geo / w_thumb
    scale_y = h_geo / h_thumb

    # Projeta para a GEO
    x_min_g = int(x_min_t * scale_x)
    y_min_g = int(y_min_t * scale_y)
    x_max_g = int(x_max_t * scale_x)
    y_max_g = int(y_max_t * scale_y)

    return {
        "x_min": x_min_g,
        "y_min": y_min_g,
        "x_max": x_max_g,
        "y_max": y_max_g
    }


def recortar_geo(path_geo: Path, roi_geo: dict,
                 path_saida: Path) -> bool:
    """
    Recorta a região de interesse da imagem GEO usando Window do rasterio.
    Não carrega a imagem inteira na RAM.
    """
    with rasterio.open(path_geo) as src:
        x_min = max(0, roi_geo["x_min"])
        y_min = max(0, roi_geo["y_min"])
        x_max = min(src.width,  roi_geo["x_max"])
        y_max = min(src.height, roi_geo["y_max"])

        # Verifica se a ROI é válida
        if x_max <= x_min or y_max <= y_min:
            print(f"  ✗ ROI inválida: {roi_geo}")
            return False

        # Janela de leitura — não carrega a imagem inteira
        window = Window(
            col_off=x_min,
            row_off=y_min,
            width=x_max - x_min,
            height=y_max - y_min
        )

        # Lê só a região de interesse
        data    = src.read(window=window)
        perfil  = src.profile.copy()
        perfil.update({
            "width":     window.width,
            "height":    window.height,
            "transform": rasterio.windows.transform(window, src.transform)
        })

        path_saida.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path_saida, "w", **perfil) as dst:
            dst.write(data)

    return True


def encontrar_geo(stac_id: str) -> Path | None:
    """Encontra o arquivo GEO correspondente ao stac_id."""
    # Extrai o nome base do stac_id
    nome = stac_id.replace("_SLC_", "_GEO_")
    candidatos = list(RAW.glob(f"{nome}*.tif"))

    # Tenta match exato
    geo_exato = RAW / f"{nome}.tif"
    if geo_exato.exists():
        return geo_exato

    # Tenta match parcial
    if candidatos:
        return candidatos[0]

    return None


def rodar_mapeamento(csv_deteccoes: str):
    """
    Lê CSV de detecções (Few-Shot ou U-Net) e recorta as ROIs nas GEOs.
    """
    df = pd.read_csv(csv_deteccoes)
    print(f"Detecções carregadas: {len(df)}")

    # Filtra só os positivos
    if "tem_navio" in df.columns:
        df_pos = df[df["tem_navio"] == True]
        col_alvo = "tem_navio"
    else:
        df_pos = df[df["eh_alvo"] == True]
        col_alvo = "eh_alvo"

    print(f"Positivos para recortar: {len(df_pos)}\n")

    resultados = []

    for _, row in df_pos.iterrows():
        arquivo = row["arquivo"]
        stac_id = Path(arquivo).stem

        print(f"Processando: {stac_id[:50]}...")

        # Encontra a GEO correspondente
        path_geo = encontrar_geo(stac_id)
        if path_geo is None:
            print(f"  ✗ GEO não encontrada para: {stac_id}")
            resultados.append({
                "stac_id":   stac_id,
                "status":    "geo_nao_encontrada",
                "recorte":   None
            })
            continue

        print(f"  GEO: {path_geo.name}")

        # Pega dimensões do thumbnail e da GEO
        with rasterio.open(path_geo) as src:
            dim_geo = (src.width, src.height)

        # Thumbnail tem dimensão fixa
        dim_thumb = (1551, 1221)

        # Pega ROI do CSV
        try:
            roi_pct = json.loads(row["roi"]) if isinstance(
                row["roi"], str) else row["roi"]
        except Exception:
            roi_pct = {"x_min": 10, "y_min": 10,
                       "x_max": 90, "y_max": 90}

        # Mapeia ROI do thumb para a GEO
        roi_geo = mapear_roi_proporcional(roi_pct, dim_thumb, dim_geo)
        print(f"  ROI thumb (%): {roi_pct}")
        print(f"  ROI GEO (px):  {roi_geo}")

        # Recorta
        nome_saida = f"recorte_{stac_id}.tif"
        path_saida = OUTPUT_DIR / row["classe"] / nome_saida

        ok = recortar_geo(path_geo, roi_geo, path_saida)
        status = "ok" if ok else "erro"
        print(f"  Status: {status}")

        resultados.append({
            "stac_id":  stac_id,
            "classe":   row["classe"],
            "roi_pct":  json.dumps(roi_pct),
            "roi_geo":  json.dumps(roi_geo),
            "path_geo": str(path_geo),
            "recorte":  str(path_saida) if ok else None,
            "status":   status
        })

    # Salva CSV
    df_out = pd.DataFrame(resultados)
    saida  = Path("output/deteccoes/recortes.csv")
    df_out.to_csv(saida, index=False)
    print(f"\n✓ Recortes salvos em: {OUTPUT_DIR}")
    print(f"✓ CSV salvo em: {saida}")
    print(f"\nSucesso: {df_out[df_out['status']=='ok'].shape[0]}")
    print(f"Falhas:  {df_out[df_out['status']!='ok'].shape[0]}")

    return df_out


if __name__ == "__main__":
    # Roda para navios
    print("=== Mapeando ROIs — Navios ===")
    rodar_mapeamento("output/deteccoes/deteccao_navios.csv")

    # Roda para urbano
    print("\n=== Mapeando ROIs — Urbano ===")
    rodar_mapeamento("output/deteccoes/deteccao_urbano.csv")