from fastai.vision.all import *
import cv2
import numpy as np
import os
import csv

# --- CONFIGURAÇÕES ---
MODELO = 'dataset_local/models/meu_modelo_sar_v2.pkl'
IMG_CAMINHO = 'dataset_local/costa/img2.png'
PASTA_PROVAS = 'provas_visuais'

os.makedirs(PASTA_PROVAS, exist_ok=True)


def colorir_classes_sar(img_gray, mask_zona_costeira, mask_navios_puros):
    h, w = img_gray.shape
    colorida = np.zeros((h, w, 3), dtype=np.uint8)

    mask_navios = mask_navios_puros == 255
    mask_costa  = (mask_zona_costeira == 255) & ~mask_navios
    mask_mar    = ~mask_costa & ~mask_navios

    colorida[mask_mar]    = [139, 90, 20]
    colorida[mask_costa]  = [0, 0, 200]
    colorida[mask_navios] = [0, 220, 0]

    img_norm   = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    colorida_f = colorida.astype(np.float32)
    resultado  = cv2.addWeighted(colorida_f, 0.65, img_norm * 255, 0.35, 0)

    return resultado.astype(np.uint8)


def gerar_mascara_3classes(mask_zona_costeira, mask_navios_puros, mask_valida):
    """
    Gera máscara de 3 classes SAR:
      0   = fundo/borda preta
      85  = mar
      170 = costa/terra
      255 = navios
    """
    H, W = mask_zona_costeira.shape
    mascara = np.zeros((H, W), dtype=np.uint8)

    mask_navios = mask_navios_puros == 255
    mask_costa  = (mask_zona_costeira == 255) & ~mask_navios
    mask_mar    = (mask_valida == 255) & ~mask_costa & ~mask_navios

    mascara[mask_mar]    = 85
    mascara[mask_costa]  = 170
    mascara[mask_navios] = 255

    mascara_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    mascara_rgb[mask_mar]    = [30,  80,  180]
    mascara_rgb[mask_costa]  = [200, 50,  50]
    mascara_rgb[mask_navios] = [50,  220, 50]

    return mascara, mascara_rgb, mask_mar, mask_costa, mask_navios


def exportar_coordenadas_csv(mascara_3c, caminho_saida):
    """
    Exporta coordenadas de cada região como CSV.
    Para mar e costa: centroide + bbox de cada contorno.
    Para navios: cada região individualmente.

    Colunas: classe, x_centro, y_centro, x_min, y_min, x_max, y_max, area_px
    """
    classes = {
        85:  'mar',
        170: 'costa',
        255: 'navio',
    }

    linhas = []

    for valor, nome in classes.items():
        mask_bin = (mascara_3c == valor).astype(np.uint8) * 255

        contornos, _ = cv2.findContours(
            mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for c in contornos:
            area = cv2.contourArea(c)
            if area < 1:
                continue

            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            x, y, w, h = cv2.boundingRect(c)

            linhas.append({
                'classe':   nome,
                'x_centro': cx,
                'y_centro': cy,
                'x_min':    x,
                'y_min':    y,
                'x_max':    x + w,
                'y_max':    y + h,
                'area_px':  int(area),
            })

    linhas.sort(key=lambda r: (r['classe'], -r['area_px']))

    with open(caminho_saida, 'w', newline='', encoding='utf-8') as f:
        campos = ['classe', 'x_centro', 'y_centro', 'x_min', 'y_min', 'x_max', 'y_max', 'area_px']
        writer = csv.DictWriter(f, fieldnames=campos, delimiter=';')
        writer.writeheader()
        writer.writerows(linhas)

    print(f"  [CSV] {len(linhas)} regiões exportadas → {caminho_saida}")
    print(f"        mar   : {sum(1 for r in linhas if r['classe']=='mar')} regiões")
    print(f"        costa : {sum(1 for r in linhas if r['classe']=='costa')} regiões")
    print(f"        navio : {sum(1 for r in linhas if r['classe']=='navio')} regiões")

    return linhas


def analise_sar_reversa(caminho):
    img = cv2.imread(str(caminho), 0)
    if img is None:
        return 0, 0

    cv2.imwrite(f"{PASTA_PROVAS}/1_original.png", img)

    H, W     = img.shape
    AREA_IMG = H * W

    # ── PRÉ-PROCESSAMENTO ADAPTATIVO ─────────────────────────────────────────
    img_blur = cv2.medianBlur(img, 5)

    pixels_validos = img_blur[img_blur > 5]
    if len(pixels_validos) == 0:
        return 0, 0

    hist      = cv2.calcHist([pixels_validos.reshape(-1, 1)], [0], None, [256], [1, 256])
    hist_norm = hist.ravel() / hist.sum()
    Q         = hist_norm.cumsum()
    bins      = np.arange(1, 257)
    fn_max    = -np.inf
    otsu_val  = 1
    for i in range(1, 256):
        p1, p2 = np.hsplit(hist_norm, [i])
        q1, q2 = Q[i], Q[255] - Q[i]
        if q1 < 1e-6 or q2 < 1e-6:
            continue
        b1, b2 = np.hsplit(bins, [i])
        m1 = np.sum(p1 * b1) / q1
        m2 = np.sum(p2 * b2) / q2
        fn = q1 * q2 * (m1 - m2) ** 2
        if fn > fn_max:
            fn_max   = fn
            otsu_val = i
    otsu_val = int(otsu_val)

    _, thresh_otsu  = cv2.threshold(img_blur, otsu_val,             255, cv2.THRESH_BINARY)
    _, thresh_baixo = cv2.threshold(img_blur, int(otsu_val * 0.70), 255, cv2.THRESH_BINARY)
    thresh_todos    = cv2.bitwise_or(thresh_otsu, thresh_baixo)

    mask_valida  = (img_blur > 5).astype(np.uint8) * 255
    thresh_todos = cv2.bitwise_and(thresh_todos, mask_valida)

    cv2.imwrite(f"{PASTA_PROVAS}/2_threshold_todos.png", thresh_todos)

    # ── COSTA / ZONA URBANA ───────────────────────────────────────────────────
    kernel_urbano        = np.ones((25, 25), np.uint8)
    mask_cidade_compacta = cv2.morphologyEx(thresh_todos, cv2.MORPH_CLOSE, kernel_urbano)

    contornos_cidade, _ = cv2.findContours(
        mask_cidade_compacta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    mask_cidade_final = np.zeros_like(thresh_todos)
    area_min_cidade   = AREA_IMG * 0.0003
    for c in contornos_cidade:
        if cv2.contourArea(c) > area_min_cidade:
            cv2.drawContours(mask_cidade_final, [c], -1, 255, -1)

    mask_cidade_final = cv2.bitwise_and(mask_cidade_final, mask_valida)

    # ── NAVIOS — adaptativo ───────────────────────────────────────────────────
    mask_mar_sujo    = cv2.bitwise_not(mask_cidade_final)
    mask_mar_sujo    = cv2.bitwise_and(mask_mar_sujo, mask_valida)
    so_mar_com_ruido = cv2.bitwise_and(thresh_todos, mask_mar_sujo)

    pixels_mar       = img[(mask_mar_sujo == 255) & (img > 5)]
    brilho_mar       = float(np.mean(pixels_mar)) if len(pixels_mar) > 0 else 50.0
    BRILHO_MIN_NAVIO = brilho_mar * 1.4

    area_min_navio = max(3,   AREA_IMG * 0.00001)
    area_max_navio = min(800, AREA_IMG * 0.0008)

    diagonal         = (H**2 + W**2) ** 0.5
    DISTANCIA_MINIMA = diagonal * 0.03

    dist_da_costa = cv2.distanceTransform(
        cv2.bitwise_not(mask_cidade_final), cv2.DIST_L2, 5
    )

    contornos_mar, _ = cv2.findContours(
        so_mar_com_ruido, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    mask_navios_puros = np.zeros_like(thresh_todos)
    qtd_navios = 0

    for c in contornos_mar:
        area = cv2.contourArea(c)
        if not (area_min_navio <= area <= area_max_navio):
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        x, y, w, h = cv2.boundingRect(c)
        aspecto = max(w, h) / (min(w, h) + 1e-5)
        dist    = dist_da_costa[cy, cx]
        brilho  = img[cy, cx]

        if area >= area_min_navio:
            print(f"  candidato | area={area:.0f} dist={dist:.1f}/{DISTANCIA_MINIMA:.1f} "
                  f"brilho={brilho:.0f}/{BRILHO_MIN_NAVIO:.1f} aspecto={aspecto:.2f} "
                  f"pos=({cx},{cy})")

        if dist   < DISTANCIA_MINIMA:   continue
        if brilho < BRILHO_MIN_NAVIO:   continue
        if aspecto < 1.1 and area > 50: continue

        qtd_navios += 1
        cv2.drawContours(mask_navios_puros, [c], -1, 255, -1)

    cv2.imwrite(f"{PASTA_PROVAS}/3_sieve_navios_puros.png", mask_navios_puros)

    # ── MÁSCARA COSTEIRA ──────────────────────────────────────────────────────
    resquicios_cidade  = cv2.subtract(so_mar_com_ruido, mask_navios_puros)
    mask_zona_costeira = cv2.add(mask_cidade_final, resquicios_cidade)

    cv2.imwrite(f"{PASTA_PROVAS}/4_resquicios_da_cidade.png", resquicios_cidade)

    kernel_dilat       = np.ones((9, 9), np.uint8)
    mask_zona_costeira = cv2.dilate(mask_zona_costeira, kernel_dilat, iterations=1)

    kernel_erode       = np.ones((5, 5), np.uint8)
    mask_zona_costeira = cv2.erode(mask_zona_costeira, kernel_erode, iterations=1)

    kernel_open        = np.ones((7, 7), np.uint8)
    mask_zona_costeira = cv2.morphologyEx(mask_zona_costeira, cv2.MORPH_OPEN, kernel_open)

    mask_zona_costeira = cv2.subtract(mask_zona_costeira, mask_navios_puros)
    mask_zona_costeira = cv2.bitwise_and(mask_zona_costeira, mask_valida)

    cv2.imwrite(f"{PASTA_PROVAS}/5_mascara_zona_costeira.png", mask_zona_costeira)

    # ── MÁSCARA 3 CLASSES ─────────────────────────────────────────────────────
    mascara_3c, mascara_rgb, m_mar, m_costa, m_navios = gerar_mascara_3classes(
        mask_zona_costeira, mask_navios_puros, mask_valida
    )
    cv2.imwrite(f"{PASTA_PROVAS}/5b_mascara_3classes.png",     mascara_3c)
    cv2.imwrite(f"{PASTA_PROVAS}/5c_mascara_3classes_rgb.png", mascara_rgb)

    legenda_mask = mascara_rgb.copy()
    cv2.rectangle(legenda_mask, (5, 5), (210, 80), (20, 20, 20), -1)
    cv2.putText(legenda_mask, "MAR    = 85",  (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30,  80,  180), 1)
    cv2.putText(legenda_mask, "COSTA  = 170", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 50,  50),  1)
    cv2.putText(legenda_mask, "NAVIO  = 255", (12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,  220, 50),  1)
    cv2.imwrite(f"{PASTA_PROVAS}/5d_mascara_legenda.png", legenda_mask)

    print(f"  [MÁSCARA] pixels mar   : {m_mar.sum()}")
    print(f"  [MÁSCARA] pixels costa : {m_costa.sum()}")
    print(f"  [MÁSCARA] pixels navio : {m_navios.sum()}")

    # ── EXPORTAR COORDENADAS CSV ──────────────────────────────────────────────
    exportar_coordenadas_csv(
        mascara_3c,
        os.path.join(PASTA_PROVAS, '5e_coordenadas.csv')
    )

    # ── COLORIZAÇÃO ───────────────────────────────────────────────────────────
    img_colorida = colorir_classes_sar(img, mask_zona_costeira, mask_navios_puros)
    cv2.imwrite(f"{PASTA_PROVAS}/6_mapa_classes_sar.png", img_colorida)

    legenda = img_colorida.copy()
    cv2.rectangle(legenda, (5, 5), (230, 80), (30, 30, 30), -1)
    cv2.putText(legenda, "MAR",           (25, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (139, 90, 20),  2)
    cv2.putText(legenda, "COSTA / TERRA", (25, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,   0,  200), 2)
    cv2.putText(legenda, "NAVIOS",        (25, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220,    0), 2)
    cv2.imwrite(f"{PASTA_PROVAS}/7_mapa_com_legenda.png", legenda)

    img_res        = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    costa_vermelha = np.zeros_like(img_res)
    costa_vermelha[mask_zona_costeira == 255] = [0, 0, 255]
    img_res        = cv2.addWeighted(img_res, 1, costa_vermelha, 0.4, 0)

    contornos_finais, _ = cv2.findContours(
        mask_navios_puros, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for c in contornos_finais:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img_res, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 255, 0), 2)
    cv2.imwrite(f"{PASTA_PROVAS}/8_resultado_pericia_reversa.png", img_res)

    total_pixels = img.shape[0] * img.shape[1]
    perc_costa   = (cv2.countNonZero(mask_zona_costeira) / total_pixels) * 100

    print(f"  [DEBUG] Otsu threshold : {otsu_val:.0f}")
    print(f"  [DEBUG] Brilho mar     : {brilho_mar:.1f} | Mín navio: {BRILHO_MIN_NAVIO:.1f}")
    print(f"  [DEBUG] Área navio     : {area_min_navio:.1f} – {area_max_navio:.1f} px")
    print(f"  [DEBUG] Dist. mínima   : {DISTANCIA_MINIMA:.1f} px")

    return qtd_navios, perc_costa


def main():
    print("🧠 Iniciando Pipeline de Perícia Reversa SAR...")
    if not os.path.exists(MODELO):
        print("❌ Erro: Arquivo .pkl não encontrado.")
        return

    learn = load_learner(MODELO)
    pred, pred_idx, probs = learn.predict(IMG_CAMINHO)
    confianca = probs[pred_idx].item()

    qtd_navios, perc_costa = analise_sar_reversa(IMG_CAMINHO)

    print("-" * 50)
    print(f"📡 CLASSIFICAÇÃO IA: {pred.upper()} ({confianca:.2%})")
    print(f"🏗️  OCUPAÇÃO COSTEIRA TOTAL: {perc_costa:.2f}%")
    print(f"🚢 NAVIOS REAIS ISOLADOS: {qtd_navios}")
    print("-" * 50)
    print(f"\n✅ Saídas geradas em: ./{PASTA_PROVAS}")
    print("  5b_mascara_3classes.png         → máscara numérica (0/85/170/255)")
    print("  5c_mascara_3classes_rgb.png     → máscara colorida")
    print("  5d_mascara_legenda.png          → máscara com legenda")
    print("  5e_coordenadas.csv              → coordenadas x,y por classe")
    print("  6_mapa_classes_sar.png          → mapa colorido por classe")
    print("  7_mapa_com_legenda.png          → idem com legenda")
    print("  8_resultado_pericia_reversa.png → visualização clássica")


if __name__ == "__main__":
    main()