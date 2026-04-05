import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import csv
import json

def fazer_crop_com_clique(classe, nome_arquivo, alvo_nome, tamanho=100, 
                           salvar_formato='yolo'):
    caminho_img = Path(f"dataset_local/{classe}/{nome_arquivo}")
    
    if not caminho_img.exists():
        print(f"❌ Arquivo não encontrado: {caminho_img}")
        return None

    print(f"🖼️ Abrindo {nome_arquivo}... Clique no centro do {alvo_nome}!")
    
    img = Image.open(caminho_img)
    largura_img, altura_img = img.size  # ← Pega o tamanho real da imagem
    
    plt.figure(figsize=(10,10))
    plt.imshow(img, cmap='gray')
    plt.title(f"CLIQUE NO ALVO: {alvo_nome.upper()}\n(Imagem: {nome_arquivo})")
    
    pontos = plt.ginput(1, timeout=0)
    plt.close()

    if pontos:
        x, y = pontos[0]
        
        # --- Coordenadas do crop (pixels absolutos) ---
        left   = max(0, int(x - tamanho // 2))
        top    = max(0, int(y - tamanho // 2))
        right  = min(largura_img, int(x + tamanho // 2))
        bottom = min(altura_img,  int(y + tamanho // 2))

        # --- Salva o crop ---
        crop = img.crop((left, top, right, bottom))
        output_dir = Path("crop_img")
        output_dir.mkdir(exist_ok=True)
        nome_final = output_dir / f"alvo_{alvo_nome}_{nome_arquivo}"
        crop.save(nome_final)
        print(f"✅ Crop salvo: {nome_final}")

        # --- Monta o dicionário de coordenadas ---
        coords = {
            "arquivo":       nome_arquivo,
            "classe":        classe,
            "alvo":          alvo_nome,
            "x_centro_px":   int(x),
            "y_centro_px":   int(y),
            "box_left":      left,
            "box_top":       top,
            "box_right":     right,
            "box_bottom":    bottom,
            "img_largura":   largura_img,
            "img_altura":    altura_img,
        }

        # ── OPÇÃO 1: YOLO ──────────────────────────────────────────────────
        # Formato:  <class_id> <x_center> <y_center> <width> <height>
        # Todos os valores normalizados entre 0 e 1
        if salvar_formato == 'yolo':
            x_c  = x / largura_img
            y_c  = y / altura_img
            w    = (right - left)  / largura_img
            h    = (bottom - top)  / altura_img
            
            class_id = 0  # ← mude se tiver múltiplas classes
            
            label_dir = Path("labels")
            label_dir.mkdir(exist_ok=True)
            
            # Um arquivo .txt por imagem (padrão YOLO)
            nome_txt = label_dir / (Path(nome_arquivo).stem + ".txt")
            with open(nome_txt, 'a') as f:
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
            print(f"📄 Label YOLO salva: {nome_txt}")

        # ── OPÇÃO 2: CSV ───────────────────────────────────────────────────
        elif salvar_formato == 'csv':
            csv_path = Path("anotacoes.csv")
            escrever_header = not csv_path.exists()
            
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=coords.keys())
                if escrever_header:
                    writer.writeheader()
                writer.writerow(coords)
            print(f"📊 Coordenadas salvas no CSV: {csv_path}")

        # ── OPÇÃO 3: JSON (uma lista com tudo) ────────────────────────────
        elif salvar_formato == 'json':
            json_path = Path("anotacoes.json")
            dados = []
            if json_path.exists():
                with open(json_path) as f:
                    dados = json.load(f)
            dados.append(coords)
            with open(json_path, 'w') as f:
                json.dump(dados, f, indent=2)
            print(f"🗂️ Coordenadas salvas no JSON: {json_path}")

        return coords
    return None


# ── Configuração ──────────────────────────────────────────────────────────────

FORMATO = 'yolo'   # troque para 'csv' ou 'json' conforme seu modelo

imagens_para_processar = [
    ('costa', 'img.png',  'navio'),
    ('costa', 'img2.png', 'navio'),
    ('costa', 'img4.png', 'navio'),
    ('costa', 'img5.png',  'navio'),
    ('costa', 'img6.png', 'navio'),
    ('costa', 'img7.png', 'navio'),
    ('costa', 'img8.png',  'navio'),
    ('costa', 'img9.png', 'navio'),
    ('costa', 'img10.png', 'navio'),
]

print("🚀 Iniciando extração de alvos...")

for classe, arquivo, alvo in imagens_para_processar:
    fazer_crop_com_clique(classe, arquivo, alvo, salvar_formato=FORMATO)

print("🏁 Todos os alvos foram processados!")