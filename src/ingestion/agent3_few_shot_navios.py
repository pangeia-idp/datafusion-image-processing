"""
agent3_few_shot_navios.py
Few-Shot Learning com CLIP para detectar navios em imagens SAR.
Roda 100% local, sem API key.
"""

import torch
import clip
import json
import pandas as pd
from pathlib import Path
from PIL import Image

SUPPORT_DIR = Path("output/support")
OUTPUT_DIR  = Path("output/deteccoes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES_NAVIOS   = ["porto", "costa"]      # positivos
CLASSES_NEGATIVAS = ["mineracao", "urbano", "vulcao"]  # negativos

# Descrições textuais para o CLIP
TEXTOS = [
    "SAR satellite image with ships or vessels in a port",
    "SAR satellite image with ships in the ocean",
    "SAR satellite image with no ships, only land",
    "SAR satellite image of urban area with no ships",
    "SAR satellite image of mine or quarry",
    "SAR satellite image of vegetation or forest",
]

print("Carregando modelo CLIP...")
device = "cuda" if torch.cuda.is_available() else "cpu"
modelo, preprocess = clip.load("ViT-B/32", device=device)
textos_tok = clip.tokenize(TEXTOS).to(device)
print(f"CLIP carregado em: {device}\n")


def extrair_embedding(path_img: Path) -> torch.Tensor:
    """Extrai o embedding visual de uma imagem com CLIP."""
    imagem = preprocess(Image.open(path_img)).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = modelo.encode_image(imagem)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding


def montar_suporte() -> dict:
    """Carrega embeddings de todas as classes."""
    suporte = {}
    todas_classes = CLASSES_NAVIOS + CLASSES_NEGATIVAS

    for classe in todas_classes:
        pasta = SUPPORT_DIR / classe
        if not pasta.exists():
            continue
        embeddings = []
        for img in sorted(pasta.glob("*.png")):
            embeddings.append(extrair_embedding(img))
        if embeddings:
            suporte[classe] = torch.cat(embeddings).mean(dim=0, keepdim=True)
            print(f"  Suporte {classe}: {len(embeddings)} imagens")

    return suporte


def classificar_imagem(path_img: Path, suporte: dict) -> dict:
    emb_alvo = extrair_embedding(path_img)

    scores = {}
    for classe, emb_suporte in suporte.items():
        sim = (emb_alvo @ emb_suporte.T).item()
        scores[classe] = round(sim, 4)

    # Média dos scores positivos vs negativos
    score_positivo = max(scores.get("porto", 0), scores.get("costa", 0))
    score_negativo = max(
        scores.get("mineracao", 0),
        scores.get("urbano", 0),
        scores.get("vulcao", 0)
    )

    # Margem: quanto o positivo supera o negativo
    margem = score_positivo - score_negativo
    tem_navio = margem > 0.01  # threshold de margem

    classe_max = max(scores, key=scores.get)
    confianca  = round(score_positivo, 4)

    return {
        "arquivo":   path_img.name,
        "classe":    path_img.parent.name,
        "tem_navio": tem_navio,
        "confianca": confianca,
        "margem":    round(margem, 4),
        "descricao": f"Positivo: {score_positivo:.3f} | Negativo: {score_negativo:.3f} | Margem: {margem:.3f}",
        "scores":    json.dumps(scores)
    }


def rodar_deteccao():
    print("Montando suporte Few-Shot...")
    suporte = montar_suporte()

    resultados = []
    for classe in CLASSES_NAVIOS:
        pasta   = SUPPORT_DIR / classe
        imagens = list(pasta.glob("*.png"))
        print(f"\n=== {classe.upper()} ({len(imagens)} imagens) ===")

        for i, img in enumerate(imagens):
            print(f"\n[{i+1}/{len(imagens)}] {img.name}")
            try:
                resultado = classificar_imagem(img, suporte)
                resultados.append(resultado)
                print(f"  Tem navio:  {resultado['tem_navio']}")
                print(f"  Confiança:  {resultado['confianca']:.2f}")
                print(f"  Scores:     {resultado['scores']}")
            except Exception as e:
                print(f"  ✗ Erro: {e}")

    df = pd.DataFrame(resultados)
    saida = OUTPUT_DIR / "deteccao_navios.csv"
    df.to_csv(saida, index=False)
    print(f"\n✓ Salvo em: {saida}")
    print(f"\nCom navio: {df['tem_navio'].sum()} | Sem navio: {(~df['tem_navio']).sum()}")
    return df

if __name__ == "__main__":
    rodar_deteccao()