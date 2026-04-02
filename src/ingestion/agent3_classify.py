import os
import base64
import json
import requests
import pandas as pd
from pathlib import Path
import yaml

# Carrega config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

RAW     = Path(cfg["paths"]["raw"])
OUTPUT  = Path("output/classifications")
OUTPUT.mkdir(parents=True, exist_ok=True)

CENARIOS = [
    "mina",
    "porto ou navio",
    "aeroporto",
    "vulcão ou atividade vulcânica",
    "área de vegetação",
    "infraestrutura urbana",
    "base militar",
    "outro"
]

SYSTEM_PROMPT = """Você é um especialista em análise de imagens SAR (Synthetic Aperture Radar) 
de satélite. Analise thumbnails de imagens SAR e classifique o cenário visível.

Responda APENAS com um JSON válido no seguinte formato, sem texto adicional:
{
    "cenario": "<cenário principal>",
    "confianca": <0.0 a 1.0>,
    "descricao": "<breve descrição do que você vê>",
    "roi": {
        "x_min": <0 a 100>,
        "y_min": <0 a 100>,
        "x_max": <0 a 100>,
        "y_max": <0 a 100>
    }
}

Onde "roi" é a região de interesse em porcentagem da imagem (0-100).
Os cenários possíveis são: mina, porto ou navio, aeroporto, 
vulcão ou atividade vulcânica, área de vegetação, 
infraestrutura urbana, base militar, outro."""


def encode_imagem(path: str) -> str:
    """Converte imagem para base64."""
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def classificar_thumbnail(path_thumb: str) -> dict:
    """Classifica um thumbnail usando a API do Claude."""
    
    img_b64 = encode_imagem(path_thumb)
    
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"Content-Type": "application/json"},
        json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1000,
            "system": SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_b64
                            }
                        },
                        {
                            "type": "text",
                            "text": "Classifique este thumbnail SAR."
                        }
                    ]
                }
            ]
        }
    )
    
    data = response.json()
    texto = data["content"][0]["text"]
    
    # Parse do JSON retornado
    resultado = json.loads(texto)
    resultado["arquivo"] = Path(path_thumb).name
    return resultado


def rodar_classificacao():
    """Classifica todos os thumbnails disponíveis."""
    
    # Busca todos os thumbnails
    thumbs = list(RAW.glob("*_thumb.png"))
    print(f"Thumbnails encontrados: {len(thumbs)}")
    
    if len(thumbs) == 0:
        print("Nenhum thumbnail encontrado em:", RAW)
        return
    
    resultados = []
    
    for i, thumb in enumerate(thumbs):
        print(f"\n[{i+1}/{len(thumbs)}] {thumb.name}")
        
        try:
            resultado = classificar_thumbnail(str(thumb))
            resultados.append(resultado)
            print(f"  ✓ Cenário: {resultado['cenario']}")
            print(f"  ✓ Confiança: {resultado['confianca']:.0%}")
            print(f"  ✓ ROI: {resultado['roi']}")
            
        except Exception as e:
            print(f"  ✗ Erro: {e}")
            resultados.append({
                "arquivo": thumb.name,
                "cenario": "erro",
                "confianca": 0,
                "descricao": str(e),
                "roi": None
            })
    
    # Salva resultados em CSV
    df = pd.DataFrame(resultados)
    saida_csv = OUTPUT / "classificacoes.csv"
    df.to_csv(saida_csv, index=False)
    print(f"\n✓ Resultados salvos em: {saida_csv}")
    
    # Resumo por cenário
    print("\n=== Resumo por Cenário ===")
    print(df["cenario"].value_counts().to_string())
    
    return df


if __name__ == "__main__":
    rodar_classificacao()