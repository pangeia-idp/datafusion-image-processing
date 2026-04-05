import os
from pathlib import Path

def report_dataset(base_path):
    path = Path(base_path)
    
    if not path.exists():
        print(f"❌ Erro: A pasta '{base_path}' não foi encontrada.")
        return

    print(f"📊 --- RELATÓRIO DE DISTRIBUIÇÃO DO DATASET ---")
    print(f"{'Classe':<15} | {'Qtd Imagens':<12} | {'Status'}")
    print("-" * 50)

    classes = [d for d in path.iterdir() if d.is_dir()]
    total_geral = 0

    for classe in sorted(classes):
        # Conta arquivos com extensões de imagem comuns
        arquivos = [f for f in classe.glob('*') if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        qtd = len(arquivos)
        total_geral += qtd
        
        # Lógica de status para te ajudar no IDP
        if qtd < 50:
            status = "⚠️ Poucos dados"
        elif qtd > 500:
            status = "🔥 Muito forte"
        else:
            status = "✅ Bom equilíbrio"
            
        print(f"{classe.name:<15} | {qtd:<12} | {status}")

    print("-" * 50)
    print(f"{'TOTAL GERAL':<15} | {total_geral:<12} | 🛰️ Pronto para o SAR")

if __name__ == "__main__":
    report_dataset('dataset_local')