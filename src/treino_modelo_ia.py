import os
import matplotlib.pyplot as plt
from fastai.vision.all import *

# 1. Configurações de Caminho
path = Path('dataset_local')
model_dir = Path('models')
model_dir.mkdir(exist_ok=True)

# 2. Augmentation (Melhorado para evitar overfitting)
batch_tfms = [
    Rotate(max_deg=180, p=1.0),
    Flip(p=0.5),
    Brightness(max_lighting=0.3, p=0.7), # Variabilidade de luz evita que decore brilho fixo
    Contrast(max_lighting=0.3, p=0.7),
    Zoom(max_zoom=1.1, p=0.5)
]

# 3. DataLoaders (Aumento do Batch Size para estabilizar o gradiente)
dls = ImageDataLoaders.from_folder(
    path,
    valid_pct=0.2,
    seed=42, # Semente fixa garante reprodutibilidade (bom para pesquisa!)
    item_tfms=Resize(224),
    batch_tfms=batch_tfms,
    bs=16
)

# 4. Criando o Modelo com Regularização (resnet18)
# wd (weight decay) ajuda a evitar overfitting
learn = vision_learner(dls, resnet18, metrics=accuracy, wd=0.1)

# 5. Treino Inteligente
print("🧠 Iniciando aprendizado profundo com controle de Fitting...")

# Adicionamos o EarlyStoppingCallback: 
# Se o modelo não melhorar por 3 épocas seguidas, ele para sozinho.
cbs = [EarlyStoppingCallback(monitor='valid_loss', min_delta=0.01, patience=3)]

# learn.fine_tune gerencia automaticamente o congelamento/descongelamento da rede
# para evitar que ela "esqueça" o que já sabe (combate underfitting inicial)
learn.fine_tune(15, cbs=cbs)

# 6. Salvamento Robusto do Modelo
export_path = model_dir/'meu_modelo_sar_v2.pkl'
learn.export(export_path)
print(f"✅ Modelo salvo em: {export_path}")

# 7. Geração e Salvamento da Matriz de Confusão
print("📊 Gerando e salvando Matriz de Confusão...")
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

# OBRIGATÓRIO PARA TERMINAL: Salvar antes de fechar o script
plt.savefig('matriz_confusao_final.png')
plt.close() # Limpa a memória do gráfico
print("✅ Gráfico salvo como 'matriz_confusao_final.png' na raiz do projeto.")