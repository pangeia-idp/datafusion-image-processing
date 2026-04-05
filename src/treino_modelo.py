from fastai.vision.all import *
import matplotlib.pyplot as plt

# Configuração de caminhos e reprodutibilidade
path = Path('dataset_ia')
set_seed(42, reproducible=True) 

# Data Loading: Ajuste de batch size para lidar com datasets pequenos/memória GPU
dls = ImageDataLoaders.from_folder(
    path, 
    train='train', 
    valid='val',
    bs=4, 
    item_tfms=Resize(128),
    batch_tfms=aug_transforms(mult=2.0) # Data augmentation agressivo para evitar overfitting
)

# Transfer Learning: ResNet18 pré-treinada para extração de features espaciais
learn = vision_learner(dls, resnet18, metrics=accuracy)

print("--- Iniciando o Fine-Tuning ---")
learn.fine_tune(5)

# Persistência do modelo para inferência futura
learn.export('meu_modelo_sar.pkl')
print("✅ Modelo exportado com sucesso.")

# --- Diagnóstico e Métricas de Performance ---
print("Gerando artefatos de avaliação...")

# Análise de erros via Matriz de Confusão
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(7,7))
plt.savefig('matriz_final_ia.png')

# Validação visual: Predições vs Labels reais
learn.show_results(max_n=4, figsize=(10,10))
plt.savefig('resultados_visuais.png')

# Debugging: Identificação das amostras com maior perda (casos críticos)
interp.plot_top_losses(4, figsize=(15,10))
plt.savefig('maiores_erros.png')