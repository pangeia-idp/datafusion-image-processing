from fastai.vision.all import *
import matplotlib.pyplot as plt

# 1. Definir o caminho dos dados organizados
path = Path('dataset_ia')

# 2. Criar o DataBlock (Como a IA deve ler suas pastas)
dls = ImageDataLoaders.from_folder(
    path, 
    train='train', 
    valid='val',
    bs=4, # <--- DIMINUA O BATCH SIZE (tente 2 ou 4 se tiver poucas imagens)
    item_tfms=Resize(128),
    batch_tfms=aug_transforms(mult=2.0) # Aumenta a criação de fotos "falsas" para ajudar
)

# 3. Criar o Learner (O Aluno)
# Usamos a ResNet18, uma rede que já "sabe ver" formas básicas
learn = vision_learner(dls, resnet18, metrics=accuracy)

print("--- Iniciando o Aprendizado ---")

# 4. O Treinamento (Fine-tuning)
# O número '5' é a quantidade de vezes que ela vai ler todo o seu dataset (Epochs)
learn.fine_tune(5)

# ... (Mantenha as partes 1, 2, 3 e 4 iguais)

# 5. Salvar o modelo treinado
# Exportamos para usar depois, mas mantemos o 'learn' na memória para os gráficos
learn.export('meu_modelo_sar.pkl')
print("✅ Treino concluído! Modelo salvo como 'meu_modelo_sar.pkl'")

# --- PARTE DE DIAGNÓSTICO (Sem precisar dar load_learner) ---

print("Gerando gráficos de performance...")

# 1. Matriz de Confusão (Onde ele acertou e errou)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(7,7))
plt.savefig('matriz_final_ia.png')
print("📊 Matriz de Confusão salva como 'matriz_final_ia.png'")

# 2. Mostrar resultados reais vs previstos
learn.show_results(max_n=4, figsize=(10,10))
plt.savefig('resultados_visuais.png')
print("🖼️ Amostras de resultados salvas como 'resultados_visuais.png'")

# 3. Ver os "Top Losses" (As imagens que mais confundiram a IA)
interp.plot_top_losses(4, figsize=(15,10))
plt.savefig('maiores_erros.png')
print("⚠️ Maiores erros salvos como 'maiores_erros.png'")