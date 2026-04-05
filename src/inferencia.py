import cv2
import numpy as np
from fastai.vision.all import load_learner, Path
import os

class SARDetections:
    def __init__(self, model_path):
        print("🚢 Carregando inteligência artificial multiclasse...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")
        
        self.learn = load_learner(model_path)
        self.classes = self.learn.dls.vocab 

    def predict(self, img_path):
        # 1. PREDIÇÃO DA IA
        pred, pred_idx, probs = self.learn.predict(img_path)
        confianca = float(probs[pred_idx])
        classe_predita = str(pred).lower()

        # 2. LÓGICA DE DECISÃO (Branching)
        # Dependendo do que a IA diz, o software toma um caminho diferente
        resultado_pdi = self._processar_por_contexto(img_path, classe_predita)

        return {
            "ambiente": classe_predita,
            "confianca": confianca,
            "detalhes": resultado_pdi
        }

    def _processar_por_contexto(self, img_path, classe):
        """
        Aqui você integra as lógicas de PDI que criamos antes.
        """
        img = cv2.imread(img_path)
        
        if classe == 'costa':
            # Aqui entraria a sua lógica de contar navios
            return "🔍 Iniciando contagem de alvos navais isolados..."
        
        elif classe == 'urbano':
            # Aqui entraria a lógica de densidade de pixels e máscaras azuis
            return "🏙️ Calculando métricas de densidade portuária/urbana..."
        
        elif classe == 'nao_urbano':
            # Área limpa: mata, campo ou mar sem nada
            return "🍃 Área rural ou preservada detectada. Baixa refletividade."
        
        return "❓ Contexto desconhecido."

# --- TESTE DO SISTEMA ---
if __name__ == "__main__":
    MODELO = 'dataset_local/models/meu_modelo_sar_v2.pkl'
    IMAGEM = 'dataset_local/nao_urbano/img.png'

    try:
        detector = SARDetections(MODELO)
        res = detector.predict(IMAGEM)

        print("\n" + "="*40)
        print(f"🌍 AMBIENTE DETECTADO: {res['ambiente'].upper()}")
        print(f"📊 CONFIANÇA: {res['confianca']:.2%}")
        print("="*40)
        
    except Exception as e:
        print(f"❌ Erro na inferência: {e}")