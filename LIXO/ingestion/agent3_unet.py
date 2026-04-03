"""
agent3_unet.py
U-Net pré-treinada para segmentação de aeroportos e zonas urbanas
em imagens SAR da Capella.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import json

SUPPORT_DIR = Path("output/support")
OUTPUT_DIR  = Path("output/deteccoes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES_ALVO      = ["urbano", "militar"]
CLASSES_NEGATIVAS = ["mineracao", "porto", "costa", "vulcao"]

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE   = 224
BATCH_SIZE = 4
EPOCHS     = 10
LR         = 1e-4

print(f"Usando: {DEVICE}")


class SARDataset(Dataset):
    def __init__(self, imagens: list, labels: list, transform=None):
        self.imagens   = imagens
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.imagens)

    def __getitem__(self, idx):
        img   = Image.open(self.imagens[idx]).convert("RGB")
        img   = np.array(img)
        label = self.labels[idx]
        mask  = np.ones((img.shape[0], img.shape[1]),
                        dtype=np.float32) * label
        if self.transform:
            aug  = self.transform(image=img, mask=mask)
            img  = aug["image"]
            mask = aug["mask"].unsqueeze(0)
        return img, mask


def montar_dataset():
    imagens, labels = [], []
    for classe in CLASSES_ALVO:
        pasta = SUPPORT_DIR / classe
        for img in pasta.glob("*.png"):
            imagens.append(img)
            labels.append(1.0)
    for classe in CLASSES_NEGATIVAS:
        pasta = SUPPORT_DIR / classe
        if not pasta.exists():
            continue
        for img in pasta.glob("*.png"):
            imagens.append(img)
            labels.append(0.0)
    print(f"  Positivos (urbano/militar): "
          f"{sum(1 for l in labels if l == 1.0)}")
    print(f"  Negativos:                 "
          f"{sum(1 for l in labels if l == 0.0)}")
    return imagens, labels


def montar_transforms():
    treino = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    val = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    return treino, val


def criar_modelo():
    modelo = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    )
    return modelo.to(DEVICE)


def treinar(modelo, loader, otimizador, criterio):
    modelo.train()
    loss_total = 0
    for imgs, masks in loader:
        imgs  = imgs.to(DEVICE)
        masks = masks.to(DEVICE)
        otimizador.zero_grad()
        preds = modelo(imgs)
        loss  = criterio(preds, masks)
        loss.backward()
        otimizador.step()
        loss_total += loss.item()
    return loss_total / len(loader)


def inferir(modelo, path_img: Path, transform) -> dict:
    img  = Image.open(path_img).convert("RGB")
    img  = np.array(img)
    orig_h, orig_w = img.shape[:2]
    aug    = transform(image=img, mask=np.zeros((orig_h, orig_w),
                                                dtype=np.float32))
    tensor = aug["image"].unsqueeze(0).to(DEVICE)
    modelo.eval()
    with torch.no_grad():
        pred = modelo(tensor)
        prob = torch.sigmoid(pred).squeeze().cpu().numpy()
    score    = float(prob.mean())
    y_idx, x_idx = np.where(prob > 0.5)
    if len(y_idx) > 0:
        roi = {
            "x_min": int(x_idx.min() / IMG_SIZE * 100),
            "y_min": int(y_idx.min() / IMG_SIZE * 100),
            "x_max": int(x_idx.max() / IMG_SIZE * 100),
            "y_max": int(y_idx.max() / IMG_SIZE * 100)
        }
    else:
        roi = {"x_min": 0, "y_min": 0, "x_max": 100, "y_max": 100}
    return {
        "arquivo": path_img.name,
        "classe":  path_img.parent.name,
        "score":   round(score, 4),
        "roi":     json.dumps(roi)
    }


def rodar_unet():
    print("=== U-Net para Aeroportos e Zonas Urbanas ===\n")

    print("Montando dataset...")
    imagens, labels = montar_dataset()
    transform_treino, transform_val = montar_transforms()

    dataset    = SARDataset(imagens, labels, transform=transform_treino)
    loader     = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("\nCarregando U-Net pré-treinada (ResNet34 + ImageNet)...")
    modelo     = criar_modelo()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=LR)
    criterio   = nn.BCEWithLogitsLoss()

    print(f"\nTreinando por {EPOCHS} épocas...")
    for epoch in range(EPOCHS):
        loss = treinar(modelo, loader, otimizador, criterio)
        print(f"  Época {epoch+1:02d}/{EPOCHS} | Loss: {loss:.4f}")

    modelo_path = OUTPUT_DIR / "unet_urbano.pt"
    torch.save(modelo.state_dict(), str(modelo_path))
    print(f"\n✓ Modelo salvo em: {modelo_path}")

    # Inferência — coleta todos os scores primeiro
    print("\n=== Inferência ===")
    resultados = []
    todas_classes = CLASSES_ALVO + CLASSES_NEGATIVAS

    for classe in todas_classes:
        pasta = SUPPORT_DIR / classe
        if not pasta.exists():
            continue
        for img in pasta.glob("*.png"):
            resultado = inferir(modelo, img, transform_val)
            resultados.append(resultado)

    # Threshold adaptativo
    scores_alvo = [r["score"] for r in resultados
                   if r["classe"] in CLASSES_ALVO]
    scores_neg  = [r["score"] for r in resultados
                   if r["classe"] in CLASSES_NEGATIVAS]

    threshold = (np.mean(scores_alvo) + np.mean(scores_neg)) / 2
    print(f"\nThreshold adaptativo: {threshold:.3f}")
    print(f"Score médio alvos:     {np.mean(scores_alvo):.3f}")
    print(f"Score médio negativos: {np.mean(scores_neg):.3f}\n")

    # Aplica threshold e mostra resultados
    for r in resultados:
        r["eh_alvo"] = r["score"] > threshold
        print(f"  {r['classe']:12} | Score: {r['score']:.3f} "
              f"| É alvo: {r['eh_alvo']}")

    # Salva CSV
    df    = pd.DataFrame(resultados)
    saida = OUTPUT_DIR / "deteccao_urbano.csv"
    df.to_csv(saida, index=False)
    print(f"\n✓ Resultados salvos em: {saida}")

    print("\n=== Resumo ===")
    print(f"Total analisado: {len(df)}")
    print(f"Detectado alvo:  {df['eh_alvo'].sum()}")
    print(f"Não alvo:        {(~df['eh_alvo']).sum()}")

    return df


if __name__ == "__main__":
    rodar_unet()