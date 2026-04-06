"""
test_siamese_unet.py

Log-ratio baseline using the UnifiedSiameseUNet architecture on 2 SAR thumbnails
from the mining AOI (Newman, Western Australia):
  - T1: CAPELLA_C13_SP_GEO_HH_20241123120750 (Nov 23)
  - T2: CAPELLA_C13_SP_GEO_HH_20241126110432 (Nov 26)

The UNet architecture is preserved for future training; the __main__ block
currently computes a normalized log-ratio as a baseline change map.

Usage:
    cd data-fusion-image-processing
    python src/change_detection_baseline.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


# ==========================================
# 1. Módulo de Atenção (Combinação do Siamese_AUNet e SRASNet)
# ==========================================
class StatisticRatioAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, in_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, feat_t1, feat_t2):
        concat_feat = torch.cat([feat_t1, feat_t2], dim=1)
        ratio_diff = torch.abs(feat_t1 - feat_t2)
        c_weight = self.channel_att(concat_feat)
        feat_weighted = ratio_diff * c_weight
        max_pool, _ = torch.max(feat_weighted, dim=1, keepdim=True)
        avg_pool = torch.mean(feat_weighted, dim=1, keepdim=True)
        s_weight = self.spatial_att(torch.cat([max_pool, avg_pool], dim=1))
        return feat_t2 * s_weight * c_weight


# ==========================================
# 2. Camada KAN-Based CNN Simplificada (ReFUnet)
# ==========================================
class KANLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.base_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.spline_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, padding=1, groups=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels * 2, out_channels, 1)
        )

    def forward(self, x):
        return self.base_conv(x) + self.spline_conv(x)


# ==========================================
# 3. Siamese U-Net Unificada
# ==========================================
class UnifiedSiameseUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True))
        self.bottleneck = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True))
        self.attention1 = StatisticRatioAttention(in_channels=64)
        self.attention2 = StatisticRatioAttention(in_channels=128)
        self.up2 = nn.ConvTranspose2d(256 * 2, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(128 * 2, 128, 3, padding=1), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(64 * 2, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.classifier = KANLayer(64, num_classes)

    def forward_encoder(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(F.max_pool2d(e2, 2))
        return e1, e2, b

    def forward(self, t1, t2):
        e1_t1, e2_t1, b_t1 = self.forward_encoder(t1)
        e1_t2, e2_t2, b_t2 = self.forward_encoder(t2)
        b_fusion = torch.cat([b_t1, b_t2], dim=1)
        skip1 = self.attention1(e1_t1, e1_t2)
        skip2 = self.attention2(e2_t1, e2_t2)
        d2 = self.up2(b_fusion)
        d2 = self.dec2(torch.cat([d2, skip2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, skip1], dim=1))
        out = self.classifier(d1)
        return torch.sigmoid(out)


# ==========================================
# Utilitários
# ==========================================
IMG_SIZE = 256  # tamanho de entrada da rede

def carregar_imagem(path: Path) -> torch.Tensor:
    """Carrega PNG SAR como tensor (1, 1, H, W) normalizado [0,1]."""
    img = Image.open(path).convert("L")           # grayscale
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)


def save_result(t1: torch.Tensor, t2: torch.Tensor,
                change_map: torch.Tensor, name_t1: str, name_t2: str,
                output_path: Path):
    """Plots T1, T2 and change map side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(t1.squeeze().numpy(), cmap="gray")
    axes[0].set_title(f"T1 — {name_t1[:30]}", fontsize=8)
    axes[0].axis("off")

    axes[1].imshow(t2.squeeze().numpy(), cmap="gray")
    axes[1].set_title(f"T2 — {name_t2[:30]}", fontsize=8)
    axes[1].axis("off")

    im = axes[2].imshow(change_map.squeeze().numpy(), cmap="hot", vmin=0, vmax=1)
    axes[2].set_title("Change Map (probability)", fontsize=8)
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle("Log-Ratio Change Detection — SAR Mining", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved to: {output_path}")


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    SUPPORT = Path(__file__).parent.parent.parent / "data-fusion-agent/output/support/mineracao"

    # Same AOI: Newman, Western Australia (lat=-21.8, lon=122.2)
    # Same satellite C13, HH polarization, ~29.3° incidence, ascending
    # T1: 23 Nov 2024 | T2: 26 Nov 2024 (3 days apart)
    path_t1 = SUPPORT / "CAPELLA_C13_SP_GEO_HH_20241123120750_20241123120759.png"
    path_t2 = SUPPORT / "CAPELLA_C13_SP_GEO_HH_20241126110432_20241126110441.png"

    print(f"T1: {path_t1.name}")
    print(f"T2: {path_t2.name}")

    t1 = carregar_imagem(path_t1)
    t2 = carregar_imagem(path_t2)
    print(f"Input shape: {t1.shape}")

    mapa_cpu = torch.abs(torch.log10(t1 + 1e-10) - torch.log10(t2 + 1e-10))
    mascara = (t1 > 0.01) & (t2 > 0.01)
    mapa_valido = mapa_cpu[mascara]
    mapa_cpu = (mapa_cpu - mapa_valido.min()) / (mapa_valido.max() - mapa_valido.min() + 1e-8)
    mapa_cpu = mapa_cpu.clamp(0, 1)
    print(f"Output shape: {mapa_cpu.shape}")
    print(f"Prob. min: {mapa_cpu.min():.4f} | max: {mapa_cpu.max():.4f} | mean: {mapa_cpu.mean():.4f}")

    output_path = Path(__file__).parent.parent / "output/change_detection_newman_wa.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_result(t1, t2, mapa_cpu, path_t1.stem, path_t2.stem, output_path)
