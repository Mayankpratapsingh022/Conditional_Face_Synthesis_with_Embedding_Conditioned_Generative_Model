"""
Generator module for conditional face synthesis project.

This module defines a UNet-style generator with skip connections and self-attention,
capable of synthesizing 128x128 face images conditioned on input embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional

class ResBlockUp(nn.Module):
    """
    Residual upsampling block for the generator.
    Performs upsampling and adds a residual connection.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.res = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x) + self.res(x)

class SelfAttention(nn.Module):
    """
    Self-attention module for spatial feature refinement in the generator.
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key   = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        attn = torch.bmm(q, k).softmax(dim=-1)
        v = self.value(x).view(B, -1, H * W)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x

class UNetStyleFusionGenerator(nn.Module):
    """
    UNet-style generator with embedding fusion and self-attention.

    Synthesizes 128x128 face images conditioned on input embeddings and noise.
    Embeddings are injected at multiple scales via skip connections.
    """
    def __init__(self, config: Any):
        """
        Initializes the UNetStyleFusionGenerator.

        Args:
            config: Configuration object with model and data parameters.
        """
        super().__init__()
        self.noise_dim = config.noise_dim
        self.base_ch = config.base_ch

        # Project embedding + noise → 4×4 feature map
        self.fc = nn.Linear(config.embedding_dim + config.noise_dim, config.base_ch * 8 * 4 * 4)

        # Project embeddings for skip injection
        self.emb_proj = nn.ModuleList([
            nn.Linear(config.embedding_dim, config.base_ch * 8 * 8 * 8),       # for 8×8, 1024 channels
            nn.Linear(config.embedding_dim, config.base_ch * 4 * 16 * 16)      # for 16×16, 512 channels
        ])

        # Decoder blocks
        self.up1 = ResBlockUp(config.base_ch * 8, config.base_ch * 8)     # 4×4 → 8×8
        self.up2 = ResBlockUp(config.base_ch * 8, config.base_ch * 4)     # 8×8 → 16×16
        self.attn = SelfAttention(config.base_ch * 4)                     # 16×16
        self.up3 = ResBlockUp(config.base_ch * 4, config.base_ch * 2)     # 16×16 → 32×32
        self.up4 = ResBlockUp(config.base_ch * 2, config.base_ch)         # 32×32 → 64×64
        self.up5 = nn.Sequential(                                         # 64×64 → 128×128
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(config.base_ch, config.channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

        self.apply(self._init_weights)

    def forward(self, emb: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass for the generator.

        Args:
            emb (torch.Tensor): Face embedding tensor of shape [B, embedding_dim].
            batch_size (Optional[int]): Optional batch size override.
        Returns:
            torch.Tensor: Generated image tensor of shape [B, channels, 128, 128].
        """
        B = emb.size(0)
        if batch_size is None:
            batch_size = B

        # Concatenate embedding and noise
        z = torch.randn(batch_size, self.noise_dim, device=emb.device)
        x = torch.cat([emb, z], dim=1)
        x = self.fc(x).view(B, self.base_ch * 8, 4, 4)

        # Embedding projections
        emb8 = self.emb_proj[0](emb).view(B, self.base_ch * 8, 8, 8)
        emb16 = self.emb_proj[1](emb).view(B, self.base_ch * 4, 16, 16)

        # Decode with skip injections
        d1 = self.up1(x)           # → [B, 1024, 8, 8]
        d1 = d1 + emb8             # inject @8
        d2 = self.up2(d1)          # → [B, 512, 16, 16]
        d2 = self.attn(d2 + emb16) # inject @16 + attention
        d3 = self.up3(d2)          # → [B, 256, 32, 32]
        d4 = self.up4(d3)          # → [B, 128, 64, 64]
        out = self.up5(d4)         # → [B, channels, 128, 128]

        return out

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """
        Initializes weights for Conv2d and Linear layers using Kaiming normal initialization.
        """
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, a=0.2)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias) 