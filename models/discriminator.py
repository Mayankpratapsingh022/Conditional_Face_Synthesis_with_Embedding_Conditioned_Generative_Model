"""
Discriminator module for conditional face synthesis project.

This module defines a projection-based conditional discriminator for GAN training.
The discriminator uses both the input image and a conditioning embedding (e.g., from FaceNet)
to guide real/fake prediction, following the BigGAN projection approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

class ConditionalDiscriminator(nn.Module):
    """
    Projection-based conditional discriminator for GANs.

    This discriminator takes both an image and a conditioning embedding as input,
    and predicts real/fake using a projection mechanism inspired by BigGAN.
    """
    def __init__(self, config: Any):
        """
        Initializes the ConditionalDiscriminator.

        Args:
            config: Configuration object with model and data parameters.
        """
        super().__init__()
        base_ch = config.base_ch // 2  # Start with smaller channels for efficiency

        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(config.channels, base_ch, kernel_size=4, stride=2, padding=1)    # 128→64
        self.conv2 = nn.Conv2d(base_ch, base_ch*2, kernel_size=4, stride=2, padding=1)          # 64→32
        self.bn2   = nn.BatchNorm2d(base_ch*2)
        self.conv3 = nn.Conv2d(base_ch*2, base_ch*4, kernel_size=4, stride=2, padding=1)        # 32→16
        self.bn3   = nn.BatchNorm2d(base_ch*4)
        self.conv4 = nn.Conv2d(base_ch*4, base_ch*8, kernel_size=4, stride=2, padding=1)        # 16→8
        self.bn4   = nn.BatchNorm2d(base_ch*8)
        self.conv5 = nn.Conv2d(base_ch*8, base_ch*16, kernel_size=4, stride=2, padding=1)       # 8→4
        self.bn5   = nn.BatchNorm2d(base_ch*16)

        # Final 4×4 feature map is flattened for dot product
        self.final_linear = nn.Linear(base_ch*16*4*4, 1)

        # Projection for conditioning (dot product with image feature)
        self.embed_proj = nn.Linear(config.embedding_dim, base_ch*16*4*4)

        # Initialize weights
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the conditional discriminator.

        Args:
            x (torch.Tensor): Input image tensor of shape [B, C, 128, 128].
            emb (torch.Tensor): Conditioning embedding tensor of shape [B, embedding_dim].

        Returns:
            torch.Tensor: Real/fake logits, conditioned on the embedding [B, 1].
        """
        B = x.size(0)

        out = F.leaky_relu(self.conv1(x), 0.2)
        out = F.leaky_relu(self.bn2(self.conv2(out)), 0.2)
        out = F.leaky_relu(self.bn3(self.conv3(out)), 0.2)
        out = F.leaky_relu(self.bn4(self.conv4(out)), 0.2)
        out = F.leaky_relu(self.bn5(self.conv5(out)), 0.2)

        out_flat = out.view(B, -1)                        # [B, C]
        logits_real_fake = self.final_linear(out_flat)     # [B, 1]

        # Conditional projection (dot product with projected embedding)
        proj = torch.sum(out_flat * self.embed_proj(emb), dim=1, keepdim=True)  # [B, 1]

        return logits_real_fake + proj

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """
        Initializes weights for Conv2d and Linear layers using normal distribution.
        """
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias) 