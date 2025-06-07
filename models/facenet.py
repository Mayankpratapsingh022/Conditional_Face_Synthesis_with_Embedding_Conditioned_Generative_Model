"""
FaceNet encoder module for conditional face synthesis project.

This module provides a wrapper around the InceptionResnetV1 FaceNet model from facenet_pytorch.
It supports both fixed and fine-tuned usage for embedding extraction in GAN training.
"""

import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
from typing import Any, Optional

class FaceNetWrapper(nn.Module):
    """
    Wrapper for the InceptionResnetV1 FaceNet model.

    This class supports both inference-only and fine-tuning modes, depending on the configuration.
    It provides a method to obtain an optimizer for fine-tuning if enabled.
    """
    def __init__(self, config: Any):
        """
        Initializes the FaceNetWrapper.

        Args:
            config: Configuration object with device and fine-tuning settings.
        """
        super().__init__()
        self.model = InceptionResnetV1(pretrained='vggface2').to(config.device)
        
        if not config.finetune_facenet:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
        else:
            self.model.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for FaceNet embedding extraction.

        Args:
            x (torch.Tensor): Input image tensor of shape [B, 3, H, W], normalized to [0, 1].
        Returns:
            torch.Tensor: Face embeddings of shape [B, 512].
        """
        return self.model(x)

    def get_optimizer(self, config: Any) -> Optional[torch.optim.Optimizer]:
        """
        Returns an optimizer for FaceNet parameters if fine-tuning is enabled.

        Args:
            config: Configuration object with optimizer settings.
        Returns:
            torch.optim.Optimizer or None: Optimizer if fine-tuning, else None.
        """
        if config.finetune_facenet:
            return torch.optim.Adam(
                self.model.parameters(),
                lr=config.lr_finetune_facenet,
                weight_decay=config.weight_decay,
            )
        return None 