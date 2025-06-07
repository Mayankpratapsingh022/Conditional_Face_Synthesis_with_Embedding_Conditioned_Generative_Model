"""
Loss functions module for conditional face synthesis project.

This module provides a class encapsulating all GAN-related losses, including adversarial,
embedding, perceptual (LPIPS), and R1 gradient penalty losses for stable GAN training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lpips import LPIPS
from typing import Any, Tuple

class GANLosses:
    """
    Encapsulates all loss functions used in conditional GAN training.

    Includes adversarial (BCE), embedding (MSE), perceptual (LPIPS), and R1 gradient penalty losses.
    """
    def __init__(self, config: Any):
        """
        Initializes the GANLosses utility.

        Args:
            config: Configuration object with device and loss weight parameters.
        """
        self.config = config
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.perceptual = LPIPS(net="vgg").to(config.device)

    def r1_penalty(self, real_preds: torch.Tensor, real_imgs: torch.Tensor) -> torch.Tensor:
        """
        Computes the R1 gradient penalty on real images.

        Args:
            real_preds (torch.Tensor): Discriminator outputs for real images.
            real_imgs (torch.Tensor): Real image tensors.
        Returns:
            torch.Tensor: Scalar R1 penalty value.
        """
        grad_real = torch.autograd.grad(
            outputs=real_preds.sum(), inputs=real_imgs, create_graph=True
        )[0]
        return grad_real.pow(2).view(real_imgs.size(0), -1).sum(1).mean()

    def discriminator_loss(
        self, real_logits: torch.Tensor, fake_logits: torch.Tensor, real_imgs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the discriminator loss with R1 penalty.

        Args:
            real_logits (torch.Tensor): Discriminator outputs for real images.
            fake_logits (torch.Tensor): Discriminator outputs for fake images.
            real_imgs (torch.Tensor): Real image tensors.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (Total discriminator loss, R1 penalty value)
        """
        loss_d = self.bce(real_logits, torch.ones_like(real_logits)) + \
                 self.bce(fake_logits, torch.zeros_like(fake_logits))
        # Add R1 penalty
        r1 = self.r1_penalty(real_logits, real_imgs)
        loss_d += self.config.r1_gamma * r1
        return loss_d, r1

    def generator_loss(
        self,
        fake_logits: torch.Tensor,
        fake_emb: torch.Tensor,
        real_emb: torch.Tensor,
        fake_imgs: torch.Tensor,
        real_imgs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the generator loss, including adversarial, embedding, and perceptual losses.

        Args:
            fake_logits (torch.Tensor): Discriminator outputs for fake images.
            fake_emb (torch.Tensor): Embeddings of generated images.
            real_emb (torch.Tensor): Embeddings of real images.
            fake_imgs (torch.Tensor): Generated images.
            real_imgs (torch.Tensor): Real images.
        Returns:
            Tuple containing (total generator loss, adversarial loss, embedding loss, perceptual loss).
        """
        adv_loss = self.bce(fake_logits, torch.ones_like(fake_logits))
        emb_loss = self.mse(fake_emb, real_emb)
        perceptual_loss = self.perceptual(fake_imgs, real_imgs).mean()

        loss_g = adv_loss \
                + self.config.lambda_emb * emb_loss \
                + self.config.lambda_lpips * perceptual_loss

        return loss_g, adv_loss, emb_loss, perceptual_loss 