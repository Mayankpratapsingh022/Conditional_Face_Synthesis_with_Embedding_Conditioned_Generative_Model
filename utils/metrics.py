"""
Metrics module for conditional face synthesis project.

This module provides a class encapsulating all key image quality metrics used for
evaluating generative models, including FID, SSIM, and PSNR. It supports batch updates
and computes/reset metrics for each evaluation phase.
"""

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from typing import Any, Dict

class Metrics:
    """
    Encapsulates image quality metrics for generative model evaluation.

    Tracks FID, SSIM, and PSNR, and provides methods for batch updates and metric computation/reset.
    """
    def __init__(self, config: Any):
        """
        Initializes the Metrics utility.

        Args:
            config: Configuration object with device information.
        """
        self.config = config
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(config.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(config.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(config.device)

    def update(self, fake_imgs: torch.Tensor, real_imgs: torch.Tensor) -> None:
        """
        Updates all metrics with a batch of real and generated images.

        Args:
            fake_imgs (torch.Tensor): Generated images (model output).
            real_imgs (torch.Tensor): Ground truth images.
        """
        # Normalize images to [0, 1]
        real_norm = (real_imgs.clamp(-1, 1) + 1) * 0.5
        fake_norm = (fake_imgs.clamp(-1, 1) + 1) * 0.5
        # Convert to uint8 for FID
        real_u8 = (real_norm * 255).to(torch.uint8)
        fake_u8 = (fake_norm * 255).to(torch.uint8)
        # Update metrics
        self.fid.update(fake_u8, real=False)
        self.fid.update(real_u8, real=True)
        self.ssim.update(fake_norm, real_norm)
        self.psnr.update(fake_norm, real_norm)

    def compute(self) -> Dict[str, float]:
        """
        Computes and returns all tracked metrics, then resets their state.

        Returns:
            Dict[str, float]: Dictionary with FID, SSIM, and PSNR scores.
        """
        metrics = {
            'fid': self.fid.compute().item(),
            'ssim': self.ssim.compute().item(),
            'psnr': self.psnr.compute().item()
        }
        # Reset metrics for next evaluation
        self.fid.reset()
        self.ssim.reset()
        self.psnr.reset()
        return metrics 