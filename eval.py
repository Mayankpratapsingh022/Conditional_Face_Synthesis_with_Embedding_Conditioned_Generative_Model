"""
Evaluation script for conditional face synthesis project.

This script loads trained generator and encoder checkpoints, reconstructs a face from a test image,
saves the generated output, and computes evaluation metrics (SSIM, PSNR, FID).
"""

import os
import torch
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict

from configs.config import Config
from models.generator import UNetStyleFusionGenerator
from models.facenet import FaceNetWrapper
from utils.metrics import Metrics

def evaluate(config: Config, checkpoint_dir: str, test_image_path: str) -> Dict[str, float]:
    """
    Evaluates the trained generator and encoder on a single test image.

    Loads model checkpoints, reconstructs the face, saves the output, and computes metrics.

    Args:
        config (Config): Configuration object.
        checkpoint_dir (str): Directory containing model checkpoints.
        test_image_path (str): Path to the test image for evaluation.
    Returns:
        Dict[str, float]: Dictionary of computed metrics (SSIM, PSNR, FID).
    """
    # Initialize models
    G = UNetStyleFusionGenerator(config).to(config.device)
    facenet = FaceNetWrapper(config).to(config.device)

    # Load checkpoints
    G.load_state_dict(torch.load(os.path.join(checkpoint_dir, "generator_final.pt")))
    facenet.model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "facenet_final.pt")))

    # Set to eval mode
    G.eval()
    facenet.model.eval()

    # Load and preprocess test image
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    img = Image.open(test_image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(config.device)

    # Get embedding and generate reconstruction
    with torch.no_grad():
        emb = facenet((img_tensor + 1) * 0.5)
        fake_img = G(emb)

    # Save reconstruction
    output_path = os.path.join(checkpoint_dir, "reconstruction.png")
    save_image(fake_img, output_path, normalize=True, value_range=(-1, 1))
    print(f"Reconstruction saved to {output_path}")

    # Compute metrics
    metrics = Metrics(config)
    metrics.update(fake_img, img_tensor)
    metric_values = metrics.compute()

    print("\nEvaluation Metrics:")
    print(f"SSIM: {metric_values['ssim']:.4f}")
    print(f"PSNR: {metric_values['psnr']:.2f}")
    print(f"FID: {metric_values['fid']:.2f}")

    return metric_values

if __name__ == "__main__":
    """
    Example usage:
        python eval.py
    Make sure to update 'checkpoint_dir' and 'test_image_path' as needed.
    """
    config = Config()
    checkpoint_dir = "checkpoints/latest"  # Update this path to your checkpoint directory
    test_image_path = "path/to/test/image.jpg"  # Update this path to your test image
    evaluate(config, checkpoint_dir, test_image_path) 