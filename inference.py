"""
Inference script for conditional face synthesis project.

This script provides a modular interface for generating face images using trained generator and encoder checkpoints.
It supports both image-conditioned and random embedding-based generation, and is designed for production and research use.
"""

import os
import torch
from torchvision.utils import save_image
import torchvision.transforms as transforms
from PIL import Image
from typing import Optional

from configs.config import Config
from models.generator import UNetStyleFusionGenerator
from models.facenet import FaceNetWrapper

def generate_face(
    config: Config,
    checkpoint_dir: str,
    input_image_path: Optional[str] = None,
    output_path: str = "generated_face.png"
) -> None:
    """
    Generates a face image using the trained generator and encoder.

    Supports both image-conditioned and random embedding-based generation.

    Args:
        config (Config): Configuration object.
        checkpoint_dir (str): Directory containing model checkpoints.
        input_image_path (Optional[str]): Path to input image for embedding (if any).
        output_path (str): Path to save the generated image.
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

    with torch.no_grad():
        if input_image_path:
            # Generate from input image embedding
            transform = transforms.Compose([
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ])
            img = Image.open(input_image_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(config.device)
            emb = facenet((img_tensor + 1) * 0.5)
        else:
            # Generate from random embedding
            emb = torch.randn(1, config.embedding_dim).to(config.device)

        # Generate face
        fake_img = G(emb)

    # Save generated image
    save_image(fake_img, output_path, normalize=True, value_range=(-1, 1))
    print(f"Generated face saved to {output_path}")

if __name__ == "__main__":
    """
    Example usage:
        python inference.py
    Update 'checkpoint_dir' and 'input_image_path' as needed.
    """
    config = Config()
    checkpoint_dir = "checkpoints/latest"  # Update this path

    # Example 1: Generate from input image
    input_image_path = "path/to/input/image.jpg"  # Update this path
    generate_face(config, checkpoint_dir, input_image_path, "reconstructed_face.png")

    # Example 2: Generate from random embedding
    generate_face(config, checkpoint_dir, output_path="random_face.png") 