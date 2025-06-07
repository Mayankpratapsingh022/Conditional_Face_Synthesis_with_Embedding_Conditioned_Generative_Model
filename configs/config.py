"""
Configuration module for conditional face synthesis project.

This module defines a dataclass-based configuration object for managing all
hyperparameters, paths, and credentials required for training and evaluation.
It loads environment variables for sensitive information and ensures all
required fields are present.
"""

from dataclasses import dataclass
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

@dataclass
class Config:
    """
    Configuration dataclass for the conditional face synthesis pipeline.

    Attributes:
        data_path (str): Path to the training image data.
        image_size (int): Size (height/width) of input/output images.
        channels (int): Number of image channels (e.g., 3 for RGB).
        embedding_dim (int): Dimensionality of the face embedding vector.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of training epochs.
        lr_generator (float): Learning rate for the generator.
        lr_discriminator (float): Learning rate for the discriminator.
        lr_finetune_facenet (float): Learning rate for FaceNet fine-tuning.
        weight_decay (float): Weight decay for optimizers.
        lambda_emb (float): Weight for embedding loss.
        lambda_lpips (float): Weight for perceptual (LPIPS) loss.
        r1_gamma (float): Weight for R1 gradient penalty.
        finetune_facenet (bool): Whether to fine-tune the FaceNet encoder.
        noise_dim (int): Dimensionality of the generator noise vector.
        base_ch (int): Base channel multiplier for generator/discriminator.
        device (str): Device to use for training ("cuda" or "cpu").
        wandb_project (str): Weights & Biases project name.
        wandb_run_name (str): Weights & Biases run name.
        hf_repo_id (str): Hugging Face Hub repository ID.
        hf_token (Optional[str]): Hugging Face API token (from env).
        wandb_token (Optional[str]): W&B API token (from env).
    """
    # Data settings
    data_path: str = "./data/images"
    image_size: int = 128
    channels: int = 3
    embedding_dim: int = 512

    # Training settings
    batch_size: int = 64
    num_epochs: int = 300

    # Optimizer settings
    lr_generator: float = 2e-4
    lr_discriminator: float = 1e-4
    lr_finetune_facenet: float = 1e-5
    weight_decay: float = 0.0

    # Loss weights
    lambda_emb: float = 1.0
    lambda_lpips: float = 0.8
    r1_gamma: float = 10.0

    # Model architecture
    finetune_facenet: bool = True
    noise_dim: int = 128
    base_ch: int = 128

    # Device selection
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

    # Logging and model hub
    wandb_project: str = "conditional-gan-facenet-exp-2"
    wandb_run_name: str = "unet_conditional_gan_facenet-exp-2"
    hf_repo_id: str = "Mayank022/conditional-gan-facenet-exp-2"

    # API keys (loaded from environment variables)
    hf_token: Optional[str] = os.getenv("HF_TOKEN")
    wandb_token: Optional[str] = os.getenv("WANDB_API_KEY")

    def __post_init__(self):
        """
        Validates that all required API tokens are present after initialization.
        Raises:
            ValueError: If any required token is missing from the environment.
        """
        if not self.hf_token:
            raise ValueError("HF_TOKEN not found in environment variables.")
        if not self.wandb_token:
            raise ValueError("WANDB_API_KEY not found in environment variables.") 