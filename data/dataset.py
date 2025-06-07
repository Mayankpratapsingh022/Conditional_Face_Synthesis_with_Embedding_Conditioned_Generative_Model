"""
Dataset module for conditional face synthesis project.

This module provides a dataset utility class for downloading, preparing, and loading
face image data for training and evaluation. It supports automatic download from
Hugging Face Hub and returns a PyTorch DataLoader with appropriate transforms.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from huggingface_hub import hf_hub_download
import zipfile
from typing import Any

class FaceDataset:
    """
    Utility class for managing the face image dataset.

    Handles downloading from Hugging Face Hub, extraction, and creation of
    a PyTorch DataLoader with standard transforms for GAN training.
    """
    def __init__(self, config: Any):
        """
        Initializes the FaceDataset utility.

        Args:
            config: Configuration object with dataset and transform parameters.
        """
        self.config = config
        self.transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        
    def download_dataset(self) -> None:
        """
        Downloads and extracts the face dataset from Hugging Face Hub.
        The dataset is saved to the path specified in the config.
        """
        zip_path = hf_hub_download(
            repo_id="Mayank022/Cropped_Face_Dataset_128x128",
            filename="output.zip",
            repo_type="dataset",
        )
        os.makedirs(self.config.data_path, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.config.data_path)
        print(f"Dataset extracted to {self.config.data_path}")

    def get_dataloader(self) -> DataLoader:
        """
        Creates and returns a PyTorch DataLoader for the face dataset.

        Returns:
            DataLoader: PyTorch DataLoader for the face image dataset.
        """
        dataset = datasets.ImageFolder(self.config.data_path, transform=self.transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        print(f"Total images in dataset: {len(dataset)}")
        return dataloader 