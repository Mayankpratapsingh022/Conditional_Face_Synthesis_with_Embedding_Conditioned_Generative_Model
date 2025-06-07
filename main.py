"""
Main entrypoint for the conditional face synthesis project.

This script provides a unified CLI for training, evaluation, and inference using the
conditional face synthesis pipeline. It parses command-line arguments and dispatches
to the appropriate module based on the selected mode.
"""

import argparse
import os
from configs.config import Config

def main() -> None:
    """
    Main CLI entrypoint for the conditional face synthesis pipeline.

    Parses command-line arguments and runs the selected mode:
        - train:     Launches the training pipeline.
        - eval:      Runs evaluation on a given input image.
        - inference: Generates a face from an input image or random embedding.
    """
    parser = argparse.ArgumentParser(description="Conditional Face Synthesis")
    parser.add_argument(
        "--mode", type=str, required=True, choices=["train", "eval", "inference"],
        help="Mode to run the model in: 'train', 'eval', or 'inference'"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints/latest",
        help="Directory containing model checkpoints"
    )
    parser.add_argument(
        "--input_image", type=str, default=None,
        help="Path to input image for reconstruction or embedding (required for eval/inference)"
    )
    parser.add_argument(
        "--output_path", type=str, default="output.png",
        help="Path to save the generated/reconstructed image (inference mode)"
    )

    args = parser.parse_args()
    config = Config()

    if args.mode == "train":
        from train import train
        train(config)
    elif args.mode == "eval":
        if not args.input_image:
            raise ValueError("--input_image is required for evaluation mode")
        from eval import evaluate
        evaluate(config, args.checkpoint_dir, args.input_image)
    elif args.mode == "inference":
        from inference import generate_face
        generate_face(
            config,
            args.checkpoint_dir,
            args.input_image,
            args.output_path
        )

if __name__ == "__main__":
    main() 