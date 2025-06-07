"""
Training script for conditional face synthesis project.

This script trains a UNet-style generator and a projection-based conditional discriminator
for face synthesis, conditioned on embeddings from a (possibly fine-tuned) FaceNet encoder.
It supports full experiment tracking with Weights & Biases (wandb), robust checkpointing,
and quantitative evaluation with FID, SSIM, and PSNR metrics.
"""

import os
import torch
import wandb
from datetime import datetime
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import ExponentialLR

from configs.config import Config
from data.dataset import FaceDataset
from models.generator import UNetStyleFusionGenerator
from models.discriminator import ConditionalDiscriminator
from models.facenet import FaceNetWrapper
from utils.losses import GANLosses
from utils.metrics import Metrics
from typing import Any

def train(config: Any) -> None:
    """
    Trains the conditional face synthesis pipeline.

    This function initializes all models, optimizers, losses, metrics, and data loaders.
    It runs the main training loop, logs progress to wandb, saves checkpoints, and
    evaluates the model at each epoch.

    Args:
        config: Configuration object with all hyperparameters and paths.
    """
    # Initialize wandb for experiment tracking
    wandb.login(key=config.wandb_token)
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=config.__dict__,
    )

    # Create checkpoint directory for this run
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = f"checkpoints/{run_id}"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Initialize models
    G = UNetStyleFusionGenerator(config).to(config.device)
    D = ConditionalDiscriminator(config).to(config.device)
    facenet = FaceNetWrapper(config).to(config.device)

    # Initialize optimizers
    optim_G = torch.optim.Adam(
        G.parameters(),
        lr=config.lr_generator * 0.5,
        betas=(0.5, 0.999),
        weight_decay=config.weight_decay,
    )
    optim_D = torch.optim.Adam(
        D.parameters(),
        lr=config.lr_discriminator,
        betas=(0.5, 0.999),
        weight_decay=config.weight_decay,
    )
    optim_f = facenet.get_optimizer(config)

    # Initialize losses and metrics
    losses = GANLosses(config)
    metrics = Metrics(config)

    # Initialize learning rate schedulers
    sched_G = ExponentialLR(optim_G, gamma=0.99)
    sched_D = ExponentialLR(optim_D, gamma=0.99)

    # Initialize dataset and dataloader
    dataset = FaceDataset(config)
    dataset.download_dataset()
    dataloader = dataset.get_dataloader()

    # Main training loop
    for epoch in range(1, config.num_epochs + 1):
        G.train()
        D.train()
        sum_g = 0.0
        sum_d = 0.0

        for batch_idx, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(config.device)

            # Compute embeddings
            emb = facenet((real_imgs + 1) * 0.5)
            if not config.finetune_facenet:
                emb = emb.detach()

            # Generate fake images
            fake_imgs = G(emb)
            real_imgs.requires_grad_()

            # Discriminator update
            real_logits = D(real_imgs, emb)
            fake_logits = D(fake_imgs.detach(), emb)
            loss_d, r1 = losses.discriminator_loss(real_logits, fake_logits, real_imgs)

            optim_D.zero_grad()
            loss_d.backward(retain_graph=True)
            optim_D.step()

            # Generator (and optional encoder) update
            fake_logits2 = D(fake_imgs, emb)
            fake_emb = facenet((fake_imgs + 1) * 0.5)
            loss_g, adv_loss, emb_loss, perceptual_loss = losses.generator_loss(
                fake_logits2, fake_emb, emb, fake_imgs, real_imgs
            )

            optim_G.zero_grad()
            if config.finetune_facenet:
                optim_f.zero_grad()

            loss_g.backward()
            optim_G.step()
            if config.finetune_facenet:
                optim_f.step()

            # Update metrics
            metrics.update(fake_imgs, real_imgs)

            sum_g += loss_g.item()
            sum_d += loss_d.item()

            # Log sample images to wandb (first batch of each epoch)
            if batch_idx == 0:
                combined = torch.cat([real_imgs[:8], fake_imgs[:8]], dim=0)
                grid = make_grid(combined, nrow=8)
                wandb.log({
                    f"Samples/Epoch_{epoch}": wandb.Image(grid, caption="Top: Real • Bottom: Fake")
                }, step=epoch)

        # Epoch-end metrics
        avg_g = sum_g / len(dataloader)
        avg_d = sum_d / len(dataloader)
        metric_values = metrics.compute()

        # Print summary for this epoch
        print(
            f"[{epoch:03d}/{config.num_epochs:03d}]  "
            f"G: {avg_g:.4f}  D: {avg_d:.4f}  "
            f"Adv: {adv_loss.item():.4f}  Emb: {emb_loss.item():.4f}  "
            f"LPIPS: {perceptual_loss.item():.4f}  "
            f"SSIM: {metric_values['ssim']:.4f}  PSNR: {metric_values['psnr']:.2f}  "
            f"FID: {metric_values['fid']:.2f}"
        )

        # Log losses and metrics to wandb
        wandb.log({
            "losses/generator": avg_g,
            "losses/discriminator": avg_d,
            "losses/adv": adv_loss.item(),
            "losses/embedding": emb_loss.item(),
            "losses/lpips": perceptual_loss.item(),
            "losses/r1_penalty": r1.item(),
            "metrics/SSIM": metric_values['ssim'],
            "metrics/PSNR": metric_values['psnr'],
            "metrics/FID": metric_values['fid'],
            "epoch": epoch,
        }, step=epoch)

        # Step learning rate schedulers
        sched_G.step()
        sched_D.step()

        # Save checkpoints at key epochs
        if epoch % 100 == 0 or epoch in [10, 50, config.num_epochs]:
            ckpt_dir_epoch = os.path.join(ckpt_dir, f"epoch_{epoch:03d}")
            os.makedirs(ckpt_dir_epoch, exist_ok=True)
            torch.save(facenet.model.state_dict(), os.path.join(ckpt_dir_epoch, f"facenet_epoch{epoch:03d}.pt"))
            torch.save(G.state_dict(), os.path.join(ckpt_dir_epoch, f"generator_epoch{epoch:03d}.pt"))
            torch.save(D.state_dict(), os.path.join(ckpt_dir_epoch, f"discriminator_epoch{epoch:03d}.pt"))
            print(f"Saved checkpoints at epoch {epoch:03d} → {ckpt_dir_epoch}")

    # Save final models
    torch.save(G.state_dict(), os.path.join(ckpt_dir, "generator_final.pt"))
    torch.save(D.state_dict(), os.path.join(ckpt_dir, "discriminator_final.pt"))
    torch.save(facenet.model.state_dict(), os.path.join(ckpt_dir, "facenet_final.pt"))

if __name__ == "__main__":
    config = Config()
    train(config) 