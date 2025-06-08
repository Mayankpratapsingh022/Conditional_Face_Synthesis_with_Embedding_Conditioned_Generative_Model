# Conditional Face Synthesis with Embedding-Conditioned Generative Models

![Pipeline arc](https://github.com/user-attachments/assets/e5d0a5fa-9cf1-40ff-a692-faf24aae579c)

This repository provides code conditional face synthesis. The system generates 128x128 face images conditioned on embeddings from a FaceNet encoder, supporting both fixed and fine-tuned encoder workflows. The codebase is designed for clarity, reproducibility, and ease of extension for research or production use.

---

## Project Structure


```bash
configs/         # Configuration management
  config.py
models/          # Generator, discriminator, and encoder architectures
  generator.py
  discriminator.py
  facenet.py
data/            # Dataset loading and transformation utilities
  dataset.py
utils/           # Loss functions, metrics, and helpers
  losses.py
  metrics.py
notebooks/       # Training and inference Colab notebooks
train.py         # Training pipeline
inference.py     # Inference script
main.py          # CLI entry point
requirements.txt # Python dependencies
```


---


## Project Motivation & Pipeline

This project addresses the challenge of generating realistic face images conditioned on high-dimensional embeddings, under strict compute and data constraints. The pipeline is as follows:

- **Face Cropping (Task 1):**
  - Used a Rust/YOLO pipeline to scrape and crop face images from raw data.
  - Resulted in a dataset of **12,000 cropped face images** (128x128), balancing dataset size and diversity as required, Used the same for training.

- **Conditional Generation (Task 2):**
  - Trained a conditional generative model to map 512-dim face embeddings to images.
  - Used FaceNet (pre-trained on VGGFace2) for embedding extraction.
  - All training, evaluation, and inference code is modular and production-ready.

---

## Codebase, Dataset, and Model Artifacts

| Category         | Description                                          | Link                                                                 |
|------------------|------------------------------------------------------|----------------------------------------------------------------------|
| GitHub Repo      | Full training and inference codebase                | [GitHub Repository](https://github.com/Mayankpratapsingh022/Conditional_Face_Synthesis_with_Embedding_Conditioned_Generative_Model) |
| Dataset          | Cropped faces (128x128) for training                | [Hugging Face Dataset](https://huggingface.co/datasets/Mayank022/Cropped_Face_Dataset_128x128) |
| Trained Model    | Final GAN model with FaceNet encoder                | [Hugging Face Model](https://huggingface.co/Mayank022/facegen-facenet-unet-gan-embedding) |
| Training Notebook| End-to-end model training pipeline in Colab         | [Colab Notebook](https://colab.research.google.com/drive/16vafB_pVNk_QJpquXwxMJXNme3BCGFqS?usp=sharing) |
| Inference Notebook| Generate images from embeddings                    | [Colab Notebook](https://colab.research.google.com/drive/1Y1s7fmyVfT2jnEL9l23jmkhISNYastds?usp=sharing) |

---

## Experimental Summary

Find detailed metrics, loss trends, and inference samples in the full Weights & Biases report:

> [Weights & Biases Report – Training, Metrics, and Sample Outputs](https://api.wandb.ai/links/mayankpratapsingh0022-other/x8zkffzn)

> [Weights & Biases Dashboard](https://wandb.ai/mayankpratapsingh0022-other/conditional-gan-facenet-exp-2/runs/e4895su5?nw=nwusermayankpratapsingh0022)

### Key Experiments:
- Explored both GAN and diffusion-based generative strategies.
- Tested multiple embedding models (ArcFace, FaceNet).
- Settled on a UNet-style GAN with self-attention, projection-based discriminator, and FaceNet encoder.

---

## Observations & Limitations

![Output ](https://github.com/user-attachments/assets/c702b93c-91e0-42f8-ad19-98e5dc379f0b)

The model was developed under a constrained timeline, with the original task suggesting a 1-day implementation. In practice, it took approximately 1.5–2 days, during which multiple architectural variations and hyperparameter combinations were tested. Early-stage experimentation included diffusion-based approaches and different embedding models (ArcFace and FaceNet). After evaluating both performance and training stability, a UNet-style GAN conditioned on FaceNet embeddings was selected as the final architecture.

Despite using a relatively small dataset (~12,000 images) and a 6-hour training cap on a single A100 GPU (Refer Colab Notebook), the model showed promising results in structure and identity retention. However, perceptual quality remains limited in some areas (e.g., eye generation), which can be addressed with larger datasets and architectural enhancements.


- **Image Quality:**
  - The generated faces show basic structure and identity traits, but perceptual quality is limited.
  - A notable issue is the model's inability to generate realistic eyes. This is likely due to many training images having dark or occluded eyes, causing the embedding space to treat these regions as complex or ambiguous.

- **Dataset Constraints:**
  - The dataset was limited to 12k images, as specified in Task 1. Using a larger, more diverse dataset (e.g., 100k+ images) would likely improve generalization and output quality.

- **Training Time:**
  - Training was capped at **6 hours** on an A100 GPU, which limited the model's ability to fully converge and learn fine details.

- **Architectural Trade-offs:**
  - More advanced architectures (e.g., transformers, cross-attention) were not used due to time constraints, but could improve results with more resources.

---

## Experimental Architectures & Rationale

During development, I experimented with several architectures and embedding strategies:

- **Diffusion Models:**
  - ArcFace Embedding + Diffusion
  - FaceNet Embedding + Diffusion
- **GANs (UNet-style):**
  - ArcFace Embedding + UNet GAN
  - FaceNet Embedding + UNet GAN

After initial runs and qualitative evaluation, **FaceNet + UNet-style GAN** was found to be the most stable and performant combination given the constraints. Diffusion models, while promising, required more compute and time to converge.

---

## Final Architecture Details

- **Encoder:** FaceNet (InceptionResnetV1, 512-dim embedding, optionally fine-tuned)
- **Generator:**
  - UNet-style architecture with skip connections and self-attention
  - Embedding and noise are projected and injected at multiple scales in the decoder
  - Decoder upsamples from 4x4 to 128x128 using residual upsampling blocks
- **Discriminator:**
  - Projection-based conditional discriminator (inspired by BigGAN)
  - Uses both image and embedding for real/fake prediction
- **Losses:**
  - Adversarial (BCE), embedding (MSE), perceptual (LPIPS), and R1 gradient penalty
- **Metrics:**
  - FID, SSIM, PSNR (all logged to W&B)

---

## Training Configuration Summary [Same included in W&B Report]

| Category            | Parameter                           | Value                                   |
|---------------------|-------------------------------------|-----------------------------------------|
| Hardware            | GPU                                 | NVIDIA A100                             |
| Training Time       | Duration                            | ~6 hours                                |
| Image               | Resolution                          | 128×128                                 |
|                     | Channels                            | 3 (RGB)                                  |
| Embedding           | Dimension                           | 512 (FaceNet)                            |
| Noise Vector        | Dimension                           | 128                                      |
| Model Depth         | Base Channel Size                   | 128                                      |
| Training            | Batch Size                          | 64                                       |
|                     | Epochs                              | 300                                      |
|                     | Learning Rate (Generator)           | 2e-4                                     |
|                     | Learning Rate (Discriminator)       | 1e-4                                     |
|                     | Learning Rate (FaceNet Fine-tune)   | 1e-5                                     |
|                     | Weight Decay                        | 0.0                                      |
| Loss Weights        | Embedding Loss (λ_emb)              | 1.0                                      |
|                     | Perceptual Loss (λ_LPIPS)           | 0.8                                      |
|                     | R1 Gradient Penalty (γ)             | 10.0                                     |
| Model Settings      | FaceNet Fine-tuning                 | Enabled                                  |
|                     | Device                              | CUDA (if available)                      |

---


## Results & Future Work

- **Current Results:**
  - The model demonstrates basic face structure and some identity preservation, but does not achieve high perceptual quality, especially in regions like the eyes.
  - W&B logs, model checkpoints, and inference notebooks are provided for transparency and reproducibility.

- **Limitations:**
  - Limited dataset size and training time are the primary bottlenecks.
  - The cropping pipeline from Task 1, while efficient, may have introduced artifacts or low-quality samples.

- **Future Improvements:**
  - Increase dataset size and diversity (e.g., 100k+ high-quality faces)
  - Allow for longer training or more advanced architectures (e.g., transformers, cross-attention)
  - Explore data augmentation and regularization strategies
  - Improve eye region synthesis by targeted data curation or loss weighting

---

## Project Overview

- **Goal:** Generate realistic 128x128 face images from face embeddings using a conditional generative model.
- **Zero-Shot Generalization:** The model is evaluated on its ability to generate high-quality faces from unseen embeddings.
- **Experiment Tracking:** All training runs, metrics, and sample generations are logged to Weights & Biases (W&B).
- **Model Sharing:** Trained model checkpoints are uploaded to the Hugging Face Hub for public access and future inference.

---

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/conditional-face-synthesis.git
   cd conditional-face-synthesis
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up environment variables:**
   - Create a `.env` file or export the following variables:
     - `HF_TOKEN` (Hugging Face API token)
     - `WANDB_API_KEY` (Weights & Biases API key)

---

## Usage

### Training
Train the conditional face synthesis model from scratch:
```bash
python main.py --mode train
```

### Evaluation
Evaluate the model on a test image and compute metrics:
```bash
python main.py --mode eval --checkpoint_dir checkpoints/latest --input_image path/to/test/image.jpg
```

### Inference
Generate a face from an input image or a random embedding:
```bash
# From an input image
python main.py --mode inference --checkpoint_dir checkpoints/latest --input_image path/to/input/image.jpg --output_path reconstructed_face.png

# From a random embedding
python main.py --mode inference --checkpoint_dir checkpoints/latest --output_path random_face.png
```

---

## Metrics

- **FID (Frechet Inception Distance):** Measures distributional similarity between generated and real images.
- **SSIM (Structural Similarity Index):** Measures perceptual similarity between images.
- **PSNR (Peak Signal-to-Noise Ratio):** Measures pixel-level similarity.

All metrics are logged to W&B and printed at the end of each evaluation.

---
# Final Remarks
This repository represents a compact but extensible solution for conditional face synthesis under tight computational and data constraints. While the current implementation is limited by dataset scale and training duration, it establishes a strong foundation for future experimentation. With additional compute, higher-quality data, and further architectural exploration, this pipeline can be scaled to generate high-fidelity, identity-consistent face images for a variety of real-world applications.
