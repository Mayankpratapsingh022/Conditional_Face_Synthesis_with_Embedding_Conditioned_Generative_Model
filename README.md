# Conditional Face Synthesis with Embedding-Conditioned Generative Models

This repository provides a modular pipeline for conditional face synthesis using PyTorch. The system generates 128x128 face images conditioned on embeddings from a FaceNet encoder, supporting both fixed and fine-tuned encoder workflows. The codebase is designed for clarity, reproducibility, and ease of extension for research or production use.

---

---

## Project Structure

```
configs/         # Configuration management
  config.py
models/          # Model architectures (generator, discriminator, facenet)
  generator.py
  discriminator.py
  facenet.py
data/            # Dataset utilities
  dataset.py
utils/           # Losses, metrics, and other utilities
  losses.py
  metrics.py
notebooks/       # Example notebooks for training and inference
train.py         # Training script
inference.py     # Inference script
main.py          # CLI entrypoint
requirements.txt # Python dependencies
```

---


## Project Motivation & Pipeline

This project addresses the challenge of generating realistic face images conditioned on high-dimensional embeddings, under strict compute and data constraints. The pipeline is as follows:

- **Face Cropping (Task 1):**
  - Used a Rust/YOLO pipeline to scrape and crop face images from raw data.
  - Resulted in a dataset of **12,000 cropped face images** (128x128), balancing dataset size and diversity as required by the assignment.

- **Conditional Generation (Task 2):**
  - Trained a conditional generative model to map 512-dim face embeddings to images.
  - Used FaceNet (pre-trained on VGGFace2) for embedding extraction.
  - All training, evaluation, and inference code is modular and production-ready.

---

## Observations & Limitations

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

## Architecture

- **Encoder:** [FaceNet (InceptionResnetV1)](https://github.com/timesler/facenet-pytorch) (pre-trained on VGGFace2, optionally fine-tuned)
- **Generator:** UNet-style GAN with skip connections and self-attention
- **Discriminator:** Projection-based conditional discriminator (inspired by BigGAN)
- **Losses:** Adversarial (BCE), embedding (MSE), perceptual (LPIPS), and R1 gradient penalty
- **Metrics:** FID, SSIM, PSNR

---

## Features

- Modular, object-oriented codebase with clear separation of concerns
- Robust configuration management using Python dataclasses and environment variables
- Automatic dataset download and preparation from Hugging Face Hub
- Full experiment tracking and visualization with W&B
- Checkpointing and model sharing via Hugging Face Hub
- Production-ready CLI for training, evaluation, and inference

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

## Model Checkpoints & Sharing

- Model checkpoints are saved during training and can be uploaded to the Hugging Face Hub for sharing and reproducibility.
- See the `upload_checkpoints_to_hf` utility in the codebase for details.


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements, bug fixes, or new features. For major changes, please discuss them in an issue first.

---

## License

This project is licensed under the MIT License. 