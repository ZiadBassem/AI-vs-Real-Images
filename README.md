## Problem Statement

AI image generators like Stable Diffusion and MidJourney can create highly realistic images that often resemble real photos.  
This project builds a **binary classifier** to detect whether an image is **AI-generated** or **real**, addressing needs in content moderation, misinformation detection, and commercial image verification.

## Dataset

The dataset was built from two balanced sources:

- **Real Images (≈470)** → collected from the Pexels API (public stock photography).  
- **AI-Generated Images (≈500)** → created using Stable Diffusion/Craiyon with varied prompts.  

### Preprocessing
- Resized to 224x224 pixels  
- Normalized for model input  
- Split: Train 70% / Validation 20% / Test 10%  

### Augmentation
- Horizontal flips  
- Small rotations (±15°)  
- Brightness/Contrast adjustments (~10%)  
- Gaussian blur (2–3px)

## Model & Training

The model was trained on Roboflow using transfer learning:  

- **Architecture**: Vision Transformer (ViT)  
- **Pretrained on**: ImageNet  
- **Training time**: ~34 minutes on Roboflow GPU  
- **Data split**: 70% train / 20% validation / 10% test  
- **Loss**: Cross Entropy  
- **Optimizer**: Adam (default settings)  

## Results & Evaluation

The Vision Transformer (ViT) model achieved:

- **Accuracy (Validation Top-1)**: ~99% after 12 epochs  

This shows the model performed extremely well in distinguishing AI-generated and real images in the dataset.

### Training Graphs
Roboflow provided the following training curves:

- **Training Loss** (decreased smoothly → model converged well)  
- **Validation Accuracy** (reached ~99% with stability after epoch 6)
