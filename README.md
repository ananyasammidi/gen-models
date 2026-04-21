# Real-Image Editing with Text Prompts

CS 5788 Project  
Authors: Amanda Lu, Tiffany Yu, Ananya Sammidi  

---

## Overview

This project explores **real-image editing guided by text prompts**, focusing on producing realistic edits while preserving the structure and identity of the original image. While text-to-image generation has become highly capable, editing existing real images remains challenging—often leading to artifacts, structural changes, or loss of detail.

Our goal is to build a pipeline that:

- Preserves unedited regions of an image  
- Applies accurate, prompt-aligned edits  
- Maintains realism and structural consistency  

---

## Approach

We combine and extend recent diffusion-based image editing techniques.

### 1. DDIM Inversion + Null-Text Optimization
We invert real images into the diffusion latent space using DDIM inversion. Null-text optimization improves reconstruction fidelity by refining unconditional embeddings.

### 2. Prompt-to-Prompt Editing
We use cross-attention control to modify specific regions of an image based on prompt changes while preserving overall layout.

Key techniques include:
- Word swapping  
- Attention re-weighting  
- Prompt refinement  

### 3. Plug-and-Play Feature Injection
For more complex edits, we inject spatial features and self-attention maps from the source image to preserve structure during generation.

---

## Dataset

We construct a custom dataset of real images including:
- Landscapes  
- People  
- Animals  
- Everyday objects  

All images are:
- Resized to **512×512**
- Paired with:
  - A source prompt (image description)
  - One or more target edit prompts

---

## Evaluation

We evaluate our system using both reconstruction and editing quality metrics:

- **PSNR** – reconstruction fidelity after inversion  
- **SSIM & Perceptual Distance** – structural preservation in unchanged regions  
- **CLIP Similarity** – semantic alignment with target prompts  

---

## Expected Results

- High-quality inversion (PSNR ~30–35 dB)  
- Strong preservation of unchanged regions (SSIM > 0.85)  
- Good semantic alignment with prompts (CLIP similarity ~0.25–0.30)  
- Plug-and-Play improves performance on structurally complex edits  

---

## Project Structure
```
├── example_images/ # Images
└── README.md

```


---

## References

- Imagic: https://arxiv.org/abs/2210.09276  
- Prompt-to-Prompt: https://prompt-to-prompt.github.io/  
- HyperStyle: https://openaccess.thecvf.com/content/CVPR2022/papers/Alaluf_HyperStyle_StyleGAN_Inversion_With_HyperNetworks_for_Real_Image_Editing_CVPR_2022_paper.pdf  

---
