"""
Phase 1: DDIM Inversion + Null-Text Inversion (NTI)
----------------------------------------------------
Standalone, no external ptp_utils dependency.

Takes a real image + source prompt, returns:
  - ddim_latents  : list of latents [x0, x1, ..., xT]  (for P2P in Phase 2)
  - uncond_embeds : list of per-timestep null embeddings (for faithful recon)
  - x_T           : the final noisy latent  (starting point for Phase 2)

Key improvements over the official notebook, taken from both sources:
  - LR decay over timesteps  (official notebook)
  - Adaptive early-stop epsilon  (official notebook)
  - Caches cond noise pred outside inner loop  (official notebook)
  - No ptp_utils / seq_aligner dependency  (this file)
  - Works with SD v1.5 (no broken auth token)  (this file)
  - Full type hints + comments  (this file)

Usage:
    pipe   = load_pipeline()
    result = run_phase1(pipe, "my_photo.jpg", "a cat sitting next to a mirror")
    # result.x_T, result.ddim_latents, result.uncond_embeds, result.image_gt
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional

from diffusers import StableDiffusionPipeline, DDIMScheduler


# ── Constants ─────────────────────────────────────────────────────────────────

NUM_DDIM_STEPS  = 50
GUIDANCE_SCALE  = 7.5
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE           = torch.float32     # float16 OK on GPU; float32 safer for grad


# ── Pipeline ──────────────────────────────────────────────────────────────────

def load_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5") -> StableDiffusionPipeline:
    """Load SD pipeline with a correctly configured DDIM scheduler."""
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=DTYPE)
    pipe.scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,     # matches training; important for inversion math
    )
    pipe = pipe.to(DEVICE)
    pipe.enable_attention_slicing()
    return pipe


# ── Image helpers ─────────────────────────────────────────────────────────────

def load_512(image_path: str, left: int = 0, right: int = 0,
             top: int = 0, bottom: int = 0) -> np.ndarray:
    """
    Load and centre-crop an image to square, then resize to 512x512.
    Offsets (left, right, top, bottom) let you trim edges before cropping,
    useful when the subject is off-centre.
    """
    image = np.array(Image.open(image_path).convert("RGB"))
    h, w, _ = image.shape
    left   = min(left,  w - 1)
    right  = min(right, w - left - 1)
    top    = min(top,   h - 1)
    bottom = min(bottom, h - top - 1)
    image  = image[top: h - bottom or h, left: w - right or w]
    h, w, _ = image.shape
    # centre-crop to square
    if h < w:
        offset = (w - h) // 2
        image  = image[:, offset: offset + h]
    elif w < h:
        offset = (h - w) // 2
        image  = image[offset: offset + w, :]
    return np.array(Image.fromarray(image).resize((512, 512)))


@torch.no_grad()
def image2latent(pipe: StableDiffusionPipeline, image: np.ndarray) -> torch.Tensor:
    """Encode a numpy HWC uint8 image to a VAE latent (scaled by 0.18215)."""
    tensor = torch.from_numpy(image).float() / 127.5 - 1.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    latent = pipe.vae.encode(tensor).latent_dist.mean
    return latent * 0.18215


@torch.no_grad()
def latent2image(pipe: StableDiffusionPipeline,
                 latent: torch.Tensor) -> np.ndarray:
    """Decode a latent to a HWC uint8 numpy array."""
    latent = latent.detach() / 0.18215
    image  = pipe.vae.decode(latent).sample
    image  = (image / 2 + 0.5).clamp(0, 1)
    image  = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    return (image * 255).astype(np.uint8)


# ── Text encoding ─────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_prompt(pipe: StableDiffusionPipeline, prompt: str) -> torch.Tensor:
    """Tokenise and encode one prompt to a (1, 77, 768) text embedding."""
    ids = pipe.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(DEVICE)
    return pipe.text_encoder(ids)[0]


@torch.no_grad()
def get_context(pipe: StableDiffusionPipeline,
                prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (uncond_emb, cond_emb) for classifier-free guidance."""
    uncond = encode_prompt(pipe, "")
    cond   = encode_prompt(pipe, prompt)
    return uncond, cond


# ── DDIM step primitives ──────────────────────────────────────────────────────

def _prev_step(scheduler: DDIMScheduler,
               noise_pred: torch.Tensor,
               t: int,
               sample: torch.Tensor) -> torch.Tensor:
    """
    Standard DDIM denoising step: x_t -> x_{t-1} (toward clean image).
    Used during NTI optimisation and reconstruction.
    """
    prev_t       = t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    alpha_t      = scheduler.alphas_cumprod[t]
    alpha_t_prev = scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else scheduler.final_alpha_cumprod
    pred_x0      = (sample - (1 - alpha_t) ** 0.5 * noise_pred) / alpha_t ** 0.5
    pred_dir     = (1 - alpha_t_prev) ** 0.5 * noise_pred
    return alpha_t_prev ** 0.5 * pred_x0 + pred_dir


def _next_step(scheduler: DDIMScheduler,
               noise_pred: torch.Tensor,
               t: int,
               sample: torch.Tensor) -> torch.Tensor:
    """
    DDIM inversion step: x_t -> x_{t+1} (toward noise).
    Uses the shifted timestep trick to stay within [0, 999].
    """
    t_cur        = min(t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps, 999)
    alpha_t      = scheduler.alphas_cumprod[t_cur] if t_cur >= 0 else scheduler.final_alpha_cumprod
    alpha_t_next = scheduler.alphas_cumprod[t]
    pred_x0      = (sample - (1 - alpha_t) ** 0.5 * noise_pred) / alpha_t ** 0.5
    pred_dir     = (1 - alpha_t_next) ** 0.5 * noise_pred
    return alpha_t_next ** 0.5 * pred_x0 + pred_dir


@torch.no_grad()
def _noise_pred_single(pipe: StableDiffusionPipeline,
                       latent: torch.Tensor,
                       t: int,
                       emb: torch.Tensor) -> torch.Tensor:
    """UNet forward pass for a single embedding (no CFG batching)."""
    return pipe.unet(latent, t, encoder_hidden_states=emb)["sample"]


# ── Phase 1a: DDIM Inversion ──────────────────────────────────────────────────

@torch.no_grad()
def ddim_inversion(pipe: StableDiffusionPipeline,
                   latent: torch.Tensor,
                   cond_emb: torch.Tensor) -> list[torch.Tensor]:
    """
    Invert a clean latent x0 -> xT by running the diffusion process forward.

    Uses guidance=1 (conditional only, no CFG). This is correct for inversion
    and matches the official implementation.

    Returns:
        all_latents : list of length NUM_DDIM_STEPS+1
                      index 0 = x0 (clean image), index -1 = xT (noise)
    """
    pipe.scheduler.set_timesteps(NUM_DDIM_STEPS)
    all_latents = [latent.clone()]
    xt = latent.clone()

    for i in tqdm(range(NUM_DDIM_STEPS), desc="DDIM inversion"):
        t          = pipe.scheduler.timesteps[NUM_DDIM_STEPS - i - 1]
        noise_pred = _noise_pred_single(pipe, xt, t, cond_emb)
        xt         = _next_step(pipe.scheduler, noise_pred, t, xt)
        all_latents.append(xt.clone())

    return all_latents  # [x0, x1, ..., xT]


# ── Phase 1b: Null-Text Inversion ─────────────────────────────────────────────

def null_text_optimization(
    pipe:            StableDiffusionPipeline,
    ddim_latents:    list[torch.Tensor],
    cond_emb:        torch.Tensor,
    uncond_emb:      torch.Tensor,
    num_inner_steps: int   = 10,
    early_stop_eps:  float = 1e-5,
    lr:              float = 1e-2,
) -> list[torch.Tensor]:
    """
    Optimise per-timestep null-text embeddings so that CFG denoising
    exactly retraces the DDIM inversion trajectory.

    Three improvements from the official notebook baked in:
      1. LR decays linearly over timesteps  -> later steps use smaller lr
      2. Adaptive epsilon  -> tolerance relaxes at later, harder timesteps
      3. Conditional noise is cached outside the inner loop  -> fewer UNet calls

    Args:
        pipe            : loaded SD pipeline
        ddim_latents    : output of ddim_inversion(), list [x0...xT]
        cond_emb        : conditional text embedding (source prompt)
        uncond_emb      : starting unconditional embedding (empty string)
        num_inner_steps : max Adam steps per timestep
        early_stop_eps  : base convergence threshold (adaptive)
        lr              : base learning rate (decays over timesteps)

    Returns:
        uncond_embeds_list : list of NUM_DDIM_STEPS tensors, one per timestep
                             pass directly to Phase 2 denoising loop
    """
    pipe.scheduler.set_timesteps(NUM_DDIM_STEPS)

    uncond_emb         = uncond_emb.clone().detach()
    uncond_embeds_list = []
    latent_cur         = ddim_latents[-1]   # start from xT, denoise toward x0

    bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS, desc="NTI optimisation")

    for i in range(NUM_DDIM_STEPS):
        t           = pipe.scheduler.timesteps[i]
        latent_prev = ddim_latents[len(ddim_latents) - i - 2]  # target x_{t-1}

        # Cache conditional noise pred — it doesn't change during inner loop
        # because only uncond_emb has gradients. Saves num_inner_steps UNet calls.
        with torch.no_grad():
            noise_pred_cond = _noise_pred_single(pipe, latent_cur, t, cond_emb)

        # LR decay: starts at lr, approaches 0 as i -> 100
        uncond_emb = uncond_emb.clone().detach().requires_grad_(True)
        optimizer  = torch.optim.Adam([uncond_emb], lr=lr * (1.0 - i / 100.0))

        j = 0
        for j in range(num_inner_steps):
            noise_pred_uncond = _noise_pred_single(pipe, latent_cur, t, uncond_emb)

            # CFG: combine unconditional and conditional predictions
            noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)

            # Predict what x_{t-1} would be with this null embedding
            latent_prev_rec = _prev_step(pipe.scheduler, noise_pred, t, latent_cur)

            # Minimise distance to the actual inversion trajectory anchor
            loss = F.mse_loss(latent_prev_rec, latent_prev)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bar.update()

            # Adaptive early stop: tolerance relaxes slightly at later timesteps
            if loss.item() < early_stop_eps + i * 2e-5:
                break

        # Fill progress bar for any skipped inner steps
        for _ in range(j + 1, num_inner_steps):
            bar.update()

        uncond_embeds_list.append(uncond_emb[:1].detach().clone())

        # Advance latent_cur using the now-optimised null embedding
        with torch.no_grad():
            context    = torch.cat([uncond_emb.detach(), cond_emb])
            latent_in  = torch.cat([latent_cur] * 2)
            noise_both = pipe.unet(latent_in, t, encoder_hidden_states=context)["sample"]
            nu, nc     = noise_both.chunk(2)
            noise_cfg  = nu + GUIDANCE_SCALE * (nc - nu)
            latent_cur = _prev_step(pipe.scheduler, noise_cfg, t, latent_cur)

        # Warm-start: initialise next timestep from this optimised embedding
        uncond_emb = uncond_emb.detach()

    bar.close()
    return uncond_embeds_list


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class Phase1Result:
    image_gt:      np.ndarray           # original loaded image  (HWC uint8)
    image_rec:     np.ndarray           # VAE round-trip (sanity check)
    x_T:           torch.Tensor         # final noisy latent  -> Phase 2 start
    ddim_latents:  list[torch.Tensor]   # full trajectory [x0 ... xT]
    uncond_embeds: list[torch.Tensor]   # per-timestep null embeddings -> Phase 2
    cond_emb:      torch.Tensor         # source prompt embedding -> Phase 2


# ── Top-level entry point ─────────────────────────────────────────────────────

def run_phase1(
    pipe:            StableDiffusionPipeline,
    image_path:      str,
    prompt:          str,
    offsets:         tuple[int, int, int, int] = (0, 0, 0, 0),
    num_inner_steps: int   = 10,
    early_stop_eps:  float = 1e-5,
    verbose:         bool  = True,
) -> Phase1Result:
    """
    Full Phase 1 pipeline. Call this, then pass the result to Phase 2.

    Args:
        pipe            : loaded SD pipeline from load_pipeline()
        image_path      : path to the source image
        prompt          : text description of the source image
        offsets         : (left, right, top, bottom) pixels to trim before crop
        num_inner_steps : NTI Adam steps per timestep (10 is standard)
        early_stop_eps  : NTI base convergence threshold
        verbose         : print step labels

    Returns:
        Phase1Result — pass .x_T, .uncond_embeds, .cond_emb to Phase 2
    """
    if verbose:
        print(f"[Phase 1] Loading: {image_path}")
    image_gt  = load_512(image_path, *offsets)
    latent_x0 = image2latent(pipe, image_gt)
    image_rec = latent2image(pipe, latent_x0)   # VAE sanity check

    if verbose:
        print(f"[Phase 1] Encoding prompt: '{prompt}'")
    uncond_emb, cond_emb = get_context(pipe, prompt)

    if verbose:
        print("[Phase 1] Running DDIM inversion  (x0 -> xT)...")
    ddim_latents = ddim_inversion(pipe, latent_x0, cond_emb)

    if verbose:
        print("[Phase 1] Running Null-Text Inversion optimisation...")
    uncond_embeds = null_text_optimization(
        pipe, ddim_latents, cond_emb, uncond_emb,
        num_inner_steps=num_inner_steps,
        early_stop_eps=early_stop_eps,
    )

    if verbose:
        print("[Phase 1] Done. Pass result.x_T + result.uncond_embeds to Phase 2.")

    return Phase1Result(
        image_gt      = image_gt,
        image_rec     = image_rec,
        x_T           = ddim_latents[-1],
        ddim_latents  = ddim_latents,
        uncond_embeds = uncond_embeds,
        cond_emb      = cond_emb,
    )


# ── Reconstruction check ──────────────────────────────────────────────────────

@torch.no_grad()
def reconstruct(pipe: StableDiffusionPipeline,
                result: Phase1Result) -> np.ndarray:
    """
    Denoise from x_T back to x_0 using the optimised null embeddings.
    The output should look very close to result.image_gt.
    Run this before Phase 2 to confirm Phase 1 worked correctly.
    """
    pipe.scheduler.set_timesteps(NUM_DDIM_STEPS)
    latent = result.x_T.clone()

    for i, t in enumerate(tqdm(pipe.scheduler.timesteps, desc="Reconstruction check")):
        uncond     = result.uncond_embeds[i].expand(*result.cond_emb.shape)
        context    = torch.cat([uncond, result.cond_emb])
        latent_in  = torch.cat([latent] * 2)
        noise_both = pipe.unet(latent_in, t, encoder_hidden_states=context)["sample"]
        nu, nc     = noise_both.chunk(2)
        noise_cfg  = nu + GUIDANCE_SCALE * (nc - nu)
        latent     = _prev_step(pipe.scheduler, noise_cfg, t, latent)

    return latent2image(pipe, latent)


# ── Example usage ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipe   = load_pipeline()
    result = run_phase1(
        pipe,
        image_path = "example_images/gnochi_mirror.jpeg",
        prompt     = "a cat sitting next to a mirror",
        offsets    = (0, 0, 200, 0),    # trim 200px from top, matches the notebook
    )

    # Sanity check: should look very close to the original
    recon = reconstruct(pipe, result)
    Image.fromarray(recon).save("reconstruction_check.png")
    print("Saved reconstruction_check.png")

    # What to pass to Phase 2:
    #   result.x_T            -> starting noisy latent
    #   result.uncond_embeds  -> per-step null embeddings for CFG
    #   result.cond_emb       -> source prompt embedding
    #   result.ddim_latents   -> full trajectory (some P2P variants need this)
