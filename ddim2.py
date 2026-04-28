import torch
import requests
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms as tfms
from diffusers import StableDiffusionPipeline, DDIMScheduler

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_STEPS = 50
GUIDANCE_SCALE = 7.5

# Load the pipeline
def load_pipeline(model_id="runwayml/stable-diffusion-v1-5"):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # float16 for GPU speed
        safety_checker=None
    ).to(DEVICE)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # memory helpers
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    return pipe


# Load images
def load_image(url, size=None):
    response = requests.get(url, timeout=10)  # fixed: was 0.2 (too short)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    if size is not None:
        img = img.resize(size)
    return img


def encode_prompt(pipe, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt):
    """Wrapper to handle the _encode_prompt -> encode_prompt deprecation."""
    try:
        # diffusers >= 0.20
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )
        if do_classifier_free_guidance:
            return torch.cat([negative_prompt_embeds, prompt_embeds])
        return prompt_embeds
    except AttributeError:
        # fallback for older diffusers
        return pipe._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )


def decode_latents(pipe, latents):
    """Wrapper to handle decode_latents deprecation."""
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = pipe.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return pipe.numpy_to_pil(image)


# Sample function (regular DDIM)
@torch.no_grad()
def sample(prompt, pipe, start_step=0, start_latents=None,
           guidance_scale=3.5, num_inference_steps=30,
           num_images_per_prompt=1, do_classifier_free_guidance=True,
           negative_prompt='', device=DEVICE):

    # Encode prompt
    text_embeddings = encode_prompt(
        pipe, prompt, device, num_images_per_prompt,
        do_classifier_free_guidance, negative_prompt
    )

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)
        start_latents *= pipe.scheduler.init_noise_sigma

    latents = start_latents.clone()

    for i in tqdm(range(start_step, num_inference_steps)):

        t = pipe.scheduler.timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        prev_t = max(1, t.item() - (1000 // num_inference_steps))  # t-1
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
        predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred
        latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt

    # Post-processing
    images = decode_latents(pipe, latents)
    return images


## Inversion
@torch.no_grad()
def invert(start_latents, prompt, pipe, guidance_scale=3.5, num_inference_steps=80,
           num_images_per_prompt=1, do_classifier_free_guidance=True,
           negative_prompt='', device=DEVICE):

    # Encode prompt
    text_embeddings = encode_prompt(
        pipe, prompt, device, num_images_per_prompt,
        do_classifier_free_guidance, negative_prompt
    )

    # Latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps
    timesteps = reversed(pipe.scheduler.timesteps)

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000 // num_inference_steps))
        next_t = t
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (1 - alpha_t_next).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)

    return torch.cat(intermediate_latents)


def edit(input_image, input_image_prompt, edit_prompt, pipe, num_steps=100, start_step=30, guidance_scale=3.5):
    input_image = input_image.resize((512, 512))
    with torch.no_grad():
        latent = pipe.vae.encode(
            tfms.functional.to_tensor(input_image)
            .unsqueeze(0)
            .to(DEVICE)
            .to(torch.float16)  # match model dtype
            * 2 - 1
        )
    l = 0.18215 * latent.latent_dist.sample()
    inverted_latents = invert(l, input_image_prompt, pipe, num_inference_steps=num_steps)
    final_im = sample(
        edit_prompt, pipe,
        start_latents=inverted_latents[-(start_step + 1)][None],
        start_step=start_step,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale
    )[0]
    return final_im