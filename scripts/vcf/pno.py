import torch
import numpy as np
from PIL import Image
from itertools import islice
from einops import rearrange
import wandb
from dotenv import load_dotenv
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder, FrozenOpenCLIPEmbedder
from torchvision import transforms

load_dotenv()

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

################### PNO NOISE REGULARIZATION ###################
import torch
import math

def compute_M1(z):  # z: (m, k)
    return torch.norm(z.mean(dim=0))

def compute_M2(z):  # z: (m, k)
    k = z.shape[1]
    cov = z.T @ z / z.shape[0]
    return torch.norm(cov - torch.eye(k, device=z.device))

def p1(M1_val, m, k):
    return max(2 * math.exp(-m * M1_val**2 / (2 * k)), 1)

def p2(M2_val, m, k):
    diff = max(math.sqrt(k + M2_val) - math.sqrt(k), 0)
    C = 4 * math.sqrt(2 / math.pi)  # as per original source
    return max(2 * math.exp(-m * diff**2 / C), 1)

def reg_loss(z, num_permutations=5):
    m, k = z.shape
    log_probs = []

    for _ in range(num_permutations):
        permuted_z = z[torch.randperm(m)]
        M1_val = compute_M1(permuted_z)
        M2_val = compute_M2(permuted_z)
        log_p1 = math.log(p1(M1_val, m, k))
        log_p2 = math.log(p2(M2_val, m, k))
        log_probs.append(log_p1 + log_p2)

    return -torch.mean(torch.tensor(log_probs, device=z.device))

def instantiate_clip_encoders(device="cuda"):
    """Instantiate and return OpenCLIP image and text encoders (ViT-H-14, laion2b_s32b_b79k)."""
    image_encoder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-H-14", version="laion2b_s32b_b79k", device=device, preproject=False
    ).to(device)
    # text_encoder = FrozenOpenCLIPEmbedder(
    #     arch="ViT-H-14", version="laion2b_s32b_b79k", device=device
    # ).to(device)
    image_encoder.eval()
    # text_encoder.eval()
    return image_encoder #, text_encoder

def extract_image_features(image_encoder, image_path, device="cuda"):
    """Extract image features using the provided encoder."""
    img = Image.open(image_path).convert("RGB")
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        img_features = image_encoder(img_tensor)
    return img_features

def extract_text_features(text_encoder, text, device="cuda"):
    """Extract text features using the provided encoder."""
    if isinstance(text, str):
        text = [text]
    with torch.no_grad():
        text_features = text_encoder(text)
    return text_features

def optimize_pno(
        model: torch.nn.Module,
        sampler: DDIMSampler,
        clip_model_img: FrozenOpenCLIPImageEmbedder,
        text_prompt: str,
        image_feat: torch.Tensor,
        steps=10,
        ddim_steps=50,
        lr_prompt=1e-3,
        lr_noise=1e-3,
        unconditional_guidance_scale=9,
        unconditional_conditioning=None,
        device="cuda",
        fusion_fn=None,
        lambda_weight=1,
        opt=None,
    ):
    """Optimize both prompt embeddings and noise for better text-to-image generation with reference image."""
    model.eval()

    image_feat_norm = torch.nn.functional.normalize(image_feat, dim=-1)

    # Step 1: Get text embeddings
    with torch.no_grad():
        text_tokens = model.get_learned_conditioning([text_prompt])  # [1, 77, 1024]
        print(f"Text tokens shape: {text_tokens.shape}")

    # Step 2: Fuse image and text features if image_feat is available
    # NOTE: for now no fusion_fn
    fused_prompt = text_tokens.clone()
    fused_prompt = fused_prompt.detach().clone().requires_grad_(True)
    # print if this requires grad
    print(f"Fused prompt requires grad: {fused_prompt.requires_grad}")

    # Step 3: Initial noise x_T
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
    initial_noise = torch.randn([1, *shape], device=device)
    # x_T = initial_noise.clone().detach().requires_grad_(True)
    # NOTE: for now no noise optimization
    x_T = initial_noise.clone().detach()
    x_T.requires_grad = False
    # print if this requires grad
    print(f"Initial noise x_T requires grad: {x_T.requires_grad}")

    # Step 4: Optimize prompt and noise
    optimizer = torch.optim.Adam([
        {'params': fused_prompt, 'lr': lr_prompt},
        # {'params': x_T, 'lr': lr_noise}
    ])

    print(f"Starting PNO optimization with {steps} steps...")
    for step in range(steps):
        print(f"Step {step+1}/{steps}")
        # fused prompt
        print(f"Fused prompt: {fused_prompt}")
        # Prepare conditioning
        cond = fused_prompt

        # Track intermediate latents for noise trajectory regularization
        intermediate_latents = []

        # Sampling loop to collect all intermediate latents
        def store_intermediate_latents(pred_x0, i):
            intermediate_latents.append(pred_x0)

        sample, _ = sampler.sample(
            S=ddim_steps,
            conditioning=cond,
            batch_size=1,
            shape=shape,
            verbose=False,
            x_T=x_T,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            img_callback=store_intermediate_latents,  # Collect intermediate latents
            enable_grad=True,
        )
        # print intermediate latents shape
        # print(f"Intermediate latents shape: {[lat.shape for lat in intermediate_latents]}")

        # Decode to image space
        x_sample = model.decode_first_stage(sample, enable_grad=True)

        # Resize for CLIP
        x_clip = torch.nn.functional.interpolate(x_sample, size=(224, 224), mode='bilinear')
        # Get CLIP image embedding
        # with torch.no_grad():
        gen_image_clip = clip_model_img(x_clip).float()
        gen_image_clip = torch.nn.functional.normalize(gen_image_clip, dim=-1)

        # Compute loss
        loss = 0
        # Add similarity between x_clip and image_feat to the loss
        sim = (gen_image_clip * image_feat_norm).sum(dim=1)
        clip_loss = - sim.mean()
        loss += clip_loss

        # Add noise trajectory regularization using all intermediate latents
        trajectory = torch.stack(intermediate_latents, dim=0)  # Shape: [ddim_steps, batch_size, latent_dim]
        r_loss = reg_loss(trajectory.view(-1, trajectory.shape[-1]))  # Flatten trajectory for reg_loss
        # loss += r_loss * lambda_weight

        print("Loss requires grad:", loss.requires_grad)
        print("Clip loss requires grad:", clip_loss.requires_grad)
        print("Reg loss requires grad:", r_loss.requires_grad)
        loss = loss.requires_grad_()
        print(f"Loss: {loss}, Clip Loss: {clip_loss}, Reg Loss: {r_loss}")
        # print requires grad
        print(f"Loss requires grad: {loss.requires_grad}")

        print("*"*30)
        # Print current values of prompt and x_T
        print(f"Current prompt: {fused_prompt[0, :10]}")  # Print the first 10 values of the prompt embedding

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        print("fused_prompt grad:", fused_prompt.grad.norm())
        # print("x_T grad:", x_T.grad.norm())
        optimizer.step()

        # Print updated values of prompt and x_T
        print(f"Updated prompt: {fused_prompt[0, :10]}")
        print("*"*30)


        # Log intermediate results to WandB
        with torch.no_grad():
            x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
            x_sample = 255. * rearrange(x_sample[0].cpu().numpy(), 'c h w -> h w c')
            wandb.log({
                "Intermediate Sample": wandb.Image(x_sample.astype(np.uint8), 
                                                  caption=f"PNO Step {step+1}/{steps}"),
                "Loss": loss.item(),
                "Clip Loss": clip_loss.item(),
                "Reg Loss": r_loss.item(),
            })

        print(f"[PNO Step {step+1}/{steps}] Loss: {loss.item():.4f}, Reg Loss: {r_loss.item():.4f}")
    
    # Clear unused variables and free GPU memory
    del x_sample, x_clip, gen_image_clip, trajectory
    torch.cuda.empty_cache()

    # Final generation with optimized parameters
    final_sample, _ = sampler.sample(
        S=ddim_steps,
        # conditioning={"c_crossattn": [fused_prompt.detach()]},
        conditioning=fused_prompt.detach(),
        batch_size=1,
        shape=shape,
        verbose=False,
        x_T=x_T.detach(),
        unconditional_guidance_scale=unconditional_guidance_scale,
        unconditional_conditioning=unconditional_conditioning
    )
    final_sample = torch.clamp((final_sample + 1.0) / 2.0, min=0.0, max=1.0)

    return final_sample
################### END PNO NOISE REGULARIZATION ###################