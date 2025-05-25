import argparse, os
import cv2
from datetime import datetime
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from imwatermark import WatermarkEncoder
import wandb
from dotenv import load_dotenv
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np, cv2
from skimage.measure import block_reduce
import math

from PIL import Image
import re


# -- Update for visualization of attention maps --
from ldm.modules.attention import BasicTransformerBlock, CrossAttention


"""
uv run python scripts/txt2img_attn.py \
  --prompt "a photo of a cat" \
  --ckpt "/scratch-shared/holy-triangle/weights/stable-diffusion-2-1/v2-1_768-ema-pruned.ckpt" \
  --config "configs/stable-diffusion/v2-inference-v.yaml" \
  --H 768 --W 768 \
  --ref_img "data/cat.jpg" \
  --ref_blend_weight 0.2 \
  --aligner_model_path "/scratch-shared/holy-triangle/weights/img2text_aligner_fixed/flickr30k_cosine/model_best.pth"
"""
# ---------- hooks ----------
class AttnStore:
    def __init__(self, unet, layer_idxs=None):
        self.store = []
        self.hooks = []
        
        # First, let's see what the actual module names look like
        print("=== All modules with attn2 ===")
        attn_modules = []
        for i, (name, module) in enumerate(unet.named_modules()):
            if hasattr(module, 'attn2'):
                print(f"{i:3d}: {name}")
                attn_modules.append((i, name, module.attn2))
        
        print(f"\nFound {len(attn_modules)} attention modules")
        
        for name, module in unet.named_modules():
            if isinstance(module, CrossAttention):
                print(name, module.heads)


        # Use simple indexing for now
        if layer_idxs is None:
            if len(attn_modules) >= 3:
                layer_idxs = [0, len(attn_modules)//2, len(attn_modules)-1]
            else:
                layer_idxs = list(range(len(attn_modules)))
        
        self.layer_idxs = layer_idxs
        self.layer_names = []
        
        # Hook the selected layers
        for idx in layer_idxs:
            if idx < len(attn_modules):
                orig_idx, name, attn_module = attn_modules[idx]
                hook = attn_module.register_forward_hook(self._hook(len(self.hooks), name))
                self.hooks.append(hook)
                self.layer_names.append(self._make_readable_name(name, orig_idx))
        
        print(f"\nHooked layers:")
        for i, name in enumerate(self.layer_names):
            print(f"  {i}: {name}")

    def _make_readable_name(self, full_name, idx):
        parts = full_name.split('.')
        # 1) Figure out which block this is in
        block_name = None
        for kind in ["input_blocks", "down_blocks", "mid_block", "up_blocks", "output_blocks"]:
            if kind in parts:
                i = parts.index(kind)
                # Normalize “mid_block” vs. plural
                pretty = kind.replace('_', ' ').title().replace('Block', 'Block')
                block_name = pretty
                # grab the index if there’s a number right after
                if i+1 < len(parts) and parts[i+1].isdigit():
                    block_name += f" {parts[i+1]}"
                break

        # 2) Grab the transformer index

        if block_name:
            return f"{block_name}".strip()
        else:
            # fallback
            return f"Layer {idx}"


    def _hook(self, hook_idx, name):
        def fn(module, input, output):
            # Store with metadata
            if isinstance(output, tuple) and len(output) > 1:
                attn_weights = output[1]
            else:
                attn_weights = output
                
            self.store.append({
                'hook_idx': hook_idx,
                'layer_name': name,
                'attention': attn_weights.detach().cpu(),
                'step': len(self.store) // len(self.hooks)
            })
        return fn

    def clear(self):
        self.store.clear()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


# Simple version of build_heatmaps that works with the debug store
def build_heatmaps(store, H, W, batch_size=3, sample_idx=0):
    if not store.store:
        raise ValueError("No attention maps collected!")

    num_hooks = len(store.hooks)
    last_maps = store.store[-num_hooks:] if len(store.store) >= num_hooks else store.store

    cond_heatmaps = []
    uncond_heatmaps = []
    used_names = []

    for i, entry in enumerate(last_maps):
        t = entry['attention']
        print(f"Attention shape: {t.shape}, Layer: {entry['layer_name']}, Hook index: {entry['hook_idx']}")

        if t.ndim == 3:
            try:
                t = t.view(2, batch_size, -1, t.shape[-1])  # [cond/uncond, batch, tokens, dim]
                cond_patch = t[1, sample_idx].mean(-1)
                uncond_patch = t[0, sample_idx].mean(-1)
            except:
                print("[warn] Cannot split into [2, B, Q, D]")
                continue
        elif t.ndim == 2:
            cond_patch = uncond_patch = t.mean(1)
        else:
            print(f"[warn] Unexpected attention shape: {t.shape}")
            continue

        for patch, target in [(cond_patch, cond_heatmaps), (uncond_patch, uncond_heatmaps)]:
            q = patch.numel()
            grid = int(math.sqrt(q))
            if grid * grid != q:
                print(f"[warn] skipping map with Q={q} (not square)")
                continue

            heat = patch.view(grid, grid).numpy().astype(np.float32)
            heat = np.nan_to_num(heat, nan=0.0, posinf=0.0, neginf=0.0)

            heat -= heat.min()
            if heat.max() > 1e-8:
                heat /= heat.max()
            else:
                heat[:] = 0

            heat = cv2.resize(heat, (W, H), interpolation=cv2.INTER_CUBIC)
            target.append(heat)

        layer_name = store.layer_names[entry['hook_idx']] if entry['hook_idx'] < len(store.layer_names) else f"Layer {entry['hook_idx']}"
        used_names.append(layer_name)

    if not cond_heatmaps or not uncond_heatmaps:
        raise ValueError("No valid heatmaps after filtering.")

    return cond_heatmaps, uncond_heatmaps, used_names

# Usage:
# store = AttnSt


def overlay(img_pil, heat, alpha=0.6, colormap=cv2.COLORMAP_JET):
    """
    Overlays a (H,W) heatmap onto a PIL RGB image.
    """
    img_pil = img_pil.convert("RGB")
    
    # Create heatmap
    heat_uint8 = (heat * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heat_uint8, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = Image.fromarray(heatmap).resize(img_pil.size, Image.LANCZOS)
    
    # Blend with original image
    blended = Image.blend(img_pil, heatmap, alpha)
    return blended


def overlay_grid(img_pil, heatmaps, titles=None, alpha=0.6, figsize_per_plot=4):
    """
    Creates a grid visualization of heatmaps overlaid on the image.
    """

    n = len(heatmaps)
    fig, axes = plt.subplots(1, n + 1, figsize=((n + 1) * figsize_per_plot, figsize_per_plot))
    
    if n == 0:
        axes = [axes]  # make iterable

    # Show original image first
    axes[0].imshow(img_pil)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Show each heatmap overlay
    for i, heat in enumerate(heatmaps):
        blended = overlay(img_pil, heat, alpha)
        axes[i + 1].imshow(blended)
        axes[i + 1].axis("off")
        
        title = titles[i] if titles and i < len(titles) else f"Heatmap {i}"
        axes[i + 1].set_title(title, pad=10)

    # Remove plt.tight_layout()
    fig.subplots_adjust(top=0.85)  # or 0.88, adjust as needed

    return fig



load_dotenv()

torch.set_grad_enabled(False)

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if device == torch.device("cuda"):
        model.cuda()
    elif device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a professional photograph of an astronaut riding a triceratops",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default=f"outputs/txt2img-samples/{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm",
        action='store_true',
        help="use DPM (2) sampler",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=3,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file, separated by newlines",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v2-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="repeat each prompt in file this often",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device on which Stable Diffusion will be run",
        choices=["cpu", "cuda"],
        default="cuda"
    )
    parser.add_argument(
        "--torchscript",
        action='store_true',
        help="Use TorchScript",
    )
    parser.add_argument(
        "--ipex",
        action='store_true',
        help="Use Intel® Extension for PyTorch*",
    )
    parser.add_argument(
        "--bf16",
        action='store_true',
        help="Use bfloat16",
    )
    parser.add_argument(
        "--log_steps",
        nargs="+",
        type=int,
        default=[0, 10, 20, 30, 40, -1],  # -1 means final step
        help="Steps to log intermediate results at"
    )
    parser.add_argument(
        "--ref_img",
        type=str,
        default=None,
        help="Path to reference image"
    )
    parser.add_argument(
        "--ref_blend_weight",
        type=float,
        default=0.05,
        help="Blend weight for reference image. 1.0 corresponds to full destruction of information in init image. Used to balance the influence of the reference image and the prompt.",
    )
    parser.add_argument(
        "--aligner_model_path",
        type=str,
        default="weights/img2text_aligner/coco_cosine/model_best.pth",
        help="Path to the aligner model. If not specified, the default model will be used.",
    )
    
    # -- Update for visualization of attention maps --
    parser.add_argument("--attn_token", type=str, default=None,
                    help="token whose attention map is blended; default = first")
    
    # Add debug flag
    parser.add_argument("--debug_attn", action='store_true',
                    help="Print debug information for attention maps")

    
    opt = parser.parse_args()
    return opt


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def main(opt):
    seed_everything(opt.seed)
    # Set project and entity name from environment variables
    opt.wandb_project = os.getenv("WANDB_PROJECT", "stable-diffusion-v2")
    opt.wandb_entity = os.getenv("WANDB_ENTITY", "FoMo-2025")

    config = OmegaConf.load(f"{opt.config}")
    
    device = torch.device("cuda") if opt.device == "cuda" else torch.device("cpu")
    model = load_model_from_config(config, f"{opt.ckpt}", device)
    
    attn = AttnStore(model.model.diffusion_model)

    
    # Set the blend weight for the reference image
    if opt.ref_img:
        model.set_blend_weight(opt.ref_blend_weight)
        model.set_use_ref_img(True)
        model.create_ref_img_encoder()
        model.create_image_to_text_aligner(opt.aligner_model_path)

    if opt.plms:
        sampler = PLMSSampler(model, device=device)
    elif opt.dpm:
        sampler = DPMSolverSampler(model, device=device)
    else:
        sampler = DDIMSampler(model, device=device)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    # save prompt to file
    with open(os.path.join(outpath, "prompt.txt"), "w") as f:
        f.write(opt.prompt)

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "SDV2"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = [p for p in data for i in range(opt.repeat)]
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    sample_count = 0
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)


    if opt.ref_img:
        if os.path.exists(opt.ref_img):
            ref_image = Image.open(opt.ref_img).convert("RGB")
            ref_image = transforms.ToTensor()(ref_image).to(device).unsqueeze(0) 
            # scale to -1 to 1
            ref_image = (ref_image - 0.5) * 2           
        else:
            print(f"Warning: Reference image not found at {opt.ref_img}. Skipping.")
    else:
        ref_image = None
    
    if opt.torchscript or opt.ipex:
        transformer = model.cond_stage_model.model
        unet = model.model.diffusion_model
        decoder = model.first_stage_model.decoder
        additional_context = torch.cpu.amp.autocast() if opt.bf16 else nullcontext()
        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

        if opt.bf16 and not opt.torchscript and not opt.ipex:
            raise ValueError('Bfloat16 is supported only for torchscript+ipex')
        if opt.bf16 and unet.dtype != torch.bfloat16:
            raise ValueError("Use configs/stable-diffusion/intel/ configs with bf16 enabled if " +
                             "you'd like to use bfloat16 with CPU.")
        if unet.dtype == torch.float16 and device == torch.device("cpu"):
            raise ValueError("Use configs/stable-diffusion/intel/ configs for your model if you'd like to run it on CPU.")

        if opt.ipex:
            import intel_extension_for_pytorch as ipex
            bf16_dtype = torch.bfloat16 if opt.bf16 else None
            transformer = transformer.to(memory_format=torch.channels_last)
            transformer = ipex.optimize(transformer, level="O1", inplace=True)

            unet = unet.to(memory_format=torch.channels_last)
            unet = ipex.optimize(unet, level="O1", auto_kernel_selection=True, inplace=True, dtype=bf16_dtype)

            decoder = decoder.to(memory_format=torch.channels_last)
            decoder = ipex.optimize(decoder, level="O1", auto_kernel_selection=True, inplace=True, dtype=bf16_dtype)

        if opt.torchscript:
            with torch.no_grad(), additional_context:
                # get UNET scripted
                if unet.use_checkpoint:
                    raise ValueError("Gradient checkpoint won't work with tracing. " +
                    "Use configs/stable-diffusion/intel/ configs for your model or disable checkpoint in your config.")

                img_in = torch.ones(2, 4, 96, 96, dtype=torch.float32)
                t_in = torch.ones(2, dtype=torch.int64)
                context = torch.ones(2, 77, 1024, dtype=torch.float32)
                scripted_unet = torch.jit.trace(unet, (img_in, t_in, context))
                scripted_unet = torch.jit.optimize_for_inference(scripted_unet)
                print(type(scripted_unet))
                model.model.scripted_diffusion_model = scripted_unet

                # get Decoder for first stage model scripted
                samples_ddim = torch.ones(1, 4, 96, 96, dtype=torch.float32)
                scripted_decoder = torch.jit.trace(decoder, (samples_ddim))
                scripted_decoder = torch.jit.optimize_for_inference(scripted_decoder)
                print(type(scripted_decoder))
                model.first_stage_model.decoder = scripted_decoder

        prompts = data[0]
        print("Running a forward pass to initialize optimizations")
        uc = None
        if opt.scale != 1.0:
            uc = model.get_learned_conditioning(batch_size * [""])
        if isinstance(prompts, tuple):
            prompts = list(prompts)

        with torch.no_grad(), additional_context:
            for _ in range(3):
                c = model.get_learned_conditioning(prompts, ref_image=ref_image)
            samples_ddim, _ = sampler.sample(S=5,
                                             conditioning=c,
                                             batch_size=batch_size,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=opt.scale,
                                             unconditional_conditioning=uc,
                                             eta=opt.ddim_eta,
                                             x_T=start_code)
            print("Running a forward pass for decoder")
            for _ in range(3):
                x_samples_ddim = model.decode_first_stage(samples_ddim)

    def clean(s):
        return s.replace(" ", "_").replace("/", "_").replace("-", "_").lower()

    wandb_run_name = (
        f"alpha={opt.ref_blend_weight:.3f}"
        f"|prompt={clean(opt.prompt)[:30]}"
        f"|ref={os.path.splitext(os.path.basename(opt.ref_img))[0] if opt.ref_img else 'noref'}"
        f"|aligner={'/'.join(opt.aligner_model_path.split('/')[-3:])}"
    )

    # TODO: for now hard coded, to be fixed
    loss = "cosine" if "cosine" in opt.aligner_model_path else "infonce"
    exclude_cls = True

    wandb.init(
    project=opt.wandb_project, 
    entity=opt.wandb_entity,
    name=wandb_run_name,
    config=vars(opt),
        tags=[
            f"loss={loss}",
            f"exclude_cls={exclude_cls}",
        ]
    )

    precision_scope = autocast if opt.precision=="autocast" or opt.bf16 else nullcontext
    with torch.no_grad(), \
        precision_scope(opt.device), \
        model.ema_scope():
            all_samples = list()
            def wandb_img_callback(pred_x0, i):
                total_steps = opt.steps
                log_steps = [s if s >= 0 else total_steps-1 for s in opt.log_steps]
                
                if i not in log_steps:
                    return
                
                # Get current iteration and batch information from global variables
                iteration = wandb_img_callback.current_iteration
                prompt_idx = wandb_img_callback.current_prompt_idx
                
                # Initialize storage dict if needed
                if not hasattr(wandb_img_callback, 'stored_images'):
                    wandb_img_callback.stored_images = {}
                
                # Create a unique key for this batch
                batch_key = f"iter{iteration}_prompt{prompt_idx}_step{i}"
                
                # Convert to image format wandb can handle
                images = []
                for sample_idx, x_sample in enumerate(pred_x0):
                    x_sample = model.decode_first_stage(x_sample.unsqueeze(0))[0]
                    x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    images.append(wandb.Image(x_sample.astype(np.uint8), 
                                              caption=f"Iter {iteration}, Prompt {prompt_idx}, Sample {sample_idx}, Step {i}/{total_steps}"))
                
                # Store images under a unique key
                wandb_img_callback.stored_images[batch_key] = {
                    'images': images,
                    'step': i,
                    'iteration': iteration,
                    'prompt_idx': prompt_idx
                }

            # Initialize the static variables
            wandb_img_callback.current_iteration = 0
            wandb_img_callback.current_prompt_idx = 0
            wandb_img_callback.stored_images = {}

            for n in trange(opt.n_iter, desc="Sampling"):
                wandb_img_callback.current_iteration = n
                for prompt_idx, prompts in enumerate(tqdm(data, desc="data")):
                    wandb_img_callback.current_prompt_idx = prompt_idx
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts, ref_image=ref_image)
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    
                    # -- Update for visualization of attention maps --
                    attn.clear()

                    samples, _ = sampler.sample(S=opt.steps,
                                                     conditioning=c,
                                                     batch_size=opt.n_samples,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=opt.ddim_eta,
                                                     x_T=start_code,
                                                     img_callback=wandb_img_callback)
                    

                    # before decoding samples or doing anything that breaks the computation graph
                    

                    x_samples = model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    # Debug attention store if needed
                    if opt.debug_attn:
                        print("\nAttention Store Contents:")
                        for key, maps in attn.store.items():
                            if maps:
                                print(f"Layer {key}: {len(maps)} maps, shape: {maps[0].shape}")

                    print(f"Number of samples: {len(x_samples)}")
                    print(f"batch size: {batch_size}, n_rows: {n_rows}")
                    for sample_idx, x_sample in enumerate(x_samples):
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        img = put_watermark(img, wm_encoder)
                        img.save(os.path.join(sample_path, f"{base_count:05}.png"))

            
                        
                        # Kleine blokken

                        cond_heatmaps, uncond_heatmaps, used_names = build_heatmaps(attn, opt.H, opt.W, batch_size=opt.n_samples, sample_idx=sample_idx)


                        
                        # check if attention maps are empty
                        if not cond_heatmaps or not uncond_heatmaps:
                            print(f"No attention maps collected for sample {base_count:05}. Skipping overlay.")
                            continue
                        
                        
                        fig_cond = overlay_grid(img, cond_heatmaps, titles=used_names)
                        fig_cond.savefig(f"{sample_path}/{base_count:05}_attn_cond.png", dpi=300)
                        plt.close(fig_cond)

                        fig_uncond = overlay_grid(img, uncond_heatmaps, titles=used_names)
                        fig_uncond.savefig(f"{sample_path}/{base_count:05}_attn_uncond.png", dpi=300)
                        plt.close(fig_uncond)



                        base_count += 1
                        sample_count += 1

                    all_samples.append(x_samples)

            # additionally, save as grid
            grid = torch.stack(all_samples, 0)
            grid = rearrange(grid, 'n b c h w -> (n b) c h w')
            grid = make_grid(grid, nrow=n_rows)

            # to image
            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            grid = Image.fromarray(grid.astype(np.uint8))
            grid = put_watermark(grid, wm_encoder)
            grid.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
            grid_count += 1
    
    # After sampling completes
    if hasattr(wandb_img_callback, 'stored_images'):
        # Group by diffusion steps to maintain slider functionality
        step_to_batch = {}
        
        # Organize batches by step
        for batch_key, batch_data in wandb_img_callback.stored_images.items():
            step = batch_data['step']
            if step not in step_to_batch:
                step_to_batch[step] = []
            step_to_batch[step].append(batch_data)
        
        # Log each step with all its images from different iterations/prompts
        for step_idx, step in enumerate(sorted(step_to_batch.keys())):
            all_batches = step_to_batch[step]
            all_images = []
            
            for batch_data in all_batches:
                all_images.extend(batch_data['images'])
            
            wandb.log({
                "intermediate_samples": all_images,
            }, step=step_idx)
        
        # Clear stored images
        wandb_img_callback.stored_images = {}

    wandb.finish()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)