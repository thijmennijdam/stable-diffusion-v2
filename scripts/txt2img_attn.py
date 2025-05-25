import argparse
import os
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
from ldm.models.diffusion.ddim import DDIMSampler # Ensure this is your MODIFIED DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.ddpm import CrossAttentionFusion
# Import PNO functions from scripts.pno
from scripts.vcf.pno import instantiate_clip_model_for_pno_trajectory_loss, optimize_prompt_noise_trajectory
from torchvision import transforms
import matplotlib.pyplot as plt

# ------ Attention code step 0/4 ------
# -- Update for visualization of attention maps --
from ldm.modules.attention import BasicTransformerBlock, CrossAttention


"""
uv run python scripts/txt2img_attn.py \
  --prompt "a photo of a cat" \
  --ckpt "/scratch-shared/holy-triangle/weights/stable-diffusion-2-1/v2-1_768-ema-pruned.ckpt" \
  --config "configs/stable-diffusion/v2-inference-v.yaml" \
  --H 768 --W 768 \
  --ref_img "data/van_gogh_starry_night.jpg" \
  --ref_blend_weight 0.75 \
  --aligner_model_path "/scratch-shared/holy-triangle/weights/img2text_aligner_fixed/flickr30k_infonce/model_best.pth"
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

# ------ Attention code step 0/4 ------

load_dotenv()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
        default=0.1,
        help="Blend weight for reference image. 1.0 corresponds to full destruction of information in init image. Used to balance the influence of the reference image and the prompt.",
    )
    # PNO Trajectory specific arguments
    pno_group = parser.add_argument_group('Prompt-Noise Trajectory Optimization (PNO Trajectory)')
    pno_group.add_argument("--use_pno_trajectory", action='store_true', help="Enable PNO trajectory optimization.")
    pno_group.add_argument("--pno_steps", type=int, default=10, help="Number of PNO optimization steps.")
    pno_group.add_argument("--lr_prompt", type=float, default=1e-4, help="Learning rate for prompt in PNO.")
    pno_group.add_argument("--lr_noise", type=float, default=1e-4, help="Learning rate for noise in PNO.")
    pno_group.add_argument("--pno_noise_reg_gamma", type=float, default=0.5, help="Weight for PNO noise regularization.")
    pno_group.add_argument("--pno_noise_reg_k_power", type=int, default=1, help="k = 4^power for PNO regularization.")
    pno_group.add_argument("--pno_noise_reg_shuffles", type=int, default=20, help="Shuffles for PNO noise regularization.")
    pno_group.add_argument("--pno_disable_reg", action='store_true', help="Disable PNO noise regularization.")
    pno_group.add_argument("--pno_clip_grad_norm", type=float, default=1.0, help="Max norm for gradient clipping in PNO (0 to disable).")
    parser.add_argument(
        "--ref_first",
        action='store_true',
        help="If set, reference image features will be concatenated before text features",
    )
    parser.add_argument(
        "--fusion_token_type",
        type=str,
        default="all",
        choices=["cls_only", "except_cls", "all"],
        help="Which image tokens to use for fusion: 'cls_only' (just CLS token), 'except_cls' (all except CLS), 'all' (all tokens)"
    )
    parser.add_argument(
        "--fusion_type",
        type=str,
        default="alpha_blend",
        choices=["alpha_blend", "cross_attention", "concat"],
        help="Use cross-attention fusion instead of concatenation"
    )
    # Aligner arguments
    parser.add_argument(
        "--aligner_version",
        type=str,
        default="v1",
        choices=["v1", "v2"],
        help="Aligner model version (v1 or v2)"
    )
    parser.add_argument(
        "--aligner_dataset",
        type=str,
        default="coco",
        help="Dataset used for training the aligner"
    )
    parser.add_argument(
        "--aligner_loss",
        type=str,
        default="infonce",
        choices=["infonce", "mmd", "mse"],
        help="Loss function used for training the aligner"
    )
    parser.add_argument(
        "--aligner_batch_size",
        type=int,
        default=64,
        help="Batch size used for training the aligner"
    )
    parser.add_argument(
        "--aligner_dropout",
        type=float,
        default=0.1,
        help="Dropout rate used for training the aligner"
    )
    parser.add_argument(
        "--aligner_exclude_cls",
        type=str2bool,
        default=True,
        const=True,
        nargs='?',
        help="Whether CLS token was excluded during aligner training"
    )
    
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
    
    # Add aligner parameters to the model config if reference image is used
    if opt.ref_img:
        config.model.params.aligner_version = opt.aligner_version
        config.model.params.aligner_dropout = opt.aligner_dropout

    device = torch.device("cuda") if opt.device == "cuda" else torch.device("cpu")
    model = load_model_from_config(config, f"{opt.ckpt}", device)
    
    # ------ Attention code step 1/4 ------
    attn = AttnStore(model.model.diffusion_model)
    # ------ Attention code step 1/4 ------

    if opt.ref_img:
        model.set_blend_weight(opt.ref_blend_weight)
        model.set_use_ref_img(True)
        model.create_ref_img_encoder()

        # Construct aligner model path
        aligner_model_path = (
            
            f"/scratch-shared/holy-triangle/weights/aligner_models/version_{opt.aligner_version}/"
            f"dataset_{opt.aligner_dataset}/"
            f"loss_{opt.aligner_loss}/"
            f"batch_{opt.aligner_batch_size}/"
            f"dropout_{opt.aligner_dropout}/"
            f"exclude_cls_{opt.aligner_exclude_cls}/"
            f"model_best.pth"
        )
        print(f"Loading aligner model from: {aligner_model_path}")
        model.create_image_to_text_aligner(aligner_model_path)
        model.ref_first = opt.ref_first
        model.fusion_token_type = opt.fusion_token_type
        model.fusion_type = opt.fusion_type

        # Create cross-attention fusion module if needed, using ref_blend_weight as alpha
        if opt.fusion_type == "cross_attention":
            model.cross_attention_fusion = CrossAttentionFusion(dim=1024, alpha=opt.ref_blend_weight).to(model.device)

    if opt.plms:
        sampler = PLMSSampler(model, device=device)
    elif opt.dpm:
        sampler = DPMSolverSampler(model, device=device)
    else:
        sampler = DDIMSampler(model, device=device)
        sampler.ddim_eta = opt.ddim_eta # Set eta on the sampler instance (needed for PNO)

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
            # ref_image = (ref_image - 0.5) * 2           
        else:
            print(f"Warning: Reference image {opt.ref_img} not found.")
            if opt.use_pno_trajectory: raise ValueError("PNO trajectory requires a valid --ref_img.")
            ref_image = None 
    elif opt.use_pno_trajectory:
        raise ValueError("--ref_img is mandatory when --use_pno_trajectory is enabled.")
    
    clip_image_encoder_pno, clip_preprocess_pno = (None, None)
    if opt.use_pno_trajectory:
        print("Instantiating CLIP model for PNO trajectory loss...")
        clip_image_encoder_pno, clip_preprocess_pno = instantiate_clip_model_for_pno_trajectory_loss(device=device)
    
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
                c = model.get_learned_conditioning(prompts)
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
        "pno" if opt.use_pno_trajectory else "standard"
        f"|fusion_type={opt.fusion_type}"
        f"|prompt={clean(opt.prompt)[:30]}"
        f"|ref_img={os.path.splitext(os.path.basename(opt.ref_img))[0] if opt.ref_img else 'noref'}"
        f"|alpha={opt.ref_blend_weight}"
        f"|aligner_v={opt.aligner_version}"
        f"|dataset={opt.aligner_dataset}"
        f"|loss={opt.aligner_loss}"
    )

    opt.create_date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    wandb.init(
        project=opt.wandb_project, 
        entity=opt.wandb_entity,
        name=wandb_run_name,
        config=vars(opt), # add all the configs in the wandb run
    )


    outer_grad_context = nullcontext() if opt.use_pno_trajectory else torch.no_grad()
    precision_scope_device = opt.device if opt.device == "cuda" else "cpu"
    current_precision_scope = autocast(device_type=precision_scope_device, enabled=(opt.precision == "autocast" or opt.bf16))
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
            
    with outer_grad_context, current_precision_scope, model.ema_scope():
        all_samples = list()

        for n in trange(opt.n_iter, desc="Sampling"):
            wandb_img_callback.current_iteration = n
            
            # Loop through batches of prompts from `data`
            for prompt_idx, prompts in enumerate(tqdm(data, desc="data")):
                wandb_img_callback.current_prompt_idx = prompt_idx

                current_batch_size = len(prompts) # Actual number of prompts in this batch
                shape = [opt.C, opt.H // opt.f, opt.W // opt.f] 
                
                x_samples = None # Will hold [0,1] image tensors for this batch

                if opt.use_pno_trajectory:
                    print(f"\nInitiating PNO Trajectory for {current_batch_size} prompt(s) from batch {prompt_idx} individually...")
                    pno_batch_outputs_list = [] 
                    # In PNO mode, we run optimization for each prompt in the current "batch" from `data`
                    # This is equivalent to `opt.n_samples` independent PNO runs if `data` had one batch of replicated prompts.
                    for p_idx_in_current_batch, single_prompt_text in enumerate(prompts):
                        print(f"  PNO Run {p_idx_in_current_batch+1}/{current_batch_size} for prompt: '{single_prompt_text}'")
                        
                        pno_img_tensor = optimize_prompt_noise_trajectory(
                            model_ldm=model, sampler_ldm=sampler, 
                            clip_image_encoder_for_loss=clip_image_encoder_pno, 
                            clip_preprocess_for_loss=clip_preprocess_pno,
                            text_prompt=single_prompt_text,
                            ref_image=ref_image,
                            opt_cmd=opt, device=device
                        )
                        if pno_img_tensor is not None:
                            pno_batch_outputs_list.append(pno_img_tensor) # These are [-1,1]
                    
                    if pno_batch_outputs_list:
                        x_samples_from_pno_raw = torch.cat(pno_batch_outputs_list, dim=0) 
                        x_samples = torch.clamp((x_samples_from_pno_raw + 1.0) / 2.0, min=0.0, max=1.0)
                    else:
                        print(f"Warning: PNO trajectory yielded no results for prompt batch {prompt_idx}.")
                        continue 

                else: # Standard Pipeline
                    print(f"\nRunning Standard Pipeline for batch {prompt_idx+1} (Size: {current_batch_size})")
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(current_batch_size * [""])
                    
                    if opt.ref_img:        
                        c = model.get_learned_conditioning(prompts, ref_image=ref_image)
                    else:
                        c = model.get_learned_conditioning(prompts)
                    
                    current_batch_start_code = None
                    if opt.fixed_code:
                        if 'fixed_start_code_base' not in locals() or fixed_start_code_base is None:
                             fixed_start_code_base = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
                        current_batch_start_code = fixed_start_code_base[:current_batch_size]

                    # ------ Attention code step 2/4 ------
                    attn.clear()
                    # ------ Attention code step 2/4 ------

                    samples_latent, _ = sampler.sample( 
                        S=opt.steps,
                        conditioning=c,
                        batch_size=current_batch_size,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=opt.scale,
                        unconditional_conditioning=uc,
                        eta=opt.ddim_eta,
                        x_T=current_batch_start_code,
                        img_callback=wandb_img_callback 
                    )
                    x_samples = model.decode_first_stage(samples_latent) 
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0) 

                # Common processing after batch generation (PNO or Standard)
                if x_samples is not None:
                    all_samples.append(x_samples.cpu())

                    # Saving individual samples
                    for sample_idx, x_sample_tensor_normalized in enumerate(x_samples): 
                        img_np_to_save = 255. * rearrange(x_sample_tensor_normalized.cpu().numpy(), 'c h w -> h w c')
                        img_pil_to_save = Image.fromarray(img_np_to_save.astype(np.uint8))
                        img_pil_to_save = put_watermark(img_pil_to_save, wm_encoder)
                        img_pil_to_save.save(os.path.join(sample_path, f"{base_count:05}.png"))
                        base_count += 1
                        
                        # ------ Attention code step 4/4 ------

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
                        
                        # ------ Attention code step 4/4 ------
        
        # Log collected intermediate DDIM images from standard pipeline
        if wandb_img_callback.stored_images:
            step_to_batch_images_map = {} # Renamed from step_to_batch
            for unique_call_key, call_data_dict in wandb_img_callback.stored_images.items():
                actual_ddim_step = call_data_dict['step']
                images_from_call = call_data_dict['images']
                if actual_ddim_step not in step_to_batch_images_map:
                    step_to_batch_images_map[actual_ddim_step] = []
                step_to_batch_images_map[actual_ddim_step].extend(images_from_call)

            sorted_ddim_steps_logged = sorted(step_to_batch_images_map.keys())
            for wandb_log_idx, ddim_s_val in enumerate(sorted_ddim_steps_logged):
                all_imgs_for_this_ddim_s = step_to_batch_images_map[ddim_s_val]
                if all_imgs_for_this_ddim_s:
                    wandb.log({"Standard DDIM Intermediate Samples": all_imgs_for_this_ddim_s}, step=wandb_log_idx) 
            wandb_img_callback.stored_images = {} 

        if all_samples:
            grid_tensor_all = torch.cat(all_samples, dim=0)
            grid_display_rows = n_rows
            if grid_tensor_all.shape[0] == 0: print("Warning: No samples for grid.")
            elif grid_tensor_all.shape[0] < grid_display_rows : grid_display_rows = grid_tensor_all.shape[0]
            
            if grid_tensor_all.shape[0] > 0:
                grid = make_grid(grid_tensor_all, nrow=grid_display_rows if grid_display_rows > 0 else 1)
                grid_numpy = 255. * rearrange(grid, 'c h w -> h w c').numpy()
                grid_pil = Image.fromarray(grid_numpy.astype(np.uint8))
                grid_pil = put_watermark(grid_pil, wm_encoder)
                grid_pil.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                grid_count += 1
    
    wandb.finish()
    print(f"Your samples are ready and waiting for you here: \n{outpath} \nEnjoy.")

if __name__ == "__main__":
    cmd_opt = parse_args()
    if cmd_opt.use_pno_trajectory and cmd_opt.ddim_eta == 0.0:
        print("Warning: PNO trajectory for intermediate noises is most effective when --ddim_eta > 0.")
    if not cmd_opt.ckpt: raise ValueError("Please specify checkpoint path with --ckpt")
    if not os.path.exists(cmd_opt.ckpt): raise FileNotFoundError(f"Checkpoint not found: {cmd_opt.ckpt}")
    if not os.path.exists(cmd_opt.config): raise FileNotFoundError(f"Config not found: {cmd_opt.config}")

    main(cmd_opt)