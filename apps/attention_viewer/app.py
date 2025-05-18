#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────────────
#  Local-SD-v2   ·   Cross-Attention Visualiser   ·   ready-to-run
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import os, argparse, tempfile
from typing import Dict, List, Tuple

import cv2, numpy as np, torch, gradio as gr
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import nn
from omegaconf import OmegaConf

# ─── your original imports ────────────────────────────────────────────────────
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.modules.attention import BasicTransformerBlock
# (keep anything else you need)

# ─────────────────── 1.  Attention recorder ──────────────────────────────────
class AttnStore:
    """Hook every cross-attention (attn2) block and keep softmax."""
    def __init__(self): self.store: Dict[str, List[torch.Tensor]] = {}
    def _hook(self, name):
        def fn(_, __, out):
            self.store.setdefault(name, []).append(out[1].detach().cpu())
        return fn
    def install(self, unet: nn.Module):
        self._handles = []
        for n, m in unet.named_modules():
            if isinstance(m, BasicTransformerBlock):
                self._handles.append(m.attn2.register_forward_hook(self._hook(n)))
    def clear(self):       [v.clear() for v in self.store.values()]
    def remove(self):      [h.remove() for h in self._handles]

# ─────────────────── 2.  Model wrapper  ──────────────────────────────────────
class SDGenerator:
    def __init__(self, cfg:str, ckpt:str, H:int=512, W:int=512,
                 device:str="cuda", sampler_kind:str="ddim"):
        self.device = torch.device(device)
        self.model  = self._load(cfg, ckpt)
        self.sampler= self._make_sampler(sampler_kind)
        self.H, self.W = H, W
        self.tok  = self.model.cond_stage_model.tokenizer
        self.attn = AttnStore(); self.attn.install(self.model.model.diffusion_model)

    def _load(self, cfg, ckpt):
        cfg = OmegaConf.load(cfg)
        model = instantiate_from_config(cfg.model)
        state = torch.load(ckpt, map_location="cpu")["state_dict"]
        model.load_state_dict(state, strict=False)
        model.eval().to(self.device)
        return model

    def _make_sampler(self, kind):
        if kind == "plms":   return PLMSSampler(self.model, device=self.device)
        if kind == "dpm":    return DPMSolverSampler(self.model, device=self.device)
        return DDIMSampler(self.model, device=self.device)          # default

    @torch.no_grad()
    def generate(self, prompt:str, steps:int=40, scale:float=9.0
                 ) -> Tuple[Image.Image, List[str], Dict[str,np.ndarray]]:
        seed_everything(42)
        b = 1
        cond = self.model.get_learned_conditioning([prompt])
        uc   = self.model.get_learned_conditioning([""])
        shape= [4, self.H//8, self.W//8]

        self.attn.clear()
        latent, _ = self.sampler.sample(
            S=steps, conditioning=cond, unconditional_conditioning=uc,
            batch_size=b, shape=shape, unconditional_guidance_scale=scale, eta=0.0)

        img = self._latents_to_pil(latent[0])
        heatmaps = self._build_heatmaps(prompt, img.size)
        return img, list(heatmaps.keys()), heatmaps

    def _latents_to_pil(self, z):
        x = self.model.decode_first_stage(z.unsqueeze(0))
        x = torch.clamp((x+1)/2,0,1)[0]
        return Image.fromarray((x.permute(1,2,0).cpu().numpy()*255).astype(np.uint8))

    def _build_heatmaps(self, prompt:str, hw:Tuple[int,int]):
        tokens = self.tok(prompt).input_ids
        ids_to_tok = self.tok.convert_ids_to_tokens(tokens)
        all_maps = torch.cat([torch.cat(v,0) for v in self.attn.store.values()], 0)  # (N,Q,K)

        maps, H8, W8 = {}, hw[1]//8, hw[0]//8
        for tidx, tok in enumerate(ids_to_tok):
            if tok in maps: continue
            v = all_maps[:, tidx].mean(0)              # mean heads & layers
            heat = v.reshape(H8, W8).numpy()
            heat = cv2.resize(heat, hw, cv2.INTER_CUBIC)
            heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
            maps[tok] = heat.astype(np.float32)
        return maps

# ─────────────────── 3.  Visual helpers ──────────────────────────────────────
def overlay(img:Image.Image, heat:np.ndarray, alpha:float=0.6)->Image.Image:
    cmap = cv2.applyColorMap((heat*255).astype(np.uint8), cv2.COLORMAP_JET)
    cmap = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
    cmap = Image.fromarray(cmap).resize(img.size, Image.BILINEAR)
    return Image.blend(img, cmap, alpha)

# ─────────────────── 4.  Gradio interface  ───────────────────────────────────
def build_demo(gen:SDGenerator):
    def run(prompt, steps, token):
        img, toks, heats = gen.generate(prompt, steps)
        if token not in heats: token = toks[0] if toks else ""
        shown = overlay(img, heats[token]) if token else img
        return shown, gr.update(choices=toks, value=token)

    with gr.Blocks(title="Local SD-v2 Attention Explorer") as demo:
        gr.Markdown("### 🔬 Cross-Attention explorer (local weights)")
        with gr.Row():
            t_prompt = gr.Textbox(value="a professional photograph of an astronaut riding a triceratops",
                                  label="Prompt")
            t_steps  = gr.Slider(10,80,value=40,step=1,label="Steps")
            t_token  = gr.Dropdown(choices=[],label="Token")
        btn = gr.Button("Generate")
        img = gr.Image(label="Image + heat-map")
        btn.click(run, [t_prompt, t_steps, t_token], [img, t_token])
    return demo

# ─────────────────── 5.  CLI entry -- config / ckpt  ─────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="v2-inference yaml")
    ap.add_argument("--ckpt",   required=True, help="model checkpoint")
    ap.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    ap.add_argument("--sampler", choices=["ddim","plms","dpm"], default="ddim")
    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--W", type=int, default=512)
    ap.add_argument("--nogui", action="store_true")
    ap.add_argument("--prompt")
    ap.add_argument("--steps", type=int, default=40)
    args = ap.parse_args()

    gen = SDGenerator(args.config, args.ckpt, H=args.H, W=args.W,
                      device=args.device, sampler_kind=args.sampler)

    if args.nogui:
        prompt = args.prompt or "a photo of a corgi in space"
        img, toks, heats = gen.generate(prompt, args.steps)
        tok = toks[0]
        overlay(img, heats[tok]).save("vis.png")
        print("✅ saved vis.png")
    else:
        build_demo(gen).queue(concurrency_count=1).launch()
