import numpy as np
import torch
import math
from torchvision import transforms
import open_clip
from einops import rearrange
import wandb

# ################### PNO TRAJECTORY HELPER FUNCTIONS ###################

def _get_pno_reg_log_probs(current_noise_seq, m, k, device):
    if m == 0 or k == 0: 
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    mean_emp = current_noise_seq.mean(dim=0) 
    noise_seq_scaled_for_cov = current_noise_seq / math.sqrt(m) if m > 0 else current_noise_seq 
    cov_emp = noise_seq_scaled_for_cov.T @ noise_seq_scaled_for_cov 
    M1_val = torch.norm(mean_emp)
    M2_val = torch.linalg.matrix_norm(cov_emp - torch.eye(k, device=device), ord=2) if k > 0 else torch.tensor(0.0, device=device)
    log_p_m1 = - (m * M1_val ** 2) / (2 * k) if k > 0 else torch.tensor(0.0, device=device)
    log_p_m1 = torch.clamp(log_p_m1, max=-math.log(2.0))
    sqrt_k_div_m = math.sqrt(k/m) if k > 0 and m > 0 else 0.0
    cov_diff_term = torch.clamp(torch.sqrt(1 + M2_val) - 1 - sqrt_k_div_m, min=0.0)
    log_p_m2 = - m * (cov_diff_term ** 2) / 2.0
    log_p_m2 = torch.clamp(log_p_m2, max=-math.log(2.0))
    return log_p_m1, log_p_m2

def pno_regularization_loss_trajectory(noise_flat, subsample_k, num_shuffles, device):
    dim_total = noise_flat.numel()
    if dim_total == 0: return torch.tensor(0.0, device=device)
    actual_subsample_k = subsample_k
    if dim_total < subsample_k : actual_subsample_k = dim_total 
    if actual_subsample_k == 0: return torch.tensor(0.0, device=device)
    if dim_total % actual_subsample_k != 0:
        padding_size = actual_subsample_k - (dim_total % actual_subsample_k)
        noise_flat_padded = torch.cat([noise_flat, torch.zeros(padding_size, device=device, dtype=noise_flat.dtype)])
        dim_total_padded = noise_flat_padded.numel()
    else:
        noise_flat_padded = noise_flat
        dim_total_padded = dim_total
    subsample_m = dim_total_padded // actual_subsample_k
    if subsample_m == 0: return torch.tensor(0.0, device=device)
    noise_seq = noise_flat_padded.view(subsample_m, actual_subsample_k)
    log_p_m1_orig, log_p_m2_orig = _get_pno_reg_log_probs(noise_seq, subsample_m, actual_subsample_k, device)
    total_log_p_m1_shuffled = torch.tensor(0.0, device=device)
    total_log_p_m2_shuffled = torch.tensor(0.0, device=device)
    for _ in range(num_shuffles):
        shuffled_indices = torch.randperm(dim_total_padded, device=device)
        noise_flat_shuffled = noise_flat_padded[shuffled_indices]
        noise_seq_shuffled = noise_flat_shuffled.view(subsample_m, actual_subsample_k)
        log_p_m1_shuff, log_p_m2_shuff = _get_pno_reg_log_probs(noise_seq_shuffled, subsample_m, actual_subsample_k, device)
        total_log_p_m1_shuffled += log_p_m1_shuff
        total_log_p_m2_shuffled += log_p_m2_shuff
    avg_log_p_m1_shuffled = total_log_p_m1_shuffled / num_shuffles if num_shuffles > 0 else torch.tensor(0.0, device=device)
    avg_log_p_m2_shuffled = total_log_p_m2_shuffled / num_shuffles if num_shuffles > 0 else torch.tensor(0.0, device=device)
    regularization = -(log_p_m1_orig + log_p_m2_orig + avg_log_p_m1_shuffled + avg_log_p_m2_shuffled)
    return regularization

def instantiate_clip_model_for_pno_trajectory_loss(device="cuda"):
    model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=device)
    model.eval() 
    clip_preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    ])
    return model, clip_preprocess

def optimize_prompt_noise_trajectory(
    model_ldm,
    sampler_ldm,
    clip_image_encoder_for_loss, 
    clip_preprocess_for_loss,
    text_prompt,
    ref_image,
    opt_cmd, 
    device="cuda"
):
    # Ensure sampler_ldm.ddim_eta is set before this function is called
    # It's accessed via sampler_ldm.ddim_eta (as per your modified DDIMSampler)

    with torch.no_grad(): 
        fused_prompt_init = model_ldm.get_learned_conditioning([text_prompt], ref_image=ref_image)
    fused_prompt_optimized = fused_prompt_init.detach().clone().requires_grad_(True)

    shape_latent = [opt_cmd.C, opt_cmd.H // opt_cmd.f, opt_cmd.W // opt_cmd.f]
    x_T_optimized = torch.randn([1, *shape_latent], device=device, dtype=torch.float32).requires_grad_(True)
    
    intermediate_noises_optimized = None
    if sampler_ldm.ddim_eta > 0: 
        intermediate_noises_optimized = torch.randn(
            [opt_cmd.steps, *shape_latent], device=device, dtype=torch.float32
        ).requires_grad_(True)

    params_to_opt = [{'params': fused_prompt_optimized, 'lr': opt_cmd.lr_prompt}]
    params_to_opt.append({'params': x_T_optimized, 'lr': opt_cmd.lr_noise})    
    if intermediate_noises_optimized is not None:
        params_to_opt.append({'params': intermediate_noises_optimized, 'lr': opt_cmd.lr_noise})
    
    optimizer = torch.optim.AdamW(params_to_opt)

    with torch.no_grad(): 
        ref_image_for_clip = clip_preprocess_for_loss(ref_image)
        clip_ref_img_features = clip_image_encoder_for_loss.encode_image(ref_image_for_clip.to(device)).float()
        clip_ref_img_features = torch.nn.functional.normalize(clip_ref_img_features, dim=-1)

    uc_pno = None
    if opt_cmd.scale != 1.0:
        with torch.no_grad(): 
            uc_pno = model_ldm.get_learned_conditioning(1 * [""], ref_image=None) 

    x_samples_img_from_pno_step = None 
    if wandb.run and opt_cmd.pno_steps >= 0: 
        with torch.no_grad(): 
            print("Logging initial state for PNO (PNO Opt Step 0)...")
            initial_samples_latent, _ = sampler_ldm.sample(
                S=opt_cmd.steps, conditioning=fused_prompt_optimized.detach(), batch_size=1, 
                shape=shape_latent, verbose=False, x_T=x_T_optimized.detach(), 
                unconditional_guidance_scale=opt_cmd.scale, unconditional_conditioning=uc_pno,
                eta=sampler_ldm.ddim_eta, enable_grad=False, 
                optimizable_intermediate_noises=intermediate_noises_optimized.detach() if intermediate_noises_optimized is not None else None,
                img_callback=None 
            )
            x_samples_img_from_pno_step = model_ldm.decode_first_stage(initial_samples_latent)
            
            initial_x_samples_img_for_clip = (x_samples_img_from_pno_step / 2 + 0.5).clamp(0,1)
            initial_x_samples_img_for_clip = clip_preprocess_for_loss(initial_x_samples_img_for_clip)
            initial_clip_gen_img_features = clip_image_encoder_for_loss.encode_image(initial_x_samples_img_for_clip).float()
            initial_clip_gen_img_features = torch.nn.functional.normalize(initial_clip_gen_img_features, dim=-1)
            initial_objective_loss = -(initial_clip_gen_img_features @ clip_ref_img_features.T).mean()
            initial_reg_term = torch.tensor(0.0, device=device) 
            if not opt_cmd.pno_disable_reg:
                initial_noise_components = [x_T_optimized.detach().flatten()]
                if intermediate_noises_optimized is not None and sampler_ldm.ddim_eta > 0:
                    initial_noise_components.append(intermediate_noises_optimized.detach().flatten())
                initial_all_opt_noises_flat = torch.cat(initial_noise_components)
                if initial_all_opt_noises_flat.numel() > 0:
                    subsample_k_val = 4 ** opt_cmd.pno_noise_reg_k_power
                    initial_reg_term = pno_regularization_loss_trajectory(
                        initial_all_opt_noises_flat, subsample_k_val, opt_cmd.pno_noise_reg_shuffles, device)
            
            initial_x_sample_viz = torch.clamp((x_samples_img_from_pno_step + 1.0) / 2.0, min=0.0, max=1.0)
            initial_x_sample_viz_np = 255. * rearrange(initial_x_sample_viz[0].cpu().numpy(), 'c h w -> h w c')
            wandb.log({
                "PNO Intermediate Sample": wandb.Image(initial_x_sample_viz_np.astype(np.uint8), 
                                                  caption=f"PNO Opt Step 0/{opt_cmd.pno_steps} (Initial)"),
                "CLIP Loss (prompt)": initial_objective_loss.item(),
                "Noise Regularization Loss": initial_reg_term.item(), 
                "PNO Total Loss": (initial_objective_loss + opt_cmd.pno_noise_reg_gamma * initial_reg_term).item(),
            }, step=-1) 

    if opt_cmd.pno_steps > 0 : 
        print(f"Starting PNO trajectory optimization with {opt_cmd.pno_steps} PNO steps...")
        for pno_step_idx in range(opt_cmd.pno_steps): 
            optimizer.zero_grad()
            samples_latent, _ = sampler_ldm.sample(
                S=opt_cmd.steps, conditioning=fused_prompt_optimized, batch_size=1, shape=shape_latent,
                verbose=False, x_T=x_T_optimized, unconditional_guidance_scale=opt_cmd.scale,
                unconditional_conditioning=uc_pno, eta=sampler_ldm.ddim_eta, enable_grad=True, 
                optimizable_intermediate_noises=intermediate_noises_optimized, img_callback=None
            )
            with torch.enable_grad(): 
                x_samples_img_from_pno_step = model_ldm.decode_first_stage(samples_latent) 
            
            x_samples_img_for_clip = (x_samples_img_from_pno_step / 2 + 0.5).clamp(0,1)
            x_samples_img_for_clip = clip_preprocess_for_loss(x_samples_img_for_clip)
            clip_gen_img_features = clip_image_encoder_for_loss.encode_image(x_samples_img_for_clip).float()
            clip_gen_img_features = torch.nn.functional.normalize(clip_gen_img_features, dim=-1)
            objective_loss = -(clip_gen_img_features @ clip_ref_img_features.T).mean()
            reg_term = torch.tensor(0.0, device=device)
            if not opt_cmd.pno_disable_reg:
                noise_components_to_reg = [x_T_optimized.flatten()]
                if intermediate_noises_optimized is not None and sampler_ldm.ddim_eta > 0:
                    noise_components_to_reg.append(intermediate_noises_optimized.flatten())
                all_opt_noises_flat = torch.cat(noise_components_to_reg)
                if all_opt_noises_flat.numel() > 0:
                    subsample_k_val = 4 ** opt_cmd.pno_noise_reg_k_power
                    reg_term = pno_regularization_loss_trajectory(
                        all_opt_noises_flat, subsample_k_val, opt_cmd.pno_noise_reg_shuffles, device)
            total_loss = objective_loss + opt_cmd.pno_noise_reg_gamma * reg_term
            total_loss.backward() 
            if opt_cmd.pno_clip_grad_norm > 0:
                parameters_with_grads = []
                for param_group in params_to_opt:
                    if isinstance(param_group['params'], torch.Tensor):
                        if param_group['params'].grad is not None: parameters_with_grads.append(param_group['params'])
                    elif isinstance(param_group['params'], list):
                        for p_tensor in param_group['params']:
                            if isinstance(p_tensor, torch.Tensor) and p_tensor.grad is not None: 
                                parameters_with_grads.append(p_tensor)
                if parameters_with_grads: 
                    torch.nn.utils.clip_grad_norm_(parameters_with_grads, opt_cmd.pno_clip_grad_norm)
            optimizer.step()
            print(f"[PNO Traj. Step {pno_step_idx+1}/{opt_cmd.pno_steps}] ObjL: {objective_loss.item():.4f}, RegL: {reg_term.item():.4f}, TotL: {total_loss.item():.4f}")
            if wandb.run: 
                with torch.no_grad():
                    x_sample_viz = torch.clamp((x_samples_img_from_pno_step + 1.0) / 2.0, min=0.0, max=1.0)
                    x_sample_viz_np = 255. * rearrange(x_sample_viz[0].cpu().numpy(), 'c h w -> h w c')
                    wandb.log({
                        "PNO Intermediate Sample": wandb.Image(x_sample_viz_np.astype(np.uint8), 
                                                          caption=f"PNO Opt Step {pno_step_idx + 1}/{opt_cmd.pno_steps}"),
                        "CLIP Loss (prompt)": objective_loss.item(),
                        "Noise Regularization Loss": reg_term.item(),
                        "PNO Total Loss": total_loss.item(),
                    }, step=pno_step_idx) 
    
    if x_samples_img_from_pno_step is None: 
        print("Warning: No PNO steps were effectively run, initial sample not generated for PNO return. Returning None.")
        return None 

    return x_samples_img_from_pno_step.detach() 
# ################# END PNO TRAJECTORY HELPER FUNCTIONS (in scripts/pno.py) #################