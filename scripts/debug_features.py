#!/usr/bin/env python3
"""
Debug script to verify that different reference images produce different features.
Run this before running txt2img.py to ensure your reference images are being processed correctly.
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import argparse
import os
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    # If config is a string (path to config file), load it first
    if isinstance(config, str):
        config = OmegaConf.load(config)
    
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

def load_and_preprocess_image(image_path, device):
    """Load and preprocess image for CLIP"""
    ref_image = Image.open(image_path).convert("RGB")
    ref_image = ref_image.resize((224, 224), Image.LANCZOS)
    ref_image = transforms.ToTensor()(ref_image).to(device).unsqueeze(0)
    return ref_image

def normalize_image_for_clip(image):
    """Properly normalize image for CLIP model"""
    # Convert from [0, 1] to CLIP normalization
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(image.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(image.device)
    
    mean = mean.view(1, 3, 1, 1)
    std = std.view(1, 3, 1, 1)
    
    normalized = (image - mean) / std
    return normalized

def extract_features(model, image_path, aligner_path, device):
    """Extract features from an image using CLIP + aligner"""
    # Load image
    ref_image = load_and_preprocess_image(image_path, device)
    
    # Setup model components
    model.set_use_ref_img(True)
    model.create_ref_img_encoder()
    model.create_image_to_text_aligner(aligner_path)
    
    # Extract features
    with torch.no_grad():
        # Normalize for CLIP
        normalized_image = normalize_image_for_clip(ref_image)
        
        # Extract CLIP features
        clip_features = model.clip_model(normalized_image)
        print(f"CLIP features shape: {clip_features.shape}")
        
        # Apply aligner
        aligned_features = model.image_to_text_aligner(clip_features)
        print(f"Aligned features shape: {aligned_features.shape}")
        
        # Process to final shape
        if aligned_features.dim() == 3:
            final_features = aligned_features.mean(dim=1)
        elif aligned_features.dim() == 2 and aligned_features.shape[0] != 1:
            final_features = aligned_features.mean(dim=0, keepdim=True)
        else:
            final_features = aligned_features
            
        return final_features.cpu().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stable-diffusion/v2-inference.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--aligner_model_path", default="weights/img2text_aligner/coco_cosine/model_best.pth")
    parser.add_argument("--ref_images", nargs="+", required=True, help="List of reference image paths")
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    # Load model
    config = "../configs/stable-diffusion/v2-inference.yaml"
    device = torch.device(args.device)
    model = load_model_from_config(config, args.ckpt, device)
    
    print("Testing reference image feature extraction...")
    print("=" * 60)
    
    features_dict = {}
    
    # Extract features for each image
    for img_path in args.ref_images:
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found, skipping...")
            continue
            
        print(f"\nProcessing: {os.path.basename(img_path)}")
        features = extract_features(model, img_path, args.aligner_model_path, device)
        features_dict[img_path] = features
        
        print(f"Features shape: {features.shape}")
        print(f"Features mean: {features.mean():.6f}")
        print(f"Features std: {features.std():.6f}")
        print(f"Features range: [{features.min():.6f}, {features.max():.6f}]")
    
    # Compare features between images
    print("\n" + "=" * 60)
    print("FEATURE COMPARISON:")
    print("=" * 60)
    
    image_names = list(features_dict.keys())
    for i in range(len(image_names)):
        for j in range(i + 1, len(image_names)):
            name1 = os.path.basename(image_names[i])
            name2 = os.path.basename(image_names[j])
            
            feat1 = features_dict[image_names[i]]
            feat2 = features_dict[image_names[j]]
            
            # Calculate cosine similarity
            cos_sim = np.dot(feat1.flatten(), feat2.flatten()) / (
                np.linalg.norm(feat1.flatten()) * np.linalg.norm(feat2.flatten())
            )
            
            # Calculate L2 distance
            l2_dist = np.linalg.norm(feat1.flatten() - feat2.flatten())
            
            print(f"{name1} vs {name2}:")
            print(f"  Cosine similarity: {cos_sim:.6f}")
            print(f"  L2 distance: {l2_dist:.6f}")
            
            if cos_sim > 0.99:
                print(f"  ⚠️  WARNING: Features are very similar! This may cause identical outputs.")
            elif cos_sim > 0.95:
                print(f"  ⚠️  Features are quite similar, differences may be subtle.")
            else:
                print(f"  ✅ Features are sufficiently different.")
            print()
    
    print("\nIf features are too similar, try:")
    print("1. Using more visually distinct reference images")
    print("2. Checking if the aligner model is working correctly") 
    print("3. Increasing the blend weight (ref_blend_weight)")
    print("4. Ensuring images are preprocessed correctly")

if __name__ == "__main__":
    main()