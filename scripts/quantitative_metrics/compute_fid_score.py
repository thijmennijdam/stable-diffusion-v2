import os
import json
from glob import glob
from PIL import Image
import torch
import argparse
from tqdm import tqdm
import torchvision.transforms as T
from torchmetrics.image.fid import FrechetInceptionDistance

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion_type_a", type=str, required=True, help="First fusion type to compare (e.g. alpha_blend)")
    parser.add_argument("--fusion_type_b", type=str, required=True, help="Second fusion type to compare (e.g. concat)")
    parser.add_argument("--base_dir", type=str, default="outputs/txt2img-samples", help="Directory containing output folders")
    parser.add_argument("--max_images", type=int, default=50, help="Max images per folder to include")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def load_images_from_folder(folder, image_size=256, max_images=50):
    images = []
    files = sorted(glob(os.path.join(folder, "*.png")))[:max_images]
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor()
    ])
    for file in files:
        img = Image.open(file).convert("RGB")
        images.append(transform(img))
    return torch.stack(images)

def collect_by_fusion_type(base_dir):
    groups = {}
    for run_dir in glob(os.path.join(base_dir, "*")):
        config_path = os.path.join(run_dir, "config.json")
        print(config_path)
        samples_dir = os.path.join(run_dir, "samples")
        if not os.path.isfile(config_path) or not os.path.isdir(samples_dir):
            continue
        with open(config_path, "r") as f:
            config = json.load(f)
        key = config.get("fusion_type", "unknown")
        if key not in groups:
            groups[key] = []
        groups[key].append(samples_dir)
    return groups

def compute_fid_between_groups(group_a_paths, group_b_paths, max_images=50, device="cuda"):
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    for folder in tqdm(group_a_paths, desc="Loading group A"):
        imgs = load_images_from_folder(folder, max_images=max_images).to(device)
        fid.update(imgs, real=True)

    for folder in tqdm(group_b_paths, desc="Loading group B"):
        imgs = load_images_from_folder(folder, max_images=max_images).to(device)
        fid.update(imgs, real=False)

    return fid.compute().item()

if __name__ == "__main__":
    args = parse_args()
    groups = collect_by_fusion_type(args.base_dir)
    print("Available fusion types:", list(groups.keys()))
    a, b = args.fusion_type_a, args.fusion_type_b
    if a in groups and b in groups:
        fid_score = compute_fid_between_groups(groups[a], groups[b], max_images=args.max_images, device=args.device)
        print(f"FID between fusion types '{a}' and '{b}': {fid_score:.4f}")
    else:
        print(f"Both fusion types '{a}' and '{b}' must be present.")



