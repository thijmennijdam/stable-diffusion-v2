import torch
import clip
from PIL import Image
import argparse
import os
import json
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Base path to search for experiment folders"
    )
    parser.add_argument(
        "--fusion_type",
        type=str,
        required=True,
        choices=["alpha_blend", "cross_attention", "concat"],
        help="Fusion type to filter folders by"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Alpha value to filter by (only used for alpha_blend fusion type)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run CLIP on"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output JSON file to save results (optional)"
    )
    return parser.parse_args()

def load_config(config_path):
    """Load config.json file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def compute_clip_scores(clip_model, preprocess, sample_path, prompt, ref_img_path, device):
    """Compute both text-to-image and image-to-image CLIP scores for a single sample"""
    try:
        # Load and preprocess sample image
        sample_image = Image.open(sample_path).convert("RGB")
        sample_input = preprocess(sample_image).unsqueeze(0).to(device)

        with torch.no_grad():
            sample_features = clip_model.encode_image(sample_input)
            
            # Compute text-to-image CLIP score
            text_input = clip.tokenize([prompt]).to(device)
            text_features = clip_model.encode_text(text_input)
            text_to_image_score = torch.nn.functional.cosine_similarity(sample_features, text_features).item()
            
            # Compute image-to-image CLIP score
            ref_image = Image.open(ref_img_path).convert("RGB")
            ref_input = preprocess(ref_image).unsqueeze(0).to(device)
            ref_features = clip_model.encode_image(ref_input)
            image_to_image_score = torch.nn.functional.cosine_similarity(sample_features, ref_features).item()
            
            # Compute average of both scores
            avg_score = (text_to_image_score + image_to_image_score) / 2.0

        return {
            'text_to_image': text_to_image_score,
            'image_to_image': image_to_image_score,
            'average': avg_score
        }
    except Exception as e:
        print(f"Error computing CLIP scores for {sample_path}: {e}")
        return None

def main():
    args = parse_args()
    
    # Load CLIP model
    print("Loading CLIP model (ViT-B/32)...")
    clip_model, preprocess = clip.load("ViT-B/32", device=args.device)
    
    # Find all experiment folders
    experiment_folders = []
    for root, dirs, files in os.walk(args.base_path):
        if "config.json" in files:
            experiment_folders.append(root)
    
    print(f"Found {len(experiment_folders)} experiment folders")
    
    # Filter folders based on criteria
    matching_folders = []
    for folder in experiment_folders:
        config_path = os.path.join(folder, "config.json")
        config = load_config(config_path)
        
        if config is None:
            continue
            
        # Check fusion type
        if config.get("fusion_type") != args.fusion_type:
            continue
            
        # Check alpha if specified and fusion type is alpha_blend
        if args.fusion_type == "alpha_blend" and args.alpha is not None:
            if abs(config.get("ref_blend_weight", 0) - args.alpha) > 1e-6:
                continue
        
        matching_folders.append((folder, config))
    
    print(f"Found {len(matching_folders)} matching folders")
    
    if not matching_folders:
        print("No matching folders found!")
        return
    
    # Initialize score tracking
    all_text_to_image_scores = []
    all_image_to_image_scores = []
    all_average_scores = []
    
    results_data = {
        'evaluation_params': {
            'fusion_type': args.fusion_type,
            'alpha': args.alpha,
            'base_path': args.base_path
        },
        'folder_results': [],
        'summary': {}
    }
    
    for folder, config in matching_folders:
        print(f"\nProcessing folder: {folder}")
        
        # Get prompt and reference image from config
        prompt = config.get("prompt")
        ref_img_path = config.get("ref_img")
        
        if not prompt:
            print(f"No prompt found in config for {folder}")
            continue
            
        if not ref_img_path or not os.path.exists(ref_img_path):
            print(f"Reference image not found for {folder}: {ref_img_path}")
            continue
            
        print(f"Using prompt: '{prompt}'")
        print(f"Using reference image: '{os.path.basename(ref_img_path)}'")
        
        # Find all sample images
        samples_folder = os.path.join(folder, "samples")
        if not os.path.exists(samples_folder):
            print(f"Samples folder not found: {samples_folder}")
            continue
            
        sample_files = glob.glob(os.path.join(samples_folder, "*.png")) + \
                      glob.glob(os.path.join(samples_folder, "*.jpg"))
        
        if not sample_files:
            print(f"No sample images found in {samples_folder}")
            continue
            
        print(f"Found {len(sample_files)} sample images")
        
        # Calculate CLIP scores for each sample
        folder_text_to_image_scores = []
        folder_image_to_image_scores = []
        folder_average_scores = []
        sample_details = []
        
        for sample_path in sample_files:
            scores = compute_clip_scores(clip_model, preprocess, sample_path, prompt, ref_img_path, args.device)
            if scores is not None:
                folder_text_to_image_scores.append(scores['text_to_image'])
                folder_image_to_image_scores.append(scores['image_to_image'])
                folder_average_scores.append(scores['average'])
                
                sample_name = os.path.basename(sample_path)
                sample_details.append({
                    'sample_name': sample_name,
                    'text_to_image_clip': scores['text_to_image'],
                    'image_to_image_clip': scores['image_to_image'],
                    'average_clip': scores['average']
                })
                
                print(f"  {sample_name}: Text-to-Image={scores['text_to_image']:.4f}, Image-to-Image={scores['image_to_image']:.4f}, Average={scores['average']:.4f}")
        
        if folder_average_scores:
            avg_text_to_image = sum(folder_text_to_image_scores) / len(folder_text_to_image_scores)
            avg_image_to_image = sum(folder_image_to_image_scores) / len(folder_image_to_image_scores)
            avg_combined = sum(folder_average_scores) / len(folder_average_scores)
            
            print(f"  Folder Averages: Text-to-Image={avg_text_to_image:.4f}, Image-to-Image={avg_image_to_image:.4f}, Combined={avg_combined:.4f}")
            
            # Add to overall scores
            all_text_to_image_scores.extend(folder_text_to_image_scores)
            all_image_to_image_scores.extend(folder_image_to_image_scores)
            all_average_scores.extend(folder_average_scores)
            
            # Store results for this folder
            folder_info = {
                'folder': folder,
                'folder_name': os.path.basename(folder),
                'fusion_type': config.get('fusion_type'),
                'alpha': config.get('ref_blend_weight'),
                'prompt': prompt,
                'ref_img': ref_img_path,
                'ref_img_name': os.path.basename(ref_img_path),
                'num_samples': len(folder_average_scores),
                'avg_text_to_image_clip': avg_text_to_image,
                'avg_image_to_image_clip': avg_image_to_image,
                'avg_combined_clip': avg_combined,
                'min_text_to_image_clip': min(folder_text_to_image_scores),
                'max_text_to_image_clip': max(folder_text_to_image_scores),
                'min_image_to_image_clip': min(folder_image_to_image_scores),
                'max_image_to_image_clip': max(folder_image_to_image_scores),
                'min_combined_clip': min(folder_average_scores),
                'max_combined_clip': max(folder_average_scores),
                'sample_details': sample_details
            }
            results_data['folder_results'].append(folder_info)
    
    # Print overall statistics and save results
    if all_average_scores:
        overall_avg_text_to_image = sum(all_text_to_image_scores) / len(all_text_to_image_scores)
        overall_avg_image_to_image = sum(all_image_to_image_scores) / len(all_image_to_image_scores)
        overall_avg_combined = sum(all_average_scores) / len(all_average_scores)
        
        print(f"\n=== Overall Statistics ===")
        print(f"Total samples processed: {len(all_average_scores)}")
        print(f"Overall average Text-to-Image CLIP: {overall_avg_text_to_image:.4f}")
        print(f"Overall average Image-to-Image CLIP: {overall_avg_image_to_image:.4f}")
        print(f"Overall average Combined CLIP: {overall_avg_combined:.4f}")
        print(f"Min Text-to-Image CLIP: {min(all_text_to_image_scores):.4f}")
        print(f"Max Text-to-Image CLIP: {max(all_text_to_image_scores):.4f}")
        print(f"Min Image-to-Image CLIP: {min(all_image_to_image_scores):.4f}")
        print(f"Max Image-to-Image CLIP: {max(all_image_to_image_scores):.4f}")
        print(f"Min Combined CLIP: {min(all_average_scores):.4f}")
        print(f"Max Combined CLIP: {max(all_average_scores):.4f}")
        
        # Add summary statistics to results
        results_data['summary'] = {
            'total_samples': len(all_average_scores),
            'total_folders': len(results_data['folder_results']),
            'overall_avg_text_to_image_clip': overall_avg_text_to_image,
            'overall_avg_image_to_image_clip': overall_avg_image_to_image,
            'overall_avg_combined_clip': overall_avg_combined,
            'overall_min_text_to_image_clip': min(all_text_to_image_scores),
            'overall_max_text_to_image_clip': max(all_text_to_image_scores),
            'overall_min_image_to_image_clip': min(all_image_to_image_scores),
            'overall_max_image_to_image_clip': max(all_image_to_image_scores),
            'overall_min_combined_clip': min(all_average_scores),
            'overall_max_combined_clip': max(all_average_scores),
            'overall_std_text_to_image_clip': float(torch.tensor(all_text_to_image_scores).std().item()) if len(all_text_to_image_scores) > 1 else 0.0,
            'overall_std_image_to_image_clip': float(torch.tensor(all_image_to_image_scores).std().item()) if len(all_image_to_image_scores) > 1 else 0.0,
            'overall_std_combined_clip': float(torch.tensor(all_average_scores).std().item()) if len(all_average_scores) > 1 else 0.0
        }
    else:
        print("No valid samples processed!")
        results_data['summary'] = {
            'total_samples': 0,
            'total_folders': 0,
            'overall_avg_text_to_image_clip': None,
            'overall_avg_image_to_image_clip': None,
            'overall_avg_combined_clip': None,
            'overall_min_text_to_image_clip': None,
            'overall_max_text_to_image_clip': None,
            'overall_min_image_to_image_clip': None,
            'overall_max_image_to_image_clip': None,
            'overall_min_combined_clip': None,
            'overall_max_combined_clip': None,
            'overall_std_text_to_image_clip': None,
            'overall_std_image_to_image_clip': None,
            'overall_std_combined_clip': None
        }
    
    # Save results to JSON file if specified
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")

if __name__ == "__main__":
    main()