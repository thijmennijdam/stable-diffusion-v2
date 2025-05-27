import torch
import clip
from PIL import Image
import argparse
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the generated image"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt used to generate the image"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run CLIP on"
    )
    return parser.parse_args()

def compute_clip_score(image_path, prompt, device):
    # Load CLIP model
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Tokenise prompt
    text_input = clip.tokenize([prompt]).to(device)

    # Compute similarity
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
        score = torch.nn.functional.cosine_similarity(image_features, text_features).item()

    print(f"CLIP score between image and prompt \"{prompt}\": {score:.4f}")
    return score

if __name__ == "__main__":
    args = parse_args()
    compute_clip_score(args.image_path, args.prompt, args.device)
