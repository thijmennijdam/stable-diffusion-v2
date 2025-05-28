import argparse
import os
import wandb
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import transforms
from datasets import load_dataset
from ldm.modules.encoders.modules import (
    FrozenOpenCLIPEmbedder,
    FrozenOpenCLIPImageEmbedder,
)
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from typing import List, Tuple
from ldm.modules.vcf.aligner import ImageToTextAlignerV1, ImageToTextAlignerV2
import numpy as np
import math
import sys

load_dotenv()

if not os.getenv("WANDB_API_KEY"):
    raise ValueError("WANDB_API_KEY is not set in the environment.")

class InfoNCELoss(nn.Module):
    def __init__(self, init_tau=0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / init_tau), dtype=torch.float32))

    def forward(self, mapped_img_tokens, text_tokens):
        # mean-pool
        img_repr = F.normalize(mapped_img_tokens.mean(dim=1), dim=1)  # [B, 1024]
        text_repr = F.normalize(text_tokens.mean(dim=1), dim=1)       # [B, 1024]

        logits = torch.matmul(img_repr, text_repr.T) * self.logit_scale.exp()
        labels = torch.arange(logits.shape[0], device=logits.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2
        
class CrossAttentionAlignLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigma = 1.0  # Hyperparameter for Gaussian kernel, can be tuned

    def forward(self, img_tokens, text_tokens):
        B, N_img, D = img_tokens.shape
        # Compute attention from image to text
        attn_scores = torch.matmul(img_tokens, text_tokens.transpose(-2, -1))  # [B, N_img, 77]
        attn_scores = attn_scores / math.sqrt(D)
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, N_img, 77], normalize over image tokens

        # Reconstruct each text token as weighted sum over image tokens
        recon_text = torch.matmul(attn_weights.transpose(1, 2), img_tokens)  # [B, 77, D]

        # Compute token-wise alignment loss
        loss = F.mse_loss(recon_text, text_tokens)
        return loss

def prepare_dataloaders(
    dataset_names: List[str],
    batch_size: int = 32,
    num_workers: int = 8
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads and preprocesses datasets, returning DataLoaders for train, val, and test.
    Flickr is split into 80/10/10; COCO uses its built-in splits.

    Args:
        dataset_names (List[str]): List of dataset names to load (e.g., ["flickr30k", "coco"]).
        batch_size (int): Batch size for the DataLoaders.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test DataLoaders.
    """
    dataset_name_dict = {
        "flickr30k": "lmms-lab/flickr30k",
        "coco": "jxie/coco_captions",
    }

    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])

    def preprocess(batch):
        images = [transform(img.convert("RGB")) for img in batch["image"]]
        texts = [
            random.choice(captions) if isinstance(captions, (list, tuple)) else captions
            for captions in batch["caption"]
        ]
        return {"image": images, "text": texts}

    train_sets, val_sets, test_sets = [], [], []

    for dataset_name in dataset_names:
        dataset_id = dataset_name_dict[dataset_name]

        if dataset_name == "flickr30k":
            full_dataset = load_dataset(dataset_id, split="test")
            full_dataset.set_transform(preprocess)
            total_len = len(full_dataset)
            indices = list(range(total_len))
            train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
            val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
            train_sets.append(Subset(full_dataset, train_idx))
            val_sets.append(Subset(full_dataset, val_idx))
            test_sets.append(Subset(full_dataset, test_idx))

        elif dataset_name == "coco":
            
            subset_ratio = 0.1  # 10% of the dataset since COCO is 616,767 images
            seed = 42  # for reproducibility

            def subsample(dataset, ratio=subset_ratio, seed=seed):
                total_len = len(dataset)
                num_samples = int(total_len * ratio)
                indices = random.Random(seed).sample(range(total_len), num_samples)
                return Subset(dataset, indices)

            full_train = load_dataset(dataset_id, split="train")
            full_val = load_dataset(dataset_id, split="validation")
            full_test = load_dataset(dataset_id, split="test")

            # Apply preprocessing first so Subset works properly
            for ds in [full_train, full_val, full_test]:
                ds.set_transform(preprocess)

            train_sets.append(subsample(full_train))
            val_sets.append(subsample(full_val))
            test_sets.append(subsample(full_test))


    train_loader = DataLoader(
        ConcatDataset(train_sets), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        ConcatDataset(val_sets), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        ConcatDataset(test_sets), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader

def evaluate_loss(
    dataloader: DataLoader,
    clip_text_encoder: nn.Module,
    clip_image_encoder: nn.Module,
    loss_fn: nn.Module,
    aligner: nn.Module,
    device: str,
    loss_type: str = "cosine",
    infonce_lambda: float = 1.0
) -> float:
    """
    Computes average loss over a given dataloader.
    """
    aligner.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            texts = batch["text"]
            text_features = clip_text_encoder.encode(texts)
            image_features = clip_image_encoder(images)

            if loss_type == "combined":
                mapped_img_tokens = aligner(image_features)
                infonce_loss = loss_fn["infonce"](mapped_img_tokens, text_features)
                cross_attention_loss = loss_fn["cross_attention"](mapped_img_tokens, text_features)
                loss = infonce_lambda * infonce_loss + cross_attention_loss
            elif loss_type == "infonce":
                mapped_img_tokens = aligner(image_features)
                loss = loss_fn(mapped_img_tokens, text_features)
            elif loss_type == "cross_attention":
                mapped_img_tokens = aligner(image_features)
                loss = loss_fn(mapped_img_tokens, text_features)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            total_loss += loss.item()
    aligner.train()
    return total_loss / len(dataloader)


def train_aligner(
    train_loader: DataLoader,
    val_loader: DataLoader,
    clip_text_encoder: nn.Module,
    clip_image_encoder: nn.Module,
    loss_fn: nn.Module,
    args: argparse.Namespace,
) -> None:
    """
    Trains the image-to-text aligner model with validation loss logging and best model saving.
    """
    device = args.device
    datasets_name = "+".join(args.datasets) if len(args.datasets) > 1 else args.datasets[0]
    save_path = f"weights/aligner_models/version_{args.version}/dataset_{datasets_name}/loss_{args.loss}/batch_{args.batch_size}/model.pth"
    save_every = args.save_every

    if args.version == "v1":
        aligner = ImageToTextAlignerV1(input_dim=1280, output_dim=1024).to(device)
    elif args.version == "v2":
        aligner = ImageToTextAlignerV2(input_dim=1280, output_dim=1024).to(device)
    else:
        raise ValueError(f"Unknown aligner version: {args.version}")

    if args.resume_from and os.path.isfile(args.resume_from):
        print(f"Loading pretrained aligner from {args.resume_from}")
        aligner.load_state_dict(torch.load(args.resume_from, map_location=device))
    
    wandb.watch(aligner, log="all")

    optimizer = torch.optim.AdamW(aligner.parameters(), lr=args.lr, weight_decay=0.01)
    T_max = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    best_val_loss = float("inf")
    best_model_path = f"{os.path.splitext(save_path)[0]}_best.pth"
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        step = 0
        for batch in progress:
            images = batch["image"].to(device)
            texts = batch["text"]
            text_features = clip_text_encoder.encode(texts)
            image_features = clip_image_encoder(images)

            optimizer.zero_grad()
            if args.loss == "combined":
                mapped_img_tokens = aligner(image_features)
                infonce_loss = loss_fn["infonce"](mapped_img_tokens, text_features)
                cross_attention_loss = loss_fn["cross_attention"](mapped_img_tokens, text_features)
                loss = args.infonce_lambda * infonce_loss + cross_attention_loss
            elif args.loss == "infonce":
                mapped_img_tokens = aligner(image_features)
                loss = loss_fn(mapped_img_tokens, text_features)
            elif args.loss == "cross_attention":
                mapped_img_tokens = aligner(image_features)
                loss = loss_fn(mapped_img_tokens, text_features)
            else:
                raise ValueError(f"Unknown loss type: {args.loss}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(aligner.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            log_dict = {
                "lr": current_lr,
                "epoch": epoch + 1,
                "step": step + 1 + epoch * len(train_loader)
            }

            if args.loss == "combined":
                log_dict["train_infonce_loss_step"] = infonce_loss.item()
                log_dict["train_cross_attention_loss_step"] = cross_attention_loss.item()
                log_dict["train_loss_step"] = loss.item()
            elif args.loss == "infonce":
                log_dict["train_loss_step"] = loss.item()
            elif args.loss == "cross_attention":
                log_dict["train_loss_step"] = loss.item()
            else:
                raise ValueError(f"Unknown loss type: {args.loss}")

            wandb.log(log_dict)

            epoch_loss += loss.item()
            progress.set_postfix(loss=loss.item())
            
            step += 1
            
            
        avg_train_loss = epoch_loss / len(train_loader)
        if args.loss == "combined":
            avg_val_loss = evaluate_loss(val_loader, clip_text_encoder, clip_image_encoder, loss_fn, aligner, device, args.loss, args.infonce_lambda)
        else:
            avg_val_loss = evaluate_loss(val_loader, clip_text_encoder, clip_image_encoder, loss_fn, aligner, device, args.loss)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(aligner.state_dict(), best_model_path)
            print(f"Best model updated and saved at {best_model_path}")
        if (epoch + 1) % save_every == 0:
            checkpoint_path = f"{os.path.splitext(save_path)[0]}_epoch_{epoch + 1}.pth"
            torch.save(aligner.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

    wandb.log({"best_val_loss": best_val_loss})
    best_model_path_filename = os.path.basename(best_model_path)
    artifact = wandb.Artifact(best_model_path_filename, type="model")
    artifact.add_file(best_model_path)
    wandb.log_artifact(artifact)
    aligner.load_state_dict(torch.load(best_model_path))
    final_test_loss = evaluate_loss(test_loader, clip_text_encoder, clip_image_encoder, loss_fn, aligner, device, args.loss, args.infonce_lambda)
    print(f"Best Model Test Loss: {final_test_loss:.4f}")
    wandb.log({"final_test_loss": final_test_loss})


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train a text-image aligner using FrozenOpenCLIPEmbedder."
    )
    parser.add_argument("--datasets", nargs="+", default="[flickr30k]")
    parser.add_argument("--loss", type=str, default="cross_attention", choices=["infonce", "cross_attention", "combined"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb_project", type=str, default="text-image-aligner")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=5, help="Save model every N epochs")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a pretrained aligner checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--version", type=str, default="v2", choices=["v1", "v2"], help="Aligner model version (v1 or v2)")
    parser.add_argument("--infonce_lambda", type=float, default=0.2, help="Weight for InfoNCE loss in combined loss")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    if not os.getenv("WANDB_API_KEY"):
        raise ValueError("WANDB_API_KEY is not set in the environment.")
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    project_name = os.getenv("WANDB_PROJECT", args.wandb_project)
    entity_name = os.getenv("WANDB_ENTITY", args.wandb_entity)
    datasets_name = "+".join(args.datasets) if len(args.datasets) > 1 else args.datasets[0]
    wandb.init(
        project=project_name,
        entity=entity_name,
        config=vars(args),
        name=f"aligner{args.version}-{datasets_name}-{args.loss}-{args.epochs}epochs-{args.batch_size}bs-{args.lr}lr{f'-lambda{args.infonce_lambda}' if args.loss == 'combined' else ''}"
    )

    loss_function = None
    if args.loss == "cross_attention":
        loss_function = CrossAttentionAlignLoss()
    elif args.loss == "infonce":
        loss_function = InfoNCELoss()
    elif args.loss == "combined":
        loss_function = {
            "infonce": InfoNCELoss(),
            "cross_attention": CrossAttentionAlignLoss()
        }
    else:
        raise ValueError(f"Unknown loss type: {args.loss}")

    new_list_datasets = []
    for datset in args.datasets:
        if "+" in datset:
            new_list_datasets.extend(datset.split("+"))
        else:
            new_list_datasets.append(datset)
    print(f"Using datasets: {new_list_datasets}")
    train_loader, val_loader, test_loader = prepare_dataloaders(new_list_datasets, args.batch_size)
    clip_text_encoder = FrozenOpenCLIPEmbedder(
        device=args.device, layer="penultimate"
    ).to(args.device)
    clip_image_encoder = FrozenOpenCLIPImageEmbedder(device=args.device).to(args.device)
    train_aligner(
        train_loader, val_loader, clip_text_encoder, clip_image_encoder, loss_function, args
    )
    wandb.finish()