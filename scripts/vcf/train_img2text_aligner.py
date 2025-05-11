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

load_dotenv()

if not os.getenv("WANDB_API_KEY"):
    raise ValueError("WANDB_API_KEY is not set in the environment.")


class ImageToTextAligner(nn.Module):
    """
    A neural module that projects image feature embeddings into the text embedding space
    using a small feedforward network with normalization and ReLU activation.
    """

    def __init__(self, dim=1024):
        """
        Args:
            dim (int): Dimensionality of both image and text embeddings.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        """
        Forward pass to project image embeddings.

        Args:
            x (Tensor): Input tensor of shape [B, N, D].

        Returns:
            Tensor: Projected tensor of shape [B, N, D].
        """
        return self.proj(x)


def cosine_similarity_loss(
    image_tokens: torch.Tensor,
    text_tokens: torch.Tensor,
    aligner: nn.Module
) -> torch.Tensor:
    """
    Computes the cosine similarity loss between aligned image and text embeddings.

    Args:
        image_tokens (torch.Tensor): Image embeddings [B, N_img_tokens, D].
        text_tokens (torch.Tensor): Text embeddings [B, N_text_tokens, D].
        aligner (nn.Module): ImageToTextAligner module.

    Returns:
        torch.Tensor: Scalar cosine similarity loss.
    """
    # Align image embeddings
    img_repr = aligner(image_tokens)  # [B, D]

    # Mean-pool both
    # img_repr = aligned_img.mean(dim=1)  # [B, D]  # no need to, as the image embeddings are already projected
    text_repr = text_tokens.mean(dim=1)  # [B, D]

    # Normalize
    img_repr = F.normalize(img_repr, dim=1)
    text_repr = F.normalize(text_repr, dim=1)

    # Cosine similarity
    cos_sim = (img_repr * text_repr).sum(dim=1)  # [B]
    loss = 1 - cos_sim.mean()
    return loss


def info_nce_loss(
    image_tokens: torch.Tensor,
    text_tokens: torch.Tensor,
    aligner: nn.Module,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    Computes the InfoNCE loss for contrastive alignment between image and text features.

    Args:
        image_tokens (torch.Tensor): Image embeddings [B, N, D].
        text_tokens (torch.Tensor): Text embeddings [B, N, D].
        aligner (nn.Module): ImageToTextAligner module.
        temperature (float): Temperature scaling factor for softmax.

    Returns:
        torch.Tensor: Scalar InfoNCE loss.
    """
    B, _, D = image_tokens.shape

    # Align and pool
    aligned_img = aligner(image_tokens).mean(dim=1)  # [B, D]
    text_repr = text_tokens.mean(dim=1)  # [B, D]

    # Normalize
    img_norm = F.normalize(aligned_img, dim=1)  # [B, D]
    text_norm = F.normalize(text_repr, dim=1)  # [B, D]

    # Similarity matrix: [B, B]
    logits = img_norm @ text_norm.T / temperature

    # Labels: diagonal is positive pair
    labels = torch.arange(B, device=logits.device)

    # Symmetric InfoNCE loss
    loss_i2t = F.cross_entropy(logits, labels)  # image -> text
    loss_t2i = F.cross_entropy(logits.T, labels)  # text -> image
    return (loss_i2t + loss_t2i) / 2


def prepare_dataloaders(
    dataset_names: list[str],
    batch_size: int = 32
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads and preprocesses datasets, returning DataLoaders for train, val, and test.
    Flickr is split into 80/10/10; COCO uses its built-in splits.

    Args:
        dataset_names (list[str]): List of dataset names to load (e.g., ["flickr30k", "coco"]).
        batch_size (int): Batch size for the DataLoaders.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test DataLoaders.
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
            train_set = load_dataset(dataset_id, split="train")
            val_set = load_dataset(dataset_id, split="validation")
            test_set = load_dataset(dataset_id, split="test")
            for ds in [train_set, val_set, test_set]:
                ds.set_transform(preprocess)
            train_sets.append(train_set)
            val_sets.append(val_set)
            test_sets.append(test_set)

    train_loader = DataLoader(ConcatDataset(train_sets), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ConcatDataset(val_sets), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ConcatDataset(test_sets), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def evaluate_loss(
    dataloader: DataLoader,
    clip_text_encoder: nn.Module,
    clip_image_encoder: nn.Module,
    loss_fn: callable,
    aligner: nn.Module,
    device: str
) -> float:
    """
    Computes average loss over a given dataloader.

    Args:
        dataloader (DataLoader): DataLoader to evaluate.
        clip_text_encoder (nn.Module): Text encoder module.
        clip_image_encoder (nn.Module): Image encoder module.
        loss_fn (callable): Loss function to evaluate.
        aligner (nn.Module): Aligner model to evaluate.
        device (str): Device to use ("cuda" or "cpu").

    Returns:
        float: Average loss over the dataloader.
    """
    aligner.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            texts = batch["text"]
            text_features = clip_text_encoder.encode(texts)
            image_features = clip_image_encoder(images)
            loss = loss_fn(image_features, text_features, aligner)
            total_loss += loss.item()
    aligner.train()
    return total_loss / len(dataloader)


def train_aligner(
    train_loader: DataLoader,
    val_loader: DataLoader,
    clip_text_encoder: nn.Module,
    clip_image_encoder: nn.Module,
    loss_fn: callable,
    args: argparse.Namespace
) -> None:
    """
    Trains the image-to-text aligner model with validation loss logging and best model saving.

    Args:
        train_loader (DataLoader): Training dataloader.
        val_loader (DataLoader): Validation dataloader.
        clip_text_encoder (nn.Module): Text encoder module.
        clip_image_encoder (nn.Module): Image encoder module.
        loss_fn (callable): Loss function to use.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    device = args.device
    save_path = args.model_path
    save_every = args.save_every

    aligner = ImageToTextAligner(dim=1024).to(device)
    optimizer = torch.optim.Adam(aligner.parameters(), lr=args.lr)
    wandb.watch(aligner, log="all")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    best_val_loss = float("inf")
    best_model_path = f"{os.path.splitext(save_path)[0]}_best.pth"

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in progress:
            images = batch["image"].to(device)
            texts = batch["text"]
            text_features = clip_text_encoder.encode(texts)
            image_features = clip_image_encoder(images)
            loss = loss_fn(image_features, text_features, aligner)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = evaluate_loss(val_loader, clip_text_encoder, clip_image_encoder, loss_fn, aligner, device)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(aligner.state_dict(), best_model_path)
            print(f"Best model updated and saved at {best_model_path}")

        if (epoch + 1) % save_every == 0:
            checkpoint_path = f"{os.path.splitext(save_path)[0]}_epoch_{epoch + 1}.pth"
            torch.save(aligner.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

    wandb.log({"best_val_loss": best_val_loss})
    wandb.log({"best_model": wandb.Artifact(best_model_path, type="model")})
    wandb.save(best_model_path)

    # Load and evaluate best model on test set
    aligner.load_state_dict(torch.load(best_model_path))
    final_test_loss = evaluate_loss(test_loader, clip_text_encoder, clip_image_encoder, loss_fn, aligner, args.device)
    print(f"Best Model Test Loss: {final_test_loss:.4f}")
    wandb.log({"final_test_loss": final_test_loss})


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train a text-image aligner using FrozenOpenCLIPEmbedder."
    )
    parser.add_argument("--datasets", type=str, nargs="+", default="flickr30k")
    parser.add_argument("--loss", type=str, default="cosine", choices=["cosine", "infonce"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb_project", type=str, default="text-image-aligner")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="weights/img2text_aligner/model.pth")
    parser.add_argument("--save_every", type=int, default=5, help="Save model every N epochs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.getenv("WANDB_API_KEY"):
        raise ValueError("WANDB_API_KEY is not set in the environment.")
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))

    loss_function = cosine_similarity_loss if args.loss == "cosine" else info_nce_loss

    new_list_datasets = []
    for datset in args.datasets:
        if "+" in datset:
            new_list_datasets.extend(datset.split("+"))
        else:
            new_list_datasets.append(datset)
    print(f"Using datasets: {new_list_datasets}")

    train_loader, val_loader, test_loader = prepare_dataloaders(new_list_datasets, args.batch_size)

    # NOTE: in FrozenOpenCLIPImageEmbedder penultimate layer is not implemented though I think we can use the output as it is (from .visual) and align it
    clip_text_encoder = FrozenOpenCLIPEmbedder(
        device=args.device, layer="penultimate"
    ).to(args.device)

    clip_image_encoder = FrozenOpenCLIPImageEmbedder(device=args.device).to(args.device)

    train_aligner(
        train_loader, val_loader, clip_text_encoder, clip_image_encoder, loss_function, args
    )

    wandb.finish()
