import argparse
import os
import wandb
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from datasets import load_dataset
from ldm.modules.encoders.modules import (
    FrozenOpenCLIPEmbedder,
    FrozenOpenCLIPImageEmbedder,
)
from dotenv import load_dotenv
load_dotenv()


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


def cosine_similarity_loss(image_tokens, text_tokens, aligner):
    """
    Computes the cosine similarity loss between aligned image and text embeddings.

    Args:
        image_tokens (Tensor): Image embeddings [B, N_img_tokens, D].
        text_tokens (Tensor): Text embeddings [B, N_text_tokens, D].
        aligner (nn.Module): ImageToTextAligner module.

    Returns:
        Tensor: Scalar cosine similarity loss.
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


def info_nce_loss(image_tokens, text_tokens, aligner, temperature=0.07):
    """
    Computes the InfoNCE loss for contrastive alignment between image and text features.

    Args:
        image_tokens (Tensor): Image embeddings [B, N, D].
        text_tokens (Tensor): Text embeddings [B, N, D].
        aligner (nn.Module): ImageToTextAligner module.
        temperature (float): Temperature scaling factor for softmax.

    Returns:
        Tensor: Scalar InfoNCE loss.
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


def prepare_dataloader(dataset_names, batch_size=32):
    """
    Loads and preprocesses datasets, returning a DataLoader.

    Args:
        dataset_names (List[str]): List of HuggingFace dataset names to load.
        batch_size (int): Batch size for DataLoader.

    Returns:
        DataLoader: Batched data loader of combined datasets.
    """
    datasets = []
    transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            # NOTE: no need for normalization since it's done in FrozenOpenCLIPImageEmbedder
            # transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ]
    )

    dataset_name_dict = {
        "flickr30k": "lmms-lab/flickr30k",
        "coco": "jxie/coco_captions",
    }
    # TODO: loading/parsing the dataset might need adustments; also maybe let's add val dataset and log the metrics 

    def preprocess(batch):
        images = [transform(img.convert("RGB")) for img in batch["image"]]
        
        # each image apparantly has multiple captions, so we randomly select one
        # NOTE: we can think of something better
        texts = [
            random.choice(captions) if isinstance(captions, (list, tuple)) else captions
            for captions in batch["caption"]
        ]
        
        return {"image": images, "text": texts}

    for dataset_name in dataset_names:
        dataset = load_dataset(dataset_name_dict[dataset_name], split="test")
        dataset.set_transform(preprocess)
        datasets.append(dataset)

    combined_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train_aligner(dataloader, clip_text_encoder, clip_image_encoder, loss_fn, args):
    """
    Trains the image-to-text aligner model.

    Args:
        dataloader (DataLoader): Training data.
        clip_text_encoder (FrozenOpenCLIPEmbedder): Pretrained text encoder.
        clip_image_encoder (FrozenOpenCLIPImageEmbedder): Pretrained image encoder.
        loss_fn (callable): Loss function (cosine similarity or InfoNCE).
        args (argparse.Namespace): Parsed arguments.

    Returns:
        nn.Module: Trained ImageToTextAligner model.
    """
    device = args.device
    save_path = args.model_path
    save_every = args.save_every

    aligner = ImageToTextAligner(dim=1024).to(device)
    optimizer = torch.optim.Adam(aligner.parameters(), lr=args.lr)
    wandb.watch(aligner, log="all")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
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

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "loss": avg_loss})

        if (epoch + 1) % save_every == 0:
            checkpoint_path = f"{os.path.splitext(save_path)[0]}_epoch_{epoch + 1}.pth"
            torch.save(aligner.state_dict(), checkpoint_path)
            print(f"Model saved at {checkpoint_path}")

    return aligner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a text-image aligner using FrozenOpenCLIPEmbedder."
    )
    parser.add_argument("--datasets", type=str, nargs="+", default=["flickr30k"])
    parser.add_argument(
        "--loss", type=str, default="cosine", choices=["cosine", "infonce"]
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb_project", type=str, default="text-image-aligner")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument(
        "--model_path", type=str, default="weights/img2text_aligner/model.pth"
    )
    parser.add_argument(
        "--save_every", type=int, default=5, help="Save model every N epochs"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(
        project=args.wandb_project, entity=args.wandb_entity, config=vars(args)
    )

    loss_function = cosine_similarity_loss if args.loss == "cosine" else info_nce_loss
    dataloader = prepare_dataloader(args.datasets, args.batch_size)

    # NOTE: verify (default settings except for penultimate layer which is the same as in our config)
    # NOTE: in FrozenOpenCLIPImageEmbedder penultimate layer is not implemented though I think we can use the output as it is (from .visual) and align it
    clip_text_encoder = FrozenOpenCLIPEmbedder(
        device=args.device, layer="penultimate"
    ).to(args.device)
       
    clip_image_encoder = FrozenOpenCLIPImageEmbedder(device=args.device).to(args.device)
    
    aligner = train_aligner(
        dataloader, clip_text_encoder, clip_image_encoder, loss_function, args
    )
    print("Training complete. Saving final model.")
    torch.save(aligner.state_dict(), args.model_path)
    print(f"Final model saved at {args.model_path}")
    wandb.log({"final_model": wandb.Artifact(args.model_path, type="model")})
    wandb.save(args.model_path)
    print("Model saved to wandb.")
    wandb.finish()
