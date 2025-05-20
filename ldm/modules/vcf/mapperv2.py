from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

class ImageToTextMapper(nn.Module):
    def __init__(self, input_dim=1280, output_dim=1024):
        super().__init__()
        self.mapper = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):  # [B, N, 1280]
        return self.mapper(x)  # [B, N, 1024]


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