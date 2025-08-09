from torch import nn
import math
from typing import List


class ImageToTextAlignerV1(nn.Module):
    """
    A neural module that projects image feature embeddings into the text embedding space
    using a small feedforward network with normalization and ReLU activation.
    """

    def __init__(self, input_dim=1280, output_dim=1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        return self.proj(x)

class ImageToTextAlignerV1_1(nn.Module):
    """
    A neural module that projects image feature embeddings into the text embedding space
    using a small feedforward network with normalization and ReLU activation.
    """

    def __init__(self, input_dim=1280, output_dim=1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim*2),
            nn.ReLU(),
            nn.Linear(input_dim*2, output_dim),
        )

    def forward(self, x):
        return self.proj(x)


class ImageToTextAlignerV2(nn.Module):
    def __init__(self, input_dim=1280, output_dim=1024, dropout=0.1):
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x): 
        x = self.layernorm(x)
        return self.mlp(x)


class ImageToTextAlignerV3(nn.Module):
    """
    A convolutional aligner that interprets image tokens as a 2D grid and
    applies 3x3 Conv2d layers to project from input_dim to output_dim.

    Expects the number of tokens to be a perfect square (e.g., 16x16 = 256 for ViT-H/14 at 224px).
    """

    def __init__(
        self,
        input_dim: int = 1280,
        output_dim: int = 1024,
        conv_hidden_dim: int = 1024,
        num_conv_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(input_dim)

        conv_layers: List[nn.Module] = []
        in_channels = input_dim
        # Build num_conv_layers-1 hidden 3x3 conv blocks
        for _ in range(max(0, num_conv_layers - 1)):
            conv_layers.extend([
                nn.Conv2d(in_channels, conv_hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout2d(p=dropout),
            ])
            in_channels = conv_hidden_dim
        # Final projection to output_dim
        conv_layers.append(nn.Conv2d(in_channels, output_dim, kernel_size=3, padding=1))

        self.conv_stack = nn.Sequential(*conv_layers)

    def forward(self, x):
        # x: [B, N_tokens, input_dim]
        batch_size, num_tokens, input_dim = x.shape
        x = self.token_norm(x)

        grid_size = int(math.sqrt(num_tokens))
        if grid_size * grid_size != num_tokens:
            raise ValueError(
                f"Conv aligner requires square number of tokens, got {num_tokens}."
            )

        # [B, N, C] -> [B, C, H, W]
        x = x.view(batch_size, grid_size, grid_size, input_dim).permute(0, 3, 1, 2).contiguous()
        y = self.conv_stack(x)
        # [B, C_out, H, W] -> [B, N, C_out]
        y = y.permute(0, 2, 3, 1).contiguous().view(batch_size, num_tokens, -1)
        return y