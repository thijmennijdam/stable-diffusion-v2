from torch import nn

class ImageToTextAlignerV1(nn.Module):
    """
    A neural module that projects image feature embeddings into the text embedding space
    using a small feedforward network with normalization and ReLU activation.
    """

    def __init__(self, input_dim=1280, output_dim=1024):
        """
        Args:
            dim (int): Dimensionality of both image and text embeddings.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        """
        Forward pass to project image embeddings.

        Args:
            x (Tensor): Input tensor of shape [B, ..., D].

        Returns:
            Tensor: Projected tensor of shape [B, ..., D].
        """
        return self.proj(x)

# V2.2
class ImageToTextAlignerV2(nn.Module):
    def __init__(self, input_dim=1280, output_dim=1024, dropout=0.1):
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim*2, output_dim),
            nn.Dropout(dropout)
        )

    def _init_weights(self, m, weight_init):
        if isinstance(m, nn.Linear):
            if weight_init == 'xavier':
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):  # [B, N, 1280]
        x = self.layernorm(x)
        return self.mlp(x)  # [B, N, 1024]
