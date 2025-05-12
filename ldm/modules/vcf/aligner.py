from torch import nn

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
            x (Tensor): Input tensor of shape [B, ..., D].

        Returns:
            Tensor: Projected tensor of shape [B, ..., D].
        """
        return self.proj(x)