from torch import nn

class ImageToTextAligner(nn.Module):
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