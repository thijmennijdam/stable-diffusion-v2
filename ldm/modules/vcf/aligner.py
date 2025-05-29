from torch import nn


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
            nn.Linear(input_dim, input_dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim*2, output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x): 
        x = self.layernorm(x)
        return self.mlp(x)