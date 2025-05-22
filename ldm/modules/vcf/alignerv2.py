from torch import nn
import torch
import torch.nn.functional as F
import numpy as np


import torch
from torch import nn
import torch.nn.functional as F

# V2.3
class ImageToTextAligner(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        output_dim: int = 1024,
        dropout: float = 0.0,
        num_heads: int = 4,
        weight_init: str = 'default'
    ):
        super().__init__()

        # 1) cross-attention: image queries attend to text keys/values
        self.norm_attn = nn.LayerNorm(input_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.drop_attn = nn.Dropout(dropout)

        # 2) MLP to project into text space
        self.norm_mlp = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, output_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        img_tokens: torch.Tensor,    # [B, N_img, input_dim]
        txt_tokens: torch.Tensor     # [B, N_txt, input_dim]
    ) -> torch.Tensor:
        # --- cross-attention + residual
        q = self.norm_attn(img_tokens)
        attn_out, _ = self.cross_attn(q, txt_tokens, txt_tokens)
        img_tokens = img_tokens + self.drop_attn(attn_out)

        # --- MLP projection
        x = self.norm_mlp(img_tokens)
        return self.mlp(x)  # [B, N_img, output_dim]


# V2.2
# class ImageToTextAligner(nn.Module):
#     def __init__(self, input_dim=1280, output_dim=1024, dropout=0.0, weight_init='default'):
#         super().__init__()
#         self.layernorm = nn.LayerNorm(input_dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(input_dim, input_dim*2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(input_dim*2, output_dim),
#             nn.Dropout(dropout)
#         )
#         if weight_init == 'xavier':
#             self.apply(lambda m: self._init_weights(m, weight_init))

#     def _init_weights(self, m, weight_init):
#         if isinstance(m, nn.Linear):
#             if weight_init == 'xavier':
#                 nn.init.xavier_normal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#     def forward(self, x):  # [B, N, 1280]
#         x = self.layernorm(x)
#         return self.mlp(x)  # [B, N, 1024]

# V2.1
# class ImageToTextAligner(nn.Module):
#     def __init__(self, input_dim=1280, output_dim=1024, dropout=0.0, weight_init='default'):
#         super().__init__()

#         self.layernorm_in = nn.LayerNorm(input_dim)
#         self.input_proj = nn.Linear(input_dim, output_dim)

#         self.res_block = nn.Sequential(
#             nn.LayerNorm(output_dim),
#             nn.Linear(output_dim, output_dim * 2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(output_dim * 2, output_dim),
#             nn.Dropout(dropout)
#         )

#         if weight_init == 'xavier':
#             self.apply(lambda m: self._init_weights(m, weight_init))

#     def _init_weights(self, m, weight_init):
#         if isinstance(m, nn.Linear):
#             if weight_init == 'xavier':
#                 nn.init.xavier_normal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#     def forward(self, x):  # [B, N_img_tokens, 1280]
#         x = self.layernorm_in(x)
#         x = self.input_proj(x)  # [B, N, 1024]
#         return x + self.res_block(x)  # [B, N, 1024], residual connection

# V2
# class ImageToTextAligner(nn.Module):
#     def __init__(self, input_dim=1280, output_dim=1024, dropout=0.0, weight_init='default'):
#         super().__init__()
#         self.layernorm = nn.LayerNorm(input_dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(input_dim, input_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(input_dim, output_dim),
#             nn.Dropout(dropout)
#         )
#         if weight_init == 'xavier':
#             self.apply(lambda m: self._init_weights(m, weight_init))

#     def _init_weights(self, m, weight_init):
#         if isinstance(m, nn.Linear):
#             if weight_init == 'xavier':
#                 nn.init.xavier_normal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#     def forward(self, x):  # [B, N, 1280]
#         x = self.layernorm(x)
#         return self.mlp(x)  # [B, N, 1024]