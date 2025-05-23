from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import math

import torch
from torch import nn
import torch.nn.functional as F


# V2.2
class ImageToTextAligner(nn.Module):
    def __init__(self, input_dim=1280, output_dim=1024, dropout=0.1, weight_init='default'):
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim*2, output_dim),
            nn.Dropout(dropout)
        )
        if weight_init == 'xavier':
            self.apply(lambda m: self._init_weights(m, weight_init))

    def _init_weights(self, m, weight_init):
        if isinstance(m, nn.Linear):
            if weight_init == 'xavier':
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):  # [B, N, 1280]
        x = self.layernorm(x)
        return self.mlp(x)  # [B, N, 1024]

# TODOS
# - check Max timesteps should actually be correctly set
# - also check other optimizer

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