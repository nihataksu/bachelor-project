import numpy as np
import torch
import torch.nn as nn
from scipy import spatial


def hilbert_curve(order):
    """Generate Hilbert curve coordinates for a given order."""
    from scipy.spatial import distance

    def hilbert_distance(x, y):
        return distance.cdist([x], [y], "cityblock")

    N = 2**order
    iter_range = np.arange(N**2)
    x, y = np.indices((N, N)).reshape(2, -1)
    x, y = x[iter_range], y[iter_range]

    hilbert_argsort = np.argsort(
        spatial.HilbertCurve(N, 2).points_to_coords(iter_range)
    )
    return x[hilbert_argsort], y[hilbert_argsort]


class HilbertPatchedEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, order, dropout, in_channels):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        x, y = hilbert_curve(order)
        self.patch_indices = (x, y)
        self.num_patches = len(x)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim), requires_grad=True
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        patches = []
        for i in range(self.num_patches):
            x_idx = self.patch_indices[0][i] * self.patch_size
            y_idx = self.patch_indices[1][i] * self.patch_size
            if x_idx + self.patch_size <= height and y_idx + self.patch_size <= width:
                patch = x[
                    :,
                    :,
                    x_idx : x_idx + self.patch_size,
                    y_idx : y_idx + self.patch_size,
                ]
                patches.append(patch.reshape(batch_size, -1))
        patches = torch.stack(patches, dim=1)
        patches = patches.permute(
            0, 2, 1
        )  # Rearrange to (batch_size, embed_dim, num_patches)

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, patches), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x


import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
        super().__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            nn.Flatten(2),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)
        self.position_embeddings = self.create_position_embeddings(
            num_patches, embed_dim
        )
        self.dropout = nn.Dropout(p=dropout)

    def create_position_embeddings(self, num_patches, embed_dim):
        """Creates positional embeddings using sine and cosine functions."""
        position_embeddings = torch.zeros(num_patches + 1, embed_dim)
        position = torch.arange(0, num_patches + 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        position_embeddings[:, 0::2] = torch.sin(position * div_term)
        position_embeddings[:, 1::2] = torch.cos(position * div_term)

        position_embeddings = position_embeddings.unsqueeze(0)  # Add batch dimension.
        return nn.Parameter(position_embeddings, requires_grad=False)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.patcher(x).permute(0, 2, 1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x


# Assuming you have the other necessary imports and setups in your environment.
