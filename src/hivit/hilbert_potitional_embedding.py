from .position_embedding_stategy import PositionalEmbeddingStrategy
import torch
from torch import nn
from .hilbert_indicies import HilbertIndices
import numpy as np


class PatchEmbeddingHilbertPositionalEmbedding(PositionalEmbeddingStrategy):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels, order):
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

        # Hilbert curve positional embeddings
        hilbert_indices = HilbertIndices(num_patches)
        position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, 2 * embed_dim, 2).float()
            * (-np.log(10000.0) / (2 * embed_dim))
        )

        # print("Embed dim", embed_dim)

        # Adjust the shape of pe to match the size of position * div_term

        # print("Number of patches", num_patches)
        pe = torch.zeros(num_patches, embed_dim)
        # assert pe.size() == torch.Size([16, 49]), "Size of pe tensor does not match expected size [16, 49]"

        # print("Size of hilbert indicies: ", hilbert_indices.size)
        # print("Size of pe tensor before assignment:", pe.size())
        # print("Size of position * div_term tensor:", (position * div_term).size())

        pe = pe[hilbert_indices]  # reorder according to Hilbert curve

        # Adds the sin and cos values for each alternating collumn to add positional embeddings
        pe[:, 0::2] = torch.sin(position * div_term)[:, 0::2]
        pe[:, 1::2] = torch.cos(position * div_term)[:, 1::2]
        # print("Size of pe tensor after assignment:", pe.size())

        # assert pe.size() == torch.Size([NUM_PATCHES, PATCH_SIZE**2])

        # Remove the cls token from positional embeddings
        self.position_embeddings = nn.Parameter(pe[:], requires_grad=False)

        # print("Size of pe tensor after assignment AAAAA:", pe.size())
        # print("Pos embed aaaaaaaa", self.position_embeddings.size())

        # Add cls token separately
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.patcher(x).permute(0, 2, 1)

        # print("Shape of x:", x.shape)
        # print("Shape of x after cat:", x.shape)
        # print("Shape of position_embeddings:", self.position_embeddings.shape)
        x = x + self.position_embeddings
        x = torch.cat((cls_token, x), dim=1)
        x = self.dropout(x)
        return x
