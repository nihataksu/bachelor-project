from .position_embedding_stategy import PositionalEmbeddingStrategy
import torch
from torch import nn
from torchvision.models import resnet50
from .hilbert_indicies import HilbertIndices
import numpy as np


class PatchEmbeddingResnetHilbertPositionalEmbedding(PositionalEmbeddingStrategy):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels, order):
        super().__init__()

        # ResNet-50 Backbone
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the classification head

        # Project ResNet-50 features to embedding dimension
        self.feature_proj = nn.Conv2d(
            in_channels=2048, out_channels=embed_dim, kernel_size=1
        )

        # Patcher for the ViT
        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels=embed_dim,
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

        pe = torch.zeros(num_patches, embed_dim)
        pe = pe[hilbert_indices]  # reorder according to Hilbert curve

        # Adds the sin and cos values for each alternating column to add positional embeddings
        pe[:, 0::2] = torch.sin(position * div_term)[:, 0::2]
        pe[:, 1::2] = torch.cos(position * div_term)[:, 1::2]

        self.position_embeddings = nn.Parameter(pe[:], requires_grad=False)

        # Add cls token separately
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Extract features using ResNet-50
        x = self.resnet(x)
        x = x.unsqueeze(-1).unsqueeze(-1)  # Add spatial dimensions
        x = self.feature_proj(x)  # Project to embedding dimension
        x = self.patcher(x).permute(0, 2, 1)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = x + self.position_embeddings
        x = torch.cat((cls_token, x), dim=1)
        x = self.dropout(x)
        return x
