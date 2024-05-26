from .position_embedding_stategy import PositionalEmbeddingStrategy
import torch
from torch import nn


class PatchEmbeddingLearnedPositionalEmbedding(PositionalEmbeddingStrategy):
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

        self.cls_token = nn.Parameter(
            torch.randn(size=(1, embed_dim)), requires_grad=True
        )
        self.position_embeddings = nn.Parameter(
            torch.randn(size=(1, num_patches + 1, embed_dim)), requires_grad=True
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.patcher(x).permute(0, 2, 1)
        # print(f"Shape of x after patching and permuting: {x.shape}")
        # print(f"Shape of cls_token: {cls_token.shape}")
        x = torch.cat((cls_token, x), dim=1)
        # print(f"Shape of x after concatenating cls_token: {x.shape}")
        # print(f"Shape of position_embeddings: {self.position_embeddings.shape}")
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x
