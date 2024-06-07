from .position_embedding_stategy import PositionalEmbeddingStrategy
import torch
from torch import nn
from torchvision.models import resnet101


class PatchEmbeddingResnetLearnedPositionalEmbedding(PositionalEmbeddingStrategy):
    def __init__(
        self, embed_dim, patch_size, num_patches, dropout, in_channels, image_size
    ):
        super().__init__()
        self.resnet101 = resnet101(pretrained=True)
        self.resnet101 = nn.Sequential(
            *list(self.resnet101.children())[:-2]
        )  # Remove the final fully connected layer

        # Calculate the feature map size
        self.feature_map_size = image_size // 32

        # Ensure the patch size is valid for the given feature map size
        if patch_size > self.feature_map_size:
            patch_size = self.feature_map_size

        # Calculate the actual number of patches
        self.num_patches = (self.feature_map_size // patch_size) ** 2

        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels=2048,  # ResNet-101 output channels before avgpool is 2048
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
            torch.randn(size=(1, self.num_patches + 1, embed_dim)), requires_grad=True
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        with torch.no_grad():
            x = self.resnet101(x)

        # print(f"Shape of ResNet-101 output: {x.shape}")

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.patcher(x).permute(0, 2, 1)

        # print(f"Shape of patcher output: {x.shape}")
        # print(f"Shape of cls_token: {cls_token.shape}")

        x = torch.cat((cls_token, x), dim=1)

        # print(f"Shape after concatenating cls_token: {x.shape}")
        # print(f"Shape of position_embeddings: {self.position_embeddings.shape}")

        x = x + self.position_embeddings
        x = self.dropout(x)
        return x
