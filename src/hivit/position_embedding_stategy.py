from torch import nn


class PositionalEmbeddingStrategy(nn.Module):
    def forward(self, x):
        raise NotImplementedError(
            "Each Embedding Strategy must implement the forward method."
        )
