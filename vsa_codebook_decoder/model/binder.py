import torch
from torch import nn
from ..codebook import vsa


class Binder(nn.Module):
    n_features: int
    latent_dim: int
    hd_placeholders: nn.Parameter

    def __init__(self, n_features: int, latent_dim: int):
        super().__init__()
        self.n_features = n_features
        self.latent_dim = latent_dim


class FourierBinder(Binder):
    def __init__(self, placeholders: torch.tensor, latent_dim: int):
        super().__init__(placeholders.shape[0], latent_dim)

        self.hd_placeholders = nn.Parameter(data=placeholders.unsqueeze(0))

    def forward(self, z):
        out = vsa.bind(self.hd_placeholders.data, z)
        return out
