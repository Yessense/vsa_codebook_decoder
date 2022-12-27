from dataclasses import dataclass
from typing import List
import torch
import vsa


@dataclass
class Feature:
    name: str
    n_values: int
    contiguous: bool
    density: float = 0.


class Codebook:
    features: List[Feature]
    latent_dim: int

    def __init__(self, features: List[Feature], latent_dim: int):
        self.features = features
        self.latent_dim = latent_dim

        self.codebook = []

        for feature in features:
            feature_vectors = torch.zeros((feature.n_values, latent_dim),
                                          dtype=torch.float32)

            if feature.contiguous:
                base_vector = vsa.generate(self.latent_dim)

                for i in range(feature.n_values):
                    feature_vectors[i] = vsa.pow(base_vector,
                                                 1 + i * feature.density)
            else:
                for i in range(feature.n_values):
                    feature_vectors[i] = vsa.generate(self.latent_dim)
            self.codebook.append(feature_vectors)


