from dataclasses import dataclass
from typing import List
import torch
from . import vsa


@dataclass
class Feature:
    name: str
    n_values: int
    contiguous: bool = False
    density: float = 0.


class Codebook:
    features: List[Feature]
    latent_dim: int
    codebook: List[torch.tensor]

    def __init__(self, features: List[Feature], latent_dim: int):
        self.features = features
        # Add placeholders Feature class to automatic creation later
        placeholders = Feature(name='Placeholders', n_values=len(self.features))
        self.features.insert(0, placeholders)

        self.latent_dim = latent_dim
        self.codebook = []

        for feature in features:
            feature_vectors = torch.zeros((feature.n_values, latent_dim),
                                          dtype=torch.float32)

            if feature.contiguous:
                base_vector = vsa.generate(self.latent_dim)
                base_vector = vsa.make_unitary(base_vector)

                for i in range(feature.n_values):
                    feature_vectors[i] = vsa.pow(base_vector,
                                                 1 + i * feature.density)
            else:
                for i in range(feature.n_values):
                    feature_vectors[i] = vsa.generate(self.latent_dim)
            self.codebook.append(feature_vectors)


if __name__ == '__main__':
    features = [Feature('shape', 3), Feature('scale', 6, contiguous=True)]
    latent_dim = 1024
    codebook = Codebook(features, latent_dim)

    pass
