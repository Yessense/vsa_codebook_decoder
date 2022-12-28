import torch

import vsa_codebook_decoder.codebook.vsa as vsa
import vsa_codebook_decoder.codebook.codebook as cb

class TestCodebook:
    def test_codebook(self):
        features = [cb.Feature('shape', 3), cb.Feature('scale', 6, contiguous=True)]
        latent_dim = 1024

        codebook = cb.Codebook(features, latent_dim)

        assert len(codebook.features) == 3
        assert len(codebook.codebook) == 3
        assert codebook.codebook[0].shape == (2, latent_dim)
        assert codebook.codebook[1].shape == (3, latent_dim)
        assert codebook.codebook[2].shape == (6, latent_dim)

        # tensor([[0.0215, 0.0286, 0.0151, ..., -0.0206, -0.0378, -0.0272],
        #         [-0.0685, -0.0025, -0.0064, ..., -0.0982, -0.0559, 0.0195],
        #         [0.0611, 0.0316, -0.0032, ..., -0.0722, -0.0585, -0.0056]])

        assert torch.allclose(codebook.codebook[1][0][0], torch.tensor(0.0215), rtol=1e-4, atol=1e-4)
        assert torch.allclose(codebook.codebook[1][0][1], torch.tensor(0.0286), rtol=1e-4, atol=1e-4)

