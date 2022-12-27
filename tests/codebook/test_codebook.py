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

