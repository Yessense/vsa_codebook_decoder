from typing import Any

import hydra
import pytorch_lightning as pl
from hydra.core.config_store import ConfigStore
from pytorch_lightning.utilities.types import STEP_OUTPUT

from vsa_codebook_decoder.dataset.paired_dsprites import Dsprites
from .binder import Binder, FourierBinder
from .decoder import Decoder
from ..codebook.codebook import Codebook
from ..config import VSADecoderConfig


class VSADecoder(pl.LightningModule):
    binder: Binder
    cfg: VSADecoderConfig

    def __init__(self, cfg: VSADecoderConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.dataset.dataset_name == 'dsprites':
            features = Codebook.make_features_from_dataset(Dsprites)  # type: ignore
        else:
            raise ValueError(f"Wrong dataset name {cfg.dataset.dataset_name}")

        self.decoder = Decoder(image_size=cfg.model.image_size,
                               latent_dim=cfg.model.latent_dim,
                               in_channels=cfg.model.decoder_config.in_channels,
                               hidden_channels=cfg.model.decoder_config.hidden_channels)
        self.codebook = Codebook(features=features,
                                 latent_dim=cfg.model.latent_dim,
                                 seed=cfg.experiment.seed)

        if cfg.model.binder == 'fourier':
            self.binder = FourierBinder(placeholders=self.codebook.placeholders)
        else:
            raise NotImplemented(f"Wrong binder type {cfg.model.binder}")

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:




cs = ConfigStore.instance()
cs.store(name="config", node=VSADecoderConfig)


@hydra.main(config_name="config")
def main(cfg: VSADecoderConfig) -> None:
    vsa_decoder = VSADecoder(cfg)
    pass


if __name__ == '__main__':
    main()
