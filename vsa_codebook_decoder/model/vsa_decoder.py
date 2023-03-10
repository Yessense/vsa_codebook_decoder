import pathlib
from typing import Any, Optional

import hydra
import pytorch_lightning as pl
import torch
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch.nn.functional as F
from torch.optim import lr_scheduler

from ..utils import iou_pytorch
from ..dataset.paired_dsprites import Dsprites
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

        features = Codebook.make_features_from_dataset(Dsprites)  # type: ignore

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

        self.save_hyperparameters()

    def encode(self, labels):
        """
        Make latent representation with vsa vectors for labels

        Parameters
        ----------
        labels: torch.tensor
            labels -> (batch_size, n_features)

        Returns
        -------
        features: torch.tensor
            features -> (batch_size, n_features, latent_dim)

        """
        # features -> (batch_size, n_features, latent_dim)
        features = torch.zeros((labels.shape[0],
                                self.codebook.n_features,
                                self.cfg.model.latent_dim), dtype=torch.float32)

        # codebook.vsa_features -> List[n_features, torch.tensor[feature_count, latent_dim]]
        # vsa_feature -> (feature_count, latent_dim)
        vsa_feature: torch.tensor
        for i, vsa_feature in enumerate(self.codebook.vsa_features):
            # vsa_value -> (latent_dim)
            vsa_value: torch.tensor
            for j, vsa_value in enumerate(vsa_feature):
                mask = labels[:, i] == j
                features[mask, i] = vsa_value.unsqueeze(0)

        features = features.to(self.device)
        features = self.binder(features)
        z = torch.sum(features, dim=1)
        return z

    def forward(self, x):
        z = self.encode(x)
        recons = self.decoder(z)
        return recons

    def step(self, batch, batch_idx, mode: str = 'Train') -> torch.tensor:
        # Logging period
        # Log Train samples once per epoch
        # Log Validation images triple per epoch
        if mode == 'Train':
            log_images = lambda x: x == 0
        elif mode == 'Validation':
            log_images = lambda x: x % 10 == 0
        else:
            raise ValueError

        image: torch.tensor
        label: torch.tensor

        image, labels = batch

        z = self.encode(labels)
        decoded_image = self.decoder(z)

        loss = F.mse_loss(decoded_image, image)
        iou = iou_pytorch(decoded_image, image)

        self.log(f"{mode}/MSE Loss", loss)
        self.log(f"{mode}/IOU", iou)

        if log_images(batch_idx):
            self.logger.experiment.log({
                f"{mode}/Images": [
                    wandb.Image(image[0], caption='Image'),
                    wandb.Image(decoded_image[0],
                                caption='Reconstruction'),
                ]})
        return loss

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self.step(batch, batch_idx, mode='Train')
        return loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        self.step(batch, batch_idx, mode='Validation')
        return None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.experiment.lr)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.cfg.experiment.lr,
                                            epochs=self.cfg.experiment.max_epochs,
                                            steps_per_epoch=self.cfg.experiment.steps_per_epoch,
                                            pct_start=self.cfg.experiment.pct_start)
        return {"optimizer": optimizer,
                "lr_scheduler": {'scheduler': scheduler,
                                 'interval': 'step',
                                 'frequency': 1}, }


cs = ConfigStore.instance()
cs.store(name="config", node=VSADecoderConfig)

path_to_dataset = pathlib.Path().absolute()


@hydra.main(version_base=None, config_name="config")
def main(cfg: VSADecoderConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.experiment.seed)
    model = VSADecoder(cfg)

    batch_size = 10
    img = torch.randn((batch_size, 1, 64, 64))
    labels = torch.randint(0, 3, (batch_size, 5))

    model.step((img, labels), 1)

    pass


if __name__ == '__main__':
    main()
