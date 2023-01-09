import os
import pathlib

import hydra
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from .dataset.generalization_dsprites import GeneralizationDspritesDataModule
from .dataset.dsprites import DspritesDatamodule
from .model.vsa_decoder import VSADecoder
from .config import VSADecoderConfig

cs = ConfigStore.instance()
cs.store(name="config", node=VSADecoderConfig)

path_to_dataset = pathlib.Path().absolute()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: VSADecoderConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.experiment.seed)

    if cfg.dataset.mode == 'dsprites':
        datamodule = DspritesDatamodule(
            path_to_data_dir=path_to_dataset / cfg.dataset.path_to_dataset,
            batch_size=cfg.experiment.batch_size,
            train_size=cfg.dataset.train_size,
            val_size=cfg.dataset.val_size)
    elif cfg.dataset.mode == 'generalization dsprites':
        datamodule = GeneralizationDspritesDataModule(
            path_to_data_dir=path_to_dataset / cfg.dataset.path_to_dataset,
            batch_size=cfg.experiment.batch_size,
            train_size=cfg.dataset.train_size,
            val_size=cfg.dataset.val_size)

    else:
        raise NotImplemented(f"Wrong dataset mode {cfg.dataset.path_to_dataset!r}")

    cfg.experiment.steps_per_epoch = cfg.dataset.train_size // cfg.experiment.batch_size

    model = VSADecoder(cfg=cfg)

    top_metric_callback = ModelCheckpoint(monitor=cfg.model.monitor,
                                          save_top_k=cfg.checkpoint.save_top_k)
    every_epoch_callback = ModelCheckpoint(
        every_n_epochs=cfg.checkpoint.every_k_epochs)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')

    callbacks = [
        top_metric_callback,
        every_epoch_callback,
        lr_monitor,
    ]

    run = wandb.init(project=cfg.dataset.mode + '_vsa',
                     name=f'{cfg.dataset.mode} -l {cfg.model.latent_dim} '
                          f'-s {cfg.experiment.seed} '
                          f'-bs {cfg.experiment.batch_size} '
                          f'vsa',
                     dir=cfg.experiment.logging_dir)

    wandb_logger = WandbLogger(experiment=run)

    wandb_logger.watch(model)

    # trainer
    trainer = pl.Trainer(accelerator=cfg.experiment.accelerator,
                         devices=cfg.experiment.devices,
                         max_epochs=cfg.experiment.max_epochs,
                         profiler=cfg.experiment.profiler,
                         callbacks=callbacks,
                         logger=wandb_logger,
                         check_val_every_n_epoch=cfg.checkpoint.check_val_every_n_epochs,
                         gradient_clip_val=cfg.experiment.gradient_clip)

    trainer.fit(model,
                datamodule=datamodule,
                ckpt_path=cfg.checkpoint.ckpt_path)


if __name__ == '__main__':
    main()
