from dataclasses import dataclass, field
from typing import Tuple, Optional, List
from vsa_codebook_decoder.dataset.paired_dsprites import Dsprites


@dataclass
class DatasetConfig:
    mode: str = "dsprites"
    path_to_dataset: str = "data/dsprites/dsprites_train.npz"
    train_size: int = 100_000
    val_size: int = 30_000


@dataclass
class DecoderConfig:
    in_channels: int = 64
    hidden_channels: int = 64


@dataclass
class ModelConfig:
    decoder_config: DecoderConfig = field(default_factory=DecoderConfig)
    latent_dim: int = 1024
    lr: float = 0.00025
    image_size: Tuple[int, int, int] = (1, 64, 64)
    binder: str = "fourier"
    monitor: str = "Validation/MSE Loss"


@dataclass
class ExperimentConfig:
    seed: int = 0
    batch_size: int = 64
    steps_per_epoch: int = 0
    accelerator: str = 'gpu'
    devices: List[int] = field(default_factory=lambda: [0])
    max_epochs: int = 300
    profiler: Optional[str] = None
    gradient_clip: float = 0.0


@dataclass
class CheckpointsConfig:
    save_top_k: int = 1
    every_k_epochs: int = 10
    check_val_every_n_epochs: int = 5
    ckpt_path: Optional[str] = None


@dataclass
class VSADecoderConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    checkpoint: CheckpointsConfig = field(default_factory=CheckpointsConfig)
