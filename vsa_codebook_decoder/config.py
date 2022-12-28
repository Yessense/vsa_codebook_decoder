from dataclasses import dataclass, field
from typing import Tuple
from vsa_codebook_decoder.dataset.paired_dsprites import Dsprites


@dataclass
class DatasetConfig:
    dataset_name: str = "dsprites"


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


@dataclass
class ExperimentConfig:
    seed: int = 0


@dataclass
class VSADecoderConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
