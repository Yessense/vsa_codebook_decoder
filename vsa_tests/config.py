from dataclasses import dataclass, field


@dataclass
class ParametersConfig:
    latent_dim: int = 2
    seed: int = 0
    n_samples: int = 10
    n_con: int = 1
    n_values: int = 1
    power_step: float = 0.1


@dataclass
class WandbConfig:
    project: str = 'vsa_stats'


@dataclass
class ExperimentConfig:
    parameters: ParametersConfig = field(default_factory=ParametersConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
