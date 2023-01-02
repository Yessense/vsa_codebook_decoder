import random
from typing import List
import itertools
import sys

sys.path.append("..")

from tqdm import tqdm

import wandb
import hydra
import torch
from hydra.core.config_store import ConfigStore

from vsa_codebook_decoder.codebook import Codebook, Feature, vsa
from vsa_codebook_decoder.dataset.paired_dsprites import Dsprites
from vsa_codebook_decoder.model.binder import FourierBinder
from vsa_tests.config import ExperimentConfig

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="test_vsa_config", node=ExperimentConfig)


@hydra.main(version_base=None, config_path='../conf', config_name="test_vsa_config")
def main(cfg: ExperimentConfig):
    wandb.init(project=cfg.wandb.project, name=f"ld: {cfg.parameters.latent_dim} "
                                               f"n_v: {cfg.parameters.n_values} "
                                               f"n_f: {cfg.parameters.n_con} "
                                               f"ps: {cfg.parameters.power_step} ")

    # features: List[Feature] = Codebook.make_features_from_dataset(Dsprites)
    features = [Feature(name=str(i),
                        n_values=cfg.parameters.n_values,
                        contiguous=True,
                        density=cfg.parameters.power_step)
                for i in range(cfg.parameters.n_con)]

    cb = Codebook(features=features,
                  latent_dim=cfg.parameters.latent_dim,
                  seed=cfg.parameters.seed)

    placeholders = cb.placeholders
    vsa_features = cb.vsa_features

    accuracy = 0.
    total_samples = 0
    # unlikely_similarities = torch.zeros(cb.n_features)
    # likely_similarities = torch.zeros(cb.n_features)

    for sample_number in tqdm(range(cfg.parameters.n_samples)):
        # create random labels and select vsa vectors for this label
        label = [random.randrange(feature.n_values) for feature in features[1:]]
        vsa_vectors = [vsa_features[feature_number][feature_vector] for
                       feature_number, feature_vector in enumerate(label)]

        x = torch.stack(vsa_vectors)
        z = vsa.bind(x, placeholders)
        s = torch.sum(z, dim=0)

        # Для каждого отдельно признака
        for group_number, feature_group in enumerate(vsa_features):
            # Unbind с плейсхолдером
            unbinded_feature_value = vsa.unbind(s, placeholders[group_number])

            sims = torch.zeros(feature_group.shape[0])
            # Проверяем сходство для всех векторов из кодбука по этому свойству
            for value_number, feature_value in enumerate(feature_group):
                sim = vsa.sim(unbinded_feature_value, feature_value)
                sims[value_number] = sim

            # Наиболее похожий вектор
            max_pos = torch.argmax(sims)
            success_unbind = max_pos == label[group_number]
            accuracy += success_unbind
            # likely_similarities[group_number] += torch.abs(sims[label[group_number]])
            # mask = torch.ones_like(sims, dtype=torch.bool)
            # mask[label[group_number]] = False
            # unlikely_similarities[group_number] += torch.mean(torch.abs(sims[mask]))

        total_samples += 1
        # if total_samples > 100:
        #     break

    accuracy = accuracy.float() / cfg.parameters.n_con / total_samples
    # likely_similarities /= total_samples
    # unlikely_similarities /= total_samples

    wandb.log({"Accuracy": accuracy})
    # wandb.log(
    #     {f"likely/{feature.name}": sim for feature, sim in zip(features[1:], likely_similarities)})
    # wandb.log({f"unlikely/{feature.name}": sim for feature, sim in
    #            zip(features[1:], unlikely_similarities)})


if __name__ == '__main__':
    main()
