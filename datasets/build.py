import numpy as np

from easydict import EasyDict
from torch.utils.data import DataLoader
from datasets.dataloader import cfl_collate_fn, cfl_collate_fn_val

from util import Mode
from util.registry import Registry

DATASETS = Registry("datasets")


def _worker_init_fn(seed):
    return lambda x: np.random.seed(seed + x)


def build_dataloader_from_config(
    config: EasyDict, mode: Mode, seed: int = None, default_args=None
):
    dataset = DATASETS.build(config, default_args=default_args)
    dataset.mode = (
        mode  # Setting the mode should raise an error if the mode is not supported
    )
    load_config = config.model_config.dataloader  # Extract dataloader config

    if mode is Mode.train:
        if seed is None:
            raise ValueError("'seed' must be defined when mode is 'train'!")
        dataloader = DataLoader(
            dataset,
            batch_size=load_config.train_batch_size,
            shuffle=True,
            collate_fn=cfl_collate_fn(),
            num_workers=load_config.train_workers,
            pin_memory=True,
            worker_init_fn=_worker_init_fn(seed),
        )
    elif mode is Mode.cluster:
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=cfl_collate_fn(),
            num_workers=load_config.cluster_workers,
            pin_memory=True,
        )
    elif mode is Mode.eval or mode is Mode.test or mode is Mode.eval_unlabeled:
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=cfl_collate_fn_val(),
            num_workers=load_config.eval_workers,
            pin_memory=True,
        )
    else:
        raise NotImplementedError(
            f"Dataloader building not implemented for mode '{mode}'"
        )

    return dataloader
