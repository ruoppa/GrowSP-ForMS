import torch.nn as nn
import torch.optim as optim

from util.registry import Registry, build_optimizer_from_cfg, build_scheduler_from_cfg
from easydict import EasyDict


MODELS = Registry("models")
LOSSES = Registry("losses")
OPTIMIZERS = Registry("optimizers", build_func=build_optimizer_from_cfg)
SCHEDULERS = Registry("schedulers", build_func=build_scheduler_from_cfg)


def build_model_from_config(config: EasyDict, **kwargs) -> nn.Module:
    """Build a neural network defined by config.NAME.

    Args:
        config (EasyDict): config. Should contain the type of neural network in parameter NAME, and at least the keyword arguments required by said NN.

    Returns:
        nn.Module: built neural network object
    """
    return MODELS.build(config, **kwargs)


def build_loss_from_config(config: EasyDict, **kwargs):
    """Build a loss function defined by config.NAME

    Args:
        config (EasyDict): Should contain the type of loss in parameter NAME, and at least the keyword arguments required by said loss.

    Returns:
        function: built loss function
    """
    return LOSSES.build(config, **kwargs)


def build_optimizer_from_config(
    config: EasyDict, base_model: nn.Module, default_args=None
) -> optim.Optimizer:
    """Build an optimizer from config defined by config.NAME. The optimizer is built for the provided neural network

    Args:
        config (EasyDict): config. Should contain the type of optimizer in parameter NAME, and at least the keyword arguments required by said optimizer.
        base_model (nn.Module): neural network the optimizer is built for
        default_args (optional): other. Defaults to None.

    Returns:
        (optim.Optimizer): built optimizer object
    """
    return OPTIMIZERS.build(config, base_model=base_model, default_args=default_args)


def build_scheduler_from_config(
    config: EasyDict, optimizer: optim.Optimizer, default_args=None
) -> optim.lr_scheduler._LRScheduler:
    """Build a scheduler from config defined by config.NAME. The scheduler is built for the provided optimizer

    Args:
        config (EasyDict): config. Should contain the type of scheduler in parameter NAME, and at least the keyword arguments required by said scheduler.
        base_model (nn.Module): optimizer the scheduler is built for.
        default_args (optional): other. Defaults to None.

    Returns:
        (optim.lr_scheduler._LRScheduler): built scheduler
    """
    return SCHEDULERS.build(config, optimizer=optimizer, default_args=default_args)
