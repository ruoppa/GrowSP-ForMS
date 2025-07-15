import os
import logging
import argparse
import glob
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim

from easydict import EasyDict
from typing import Tuple
from pathlib import Path
from util.logger import print_log
from util.misc import prompt_bool


def load_model(
    base_model: nn.Module,
    classifier: nn.Linear,
    ckpt_path: str,
    config: EasyDict = None,
    eval: bool = False,
    logger: logging.Logger = None,
) -> Tuple[int, str]:
    """Load a model from state dict

    Args:
        base_model (nn.Module): neural network model
        classifier: (nn.Linear): linear classifier
        ckpt_path (str): path to checkpoint file
        config (EasyDict, optional): model config dict. Defaults to None.
        eval (bool, optional): set to True when using evaluation mode (training not continued). Defaults to False.
        logger (logging.Logger, optional): logger object. Defaults to None.

    Raises:
        FileNotFoundError: if no checkpoint found from path
        RuntimeError: if checkpoint weight mismatch with model
        ValueError: if loaded state is invalid

    Returns:
        Tuple[int, str]: current epoch and state (based on the loaded model)
    """
    if not eval:
        assert config is not None, "'config' is required when 'eval' is False!"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint file found from '{ckpt_path}'!")
    print_log(f"Loading model weights from '{ckpt_path}'...", logger=logger)

    # load state dict
    state_dict = torch.load(ckpt_path)
    # parameter resume of base model
    if state_dict.get("model") is not None:
        base_ckpt = {
            k.replace("module.", ""): v for k, v in state_dict["model"].items()
        }
    elif state_dict.get("base_model") is not None:
        base_ckpt = {
            k.replace("module.", ""): v for k, v in state_dict["base_model"].items()
        }
    else:
        raise RuntimeError(
            "Mismatch of ckpt weight. No matching weights for model found!"
        )
    base_model.load_state_dict(base_ckpt, strict=True)
    if state_dict.get("classifier") is not None:
        classifier.load_state_dict(state_dict["classifier"])
    else:
        raise RuntimeError(
            "Mismatch of ckpt weight. No matching weights for 'classifier' found!"
        )

    epoch = state_dict.get("epoch", 0)
    state = state_dict.get("state", "pretrain")
    if eval:
        # In eval mode it does not matter if the epoch conflicts with the model config, thus we do not validate the epoch, only the state
        if state != "pretrain" and state != "grow":
            raise ValueError(
                f"'state' should be one of ['pretrain', 'grow'] but got '{state}' instead!"
            )
    else:
        _validate_epoch_and_state(
            config, epoch, state
        )  # Ensure that there is no conflict between loaded parameters and model config

    return epoch, state


def load_optimizer(
    optimizer: optim.Optimizer,
    state: str,
    ckpt_path: str,
    logger: logging.Logger = None,
):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"no checkpoint file found from '{ckpt_path}")
    print_log(f"Loading optimizer from '{ckpt_path}'...", logger=logger)
    # load state dict
    state_dict = torch.load(ckpt_path)
    # optimizer
    optimizer_key = _get_state_postfix("optimizer", state)
    if state_dict.get(optimizer_key) is not None:
        optimizer.load_state_dict(state_dict[optimizer_key])
    else:
        raise RuntimeError(
            f"Mismatch of ckpt weight. No matching weights for {optimizer_key} found!"
        )


def load_scheduler(
    scheduler: optim.lr_scheduler._LRScheduler,
    state: str,
    ckpt_path: str,
    logger: logging.Logger = None,
    is_growth_start: bool = False,
):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"no checkpoint file found from '{ckpt_path}")
    print_log(f"Loading scheduler from '{ckpt_path}'...", logger=logger)
    # Scheduler is not loaded in case the loading happens at the first epoch of the growth stage. This allows altering the
    # the learning rate shceduler parameters in the config
    if not is_growth_start:
        # load state dict
        state_dict = torch.load(ckpt_path)
        # scheduler
        scheduler_key = _get_state_postfix("scheduler", state)
        if state_dict.get(scheduler_key) is not None:
            scheduler.load_state_dict(state_dict[scheduler_key])
        else:
            raise RuntimeError(
                f"Mismatch of ckpt weight. No matching weights for {scheduler_key} found!"
            )


def save_checkpoint(
    base_model: nn.Module,
    classifier: nn.Linear,
    optimizer_1: optim.Optimizer,
    optimizer_2: optim.Optimizer,
    scheduler_1: optim.lr_scheduler._LRScheduler,
    scheduler_2: optim.lr_scheduler._LRScheduler,
    epoch: int,
    state: str,
    ckpt_filename: str,
    args: argparse.Namespace,
    save_interval: int,
    logger=None,
):
    save_path = os.path.join(args.experiment_path, ckpt_filename)

    torch.save(
        {
            "base_model": base_model.state_dict(),
            "classifier": classifier.state_dict(),
            "optimizer_1": optimizer_1.state_dict(),
            "optimizer_2": optimizer_2.state_dict(),
            "scheduler_1": scheduler_1.state_dict(),
            "scheduler_2": scheduler_2.state_dict(),
            "epoch": epoch,
            "state": state,
        },
        save_path,
    )
    print_log(f"Saved checkpoint at '{save_path}'", logger=logger)
    # If save_latest_only is set, delete the previous checkpoint
    if args.save_latest_only:
        _delete_previous_checkpoint(epoch, args, save_interval)


def _delete_previous_checkpoint(
    epoch: int, args: argparse.Namespace, save_interval: int
) -> None:
    prev_ckpt_filename = f"{args.experiment_name}_{epoch - save_interval}_ckpt.pth"
    save_path = os.path.join(args.experiment_path, prev_ckpt_filename)
    if os.path.exists(save_path):
        os.remove(save_path)


def parse_load_ckpt(args: argparse.Namespace, logger: logging.Logger = None) -> str:
    """Parse the load_ckpt argument

    Args:
        args (argparse.Namespace): args passed to main
        logger (logging.Logger, optional): logger object. Defaults to None.

    Returns:
        str: path to checkpoint
    """
    ckpt_path = args.load_ckpt
    # Parse load_ckpt
    if os.sep not in args.load_ckpt:
        is_int = True
        try:
            epoch = int(ckpt_path)
        except ValueError:
            is_int = False
        if is_int:
            ckpt_path = _get_ckpt_path_by_epoch(args, epoch, logger)
        elif args.load_ckpt == "latest":
            ckpt_path = _get_latest_ckpt_path(args, logger)
        else:
            # Checkpoint at experiment path
            ckpt_path = os.path.join(args.experiment_path, ckpt_path)

    return ckpt_path


def _get_latest_ckpt_path(
    args: argparse.Namespace, logger: logging.Logger = None
) -> str:
    """Get the path of the latest checkpoint. This function is based on the assumption that all checkpoints
    are named with the convention: <experiment_name>_<epoch_number>_ckpt.pth

    Args:
        args (argparse.Namespace): arguments passed to script
        logger (logging.Logger, optional): logger. Defaults to None.

    Raises:
        FileNotFoundError: if no suitable checkpoint files are found

    Returns:
        str: path to the latest checkpoint
    """
    ckpt_file_list = sorted(glob.glob(os.path.join(args.experiment_path, "*.pth")))
    latest_epoch, latest_path = -1, ""
    first_ind = len(args.experiment_name)

    for ckpt_path in ckpt_file_list:
        ckpt = Path(ckpt_path).name[first_ind:]
        split_ckpt = ckpt.split("_")
        try:
            epoch = int(split_ckpt[1])
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_path = ckpt_path
        except Exception:
            print_log(
                f"Failed to parse checkpoint filename '{ckpt}', ignoring file!",
                logger=logger,
            )
            continue

    if latest_epoch == -1:
        raise FileNotFoundError(
            f"load failed, no suitable checkpoint files found in '{args.experiment_path}'!"
        )

    return latest_path


def _get_ckpt_path_by_epoch(
    args: argparse.Namespace, epoch: int, logger: logging.Logger = None
) -> str:
    """Find the checkpoint file that is closest to 'epoch'. This function is once again based on the assumption that
    all checkpoints are named with the convention: <experiment_name>_<epoch_number>_ckpt.pth

    Args:
        args (argparse.Namespace): arguments passed to script
        epoch (int): epoch we want to load from
        logger (logging.Logger, optional): logger object. Defaults to None.

    Raises:
        FileNotFoundError: if no suitable checkpoint files are found

    Returns:
        str: path to checkpoint that is closest to the given epoch
    """
    ckpt_file_list = sorted(glob.glob(os.path.join(args.experiment_path, "*.pth")))
    closest_epoch, min_diff, closest_path = 0, np.inf, ""
    first_ind = len(args.experiment_name)

    for ckpt_path in ckpt_file_list:
        ckpt = Path(ckpt_path).name[first_ind:]
        split_ckpt = ckpt.split("_")
        try:
            current_epoch = int(split_ckpt[1])
            diff = np.abs(epoch - current_epoch)
            if diff < min_diff or (diff == min_diff and current_epoch > closest_epoch):
                # If difference is equal, a later epoch is prefered
                min_diff = diff
                closest_epoch = current_epoch
                closest_path = ckpt_path
            if current_epoch == epoch:
                break
        except Exception:
            print_log(
                f"Failed to parse checkpoint filename '{ckpt}', ignoring file!",
                logger=logger,
            )
            continue
    if np.isinf(min_diff):
        raise FileNotFoundError(
            f"load failed, no suitable checkpoint files found in '{args.experiment_path}'!"
        )
    if closest_epoch != epoch:
        load_closest = args.yes
        if not load_closest:
            load_closest = prompt_bool(
                f"No exact match for epoch {epoch}. Do you want to load the closest checkpoint @ epoch {closest_epoch}?"
            )
        if load_closest:
            print_log(
                f"Loading closest checkpoint @ epoch {closest_epoch}...", logger=logger
            )
        else:
            print("Terminating...")
            exit()

    return closest_path


def _get_state_postfix(name: str, state: str) -> str:
    """The optimizer and scheduler depened on the state (pretrain vs. growing superpoints).
    This function determines which to load based on the state

    Args:
        name (str): type of model to load
        state (str): state

    Raises:
        ValueError: if state is not a permitted value

    Returns:
        str: key to load from
    """
    if state == "pretrain":
        return name + "_1"
    elif state == "grow":
        return name + "_2"
    else:
        raise ValueError(f"State '{state}' not recognized!")


def _validate_epoch_and_state(config: EasyDict, epoch: int, state: str) -> None:
    """Validate that the epoch and state loaded from checkpoint are not in conflict with the config.
    Raise an error if there's a conflict and do nothing otherwise.

    Args:
        config (EasyDict): model config
        epoch (int): epoch loaded from checkpoint
        state (str): state loaded from checkpoint

    Raises:
        ValueError: if epoch is not at least 0
        ValueError: if state is 'prertrain' and epoch is greater than the max pretrain epoch in config
        ValueError: if state is 'grow' and epoch is greater than the max growth epoch in config
        ValueError: if state is not either 'pretrain' or 'grow'
    """
    if epoch < 0:
        raise ValueError("'epoch' should be at least 0!")
    if state == "pretrain":
        if epoch > config.growsp.pretrain_epochs:
            raise ValueError(
                f"Conflict between loaded values and model config! 'epoch' should be <= 'pretrain_epochs', "
                f"in state '{state}' but got {epoch} and {config.growsp.pretrain_epochs} instead!"
            )
    elif state == "grow":
        if epoch > config.growsp.grow_epochs:
            raise ValueError(
                f"Conflict between loaded values and model config! 'epoch' should be <= 'grow_epochs', "
                f"in state '{state}' but got {epoch} and {config.growsp.grow_epochs} instead!"
            )
    else:
        raise ValueError(
            f"'state' should be one of ['pretrain', 'grow'] but got '{state}' instead!"
        )
