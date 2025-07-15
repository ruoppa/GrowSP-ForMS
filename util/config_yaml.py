import os
import argparse
import logging
import yaml

from easydict import EasyDict
from typing import Union
from pathlib import Path
from .logger import print_log
from .misc import prompt_bool


def log_args_to_file(
    args: argparse.Namespace,
    pre: str = "args",
    logger: Union[logging.Logger, None] = None,
):
    for key, val in args.__dict__.items():
        print_log(f"{pre}.{key} : {val}", logger=logger)


def log_config_to_file(
    cfg: EasyDict, pre: str = "cfg", logger: Union[logging.Logger, None] = None
):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            print_log(f"{pre}.{key} = edict()", logger=logger)
            log_config_to_file(cfg[key], pre=pre + "." + key, logger=logger)
            continue
        print_log(f"{pre}.{key} : {val}", logger=logger)


def merge_new_config(config: EasyDict, new_config: dict) -> EasyDict:
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == "_base_":
                with open(new_config["_base_"], "r") as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except Exception:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config


def cfg_from_yaml_file(cfg_file: str, logger: Union[logging.Logger, None] = None):
    config = EasyDict()
    print_log(f"Loading config from '{cfg_file}'...", logger)
    with open(cfg_file, "r") as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except Exception:
            new_config = yaml.load(f)
    merge_new_config(config=config, new_config=new_config)
    return config


def copy_experiment_config(
    args: argparse.Namespace,
    config_path: str,
    logger: Union[logging.Logger, None] = None,
):
    """Create a copy of given config file in experiment path. If a copy already exists, the user is prompted to pick whether to
    overwrite it or not.

    Args:
        args (argparse.Namespace): args passed to main
        config_path (str): path to config file
        logger (Union[logging.Logger, None], optional): logger object. Defaults to None.
    """
    config_name = Path(config_path).name
    new_config_path = os.path.join(
        args.cfg_copy_path, config_name
    )  # Path where copy of config is saved
    create_copy = True
    if os.path.exists(new_config_path):
        create_copy = args.yes
        if not create_copy:
            create_copy = prompt_bool(
                f"Copy of '{config_name}' at '{args.cfg_copy_path}' already exists. Do you want to overwrite it?"
            )
    if create_copy:
        os.system("cp %s %s" % (config_path, new_config_path))
        print_log(f"Copied '{config_name}' to {args.cfg_copy_path}", logger=logger)
