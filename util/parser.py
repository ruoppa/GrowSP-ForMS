import os
import argparse
import shutil

from typing import Tuple

from util.misc import prompt_bool
from util import Mode


def get_args():
    parser = argparse.ArgumentParser()
    # Config
    parser.add_argument(
        "--model_name", type=str, default="ResNet16", help="backbone model name"
    )
    parser.add_argument(
        "--experiment_name", type=str, default="test", help="experiment name"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="EvoMS", help="name of dataset to use"
    )
    # Neural network load options
    parser.add_argument(
        "--load_ckpt",
        type=str,
        default=None,
        help="path to load mode checkpoint from. 'latest' to load the newest checkpoint",
    )
    parser.add_argument(
        "--cfg_from_ckpt",
        action="store_true",
        help="load model and dataset config from checkpoint as well. Requires --load_ckpt to be set.",
    )
    # Other
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="automatically answer 'yes' to any propmts",
    )
    parser.add_argument(
        "--disable_config_dump",
        action="store_true",
        help="disable dumping model config .yaml files in experiment path. True by default if cfg_from_ckpt is set.",
    )
    parser.add_argument("--log_args", action="store_true", help="log args to file")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="save visualization of predicted classes for each point cloud",
    )
    parser.add_argument(
        "--plot_id",
        type=int,
        default=None,
        help="id of plot to evaluate. Required in 'eval_unlabeled' mode, ignored in other modes.",
    )
    # Set mode (data preparation, training or eval)
    parser.add_argument(
        "--mode",
        choices=[
            "train",
            "eval",
            "eval_unlabeled",
            "test",
            "data_prepare",
            "initial_sp",
            "populate_rgb",
        ],
        default="train",
    )
    # If set, only save the latest checkpoint (and delete the previous one when saving)
    parser.add_argument(
        "--save_latest_only",
        action="store_true",
        help="if set, only save the latest checkpoint during training",
    )
    # Convenience arguments that overload config values (these do not affect the training process and changing them often may be useful)
    parser.add_argument(
        "--l_min",
        type=float,
        default=None,
        help="overload the linearity threshold in model config file",
    )
    parser.add_argument(
        "--n_overseg_classes",
        type=int,
        default=None,
        help="overload the number of oversegmentation classes in model config file",
    )

    # Parse args
    args = parser.parse_args()
    # Determine current mode
    prepare_mode, sp_mode, rgb_mode = (
        args.mode == "data_prepare",
        args.mode == "initial_sp",
        args.mode == "populate_rgb",
    )
    no_nn = prepare_mode or sp_mode or rgb_mode

    # Set dataset config path
    args.dataset_config_path = os.path.join(
        ".", "cfgs", "datasets", args.dataset_name + ".yaml"
    )

    if no_nn:
        if not rgb_mode:  # No separate config file for populating missing rgb values
            # Set config path
            config_name = _get_cfg_name_no_nn((prepare_mode, sp_mode))
            args.config_path = os.path.join(
                ".", "cfgs", "datasets", args.dataset_name, config_name
            )

    # Validate arguments and perform operations that are specifc to neural network related modes
    else:
        args.mode = Mode(args.mode)
        _validate_arguments(args)
        args.config_path = os.path.join(
            ".", "cfgs", "models", args.model_name + ".yaml"
        )
        # Path to directory for saving experiment results
        args.experiment_path = os.path.join(
            ".", "ckpt", args.dataset_name, args.model_name, args.experiment_name
        )
        args.log_name = f"{args.dataset_name}_{args.model_name}_{args.experiment_name}"
        if (
            not os.path.exists(args.experiment_path) and args.mode is Mode.train
        ):  # Create directory if necessary
            os.makedirs(args.experiment_path)
            print(f"Created experiment path successfully at '{args.experiment_path}'")
        elif not args.load_ckpt:
            # Prompt user whether to overwrite experiment directory since the model is trained from scratch
            _handle_path_overwrite(args)
        # Also make sure that log, config and vis folders exist at experiment path
        _create_validate_directories(args)

    return args


def _validate_arguments(args: argparse.Namespace):
    """Validate arguments (some are mutually exclusive or some may be required if certain argument is set)

    Args:
        args (argparse.Namespace): arguments

    Raises:
        ValueError: if 'load_ckpt' is not provided in 'eval/test/eval_unlabeled' mode
        ValueError: if 'load_ckpt' is not provided when 'cfg_from_ckpt' is set
        ValueError: if 'plot_id' is not provided in 'eval_unlabeled' mode
    """
    if args.mode is not Mode.train and args.load_ckpt is None:
        raise ValueError(f"Argument 'load_ckpt' is required in mode {args.mode.value}!")
    if args.cfg_from_ckpt:
        if args.load_ckpt is None:
            raise ValueError(
                "Argument 'load_ckpt' is required if 'cfg_from_ckpt' is set!"
            )
        # Config cannot be dumped when cfg_from_ckpt is set
        args.disable_config_dump = True
    if args.mode is Mode.eval_unlabeled and args.plot_id is None:
        raise ValueError(f"Argument 'plot_id' is required in mode {args.mode.value}!")


def _get_cfg_name_no_nn(mode_bools: Tuple[bool, bool]) -> str:
    """Get config file name for non neural network modes (except populate_rgb, which requires no separate config file)

    Args:
        mode_bools (Tuple[bool, bool]): bool tuple that indicates which is the current mode.

    Returns:
        str: name of the config file
    """
    prepare_mode, sp_mode = mode_bools
    if prepare_mode:
        config_name = "preprocess.yaml"
    elif sp_mode:
        config_name = "initial_superpoints.yaml"

    return config_name


def _handle_path_overwrite(args: argparse.Namespace):
    """Handle case where experiment path exists but load_ckpt is not set

    Args:
        args (argparse.Namespace): arguments
    """
    overwrite = args.yes
    if not overwrite:  # If -y option used, skip prompting and overwrite
        overwrite = prompt_bool(
            f"Experiment path '{args.experiment_path}' already exists. Do you want to overwrite it?"
        )
    if overwrite:
        print(f"Attempting to delete '{args.experiment_path}' and all its contents...")
        try:
            shutil.rmtree(args.experiment_path)
            os.makedirs(args.experiment_path)  # Create experiment directory again
        except OSError as e:
            print(f"Error: {e.filename} - {e.strerror}")
            print("Terminating...")
            exit()
        print("Success!")


def _create_validate_directories(args: argparse.Namespace):
    """Check whether the required directories exist and generate them if necessary. The function also changes the values
    of config_path and dataset_config_path if 'cfg_from_ckpt' is set

    Args:
        args (argparse.Namespace): arguments
    """
    args.log_path = os.path.join(args.experiment_path, "logs")
    args.cfg_copy_path = os.path.join(args.experiment_path, "cfgs")
    args.vis_path = os.path.join(args.experiment_path, "vis")
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(args.cfg_copy_path) and args.mode is Mode.train:
        os.makedirs(args.cfg_copy_path)
    if (
        args.visualize
        and not os.path.exists(args.vis_path)
        and args.mode is not Mode.train
        and os.path.exists(args.experiment_path)
    ):
        os.makedirs(args.vis_path)
    if args.cfg_from_ckpt:
        args.config_path = os.path.join(args.cfg_copy_path, args.model_name + ".yaml")
        args.dataset_config_path = os.path.join(
            args.cfg_copy_path, args.dataset_name + ".yaml"
        )
