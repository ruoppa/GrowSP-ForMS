import os
import warnings

import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, List
from pathlib import Path
from itertools import takewhile

from util import misc


def visualize_loss(
    log_path: str,
    first_epoch: int = 1,
    last_epoch: int = np.inf,
    keep_older_duplicate: bool = False,
    save: bool = False,
    save_path: str = "./figures/losses",
) -> None:
    """NOTE: that this function is very excepts a very specific format, i.e. if you have altered the logging format somehow, the
    function will most likely not work as intended.

    Args:
        log_path (str): path where the log file is located
        first_epoch (int, optional): First epoch to include in the figure. Defaults to 1.
        last_epoch (int, optional): Last epoch to include in the figure. Defaults to np.inf.
        keep_older_duplicate (bool, optional): If multiple records from the same epoch/batch exist, keep the older record. By default
            the newer record is kep. Defaults to False.
        save (bool, optional): if True, save the figure as a .png. Defaults to False.
        save_path (str, optional): path to the directory where the figure should be saved. Defaults to "./figures/losses"
    """
    loss_by_epoch = {}
    max_batches_by_epoch = {}

    with open(log_path) as fp:
        log_lines = fp.readlines()
        for line in log_lines:
            line = line.strip()
            split_line = line.split(" | ")
            if len(split_line) > 0:
                try:
                    epoch = int(split_line[0].split("epoch")[1].strip())
                    batch = int(split_line[1].split("[")[1].split("/")[0].strip())
                    loss = float(split_line[2].split("mean loss:")[1].strip())
                    if epoch >= first_epoch and epoch <= last_epoch:
                        contains_epoch = loss_by_epoch.get(epoch, False)
                        if not contains_epoch:
                            # No records for current epoch yet
                            loss_by_epoch[epoch] = {batch: loss}
                            # Extract number of batches in current epoch
                            max_batches = int(
                                split_line[1]
                                .split("[")[1]
                                .split("/")[1]
                                .strip()
                                .split(" (")[0]
                                .strip()
                            )
                            max_batches_by_epoch[epoch] = max_batches
                        else:
                            # Check if the dict for current epoch contains records for the current batch
                            contains_batch = loss_by_epoch.get(epoch).get(batch, False)
                            if not keep_older_duplicate or (
                                not contains_batch and keep_older_duplicate
                            ):
                                loss_by_epoch[epoch][batch] = loss
                except Exception:
                    pass  # Could not parse line (something wrong with the format)

    # Extract losses in a sorted order
    sorted_losses = []
    batches_by_epoch = []
    sorted_epochs = list(loss_by_epoch.keys())
    sorted_epochs.sort()

    for epoch in sorted_epochs:
        epoch_batches = loss_by_epoch[epoch]
        sorted_batches = list(epoch_batches.keys())
        sorted_batches.sort()
        epoch_losses = [epoch_batches[batch] for batch in sorted_batches]
        sorted_losses.extend(epoch_losses)
        batches_by_epoch.append(sorted_batches)

    next_epochs = sorted_epochs[1:].copy()
    next_epochs.append(next_epochs[-1] + 1)
    epoch_progression = []

    for start, end, batches in zip(sorted_epochs, next_epochs, batches_by_epoch):
        # Change batch number into percentage of progression within current epoch
        batches = np.array(batches).astype(np.float32) / float(
            max_batches_by_epoch[start]
        )
        epoch_diff = end - start
        batches = batches * epoch_diff + start
        epoch_progression.append(batches)

    epoch_progression = np.concatenate(epoch_progression)
    sorted_losses = np.array(sorted_losses)

    fig = plt.figure(figsize=(10, 8), dpi=200)
    plt.plot(epoch_progression, sorted_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GrowSP loss")
    plt.show()

    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pass
        filename = os.path.join(save_path, _generate_filename(log_path, "loss"))
        fig.savefig(filename)


def visualize_accuracy(
    type: str,
    log_path: str,
    first_epoch: int = 1,
    last_epoch: int = np.inf,
    keep_older_duplicate: bool = False,
    metric_lines: Tuple[int, int] = (1, 3),
    metric_names: List[str] = None,
    save: bool = False,
    save_path: str = "./figures/metrics",
) -> None:
    warnings.showwarning = misc.showwarning
    type = type.lower()
    assert type in ["superpoint", "primitive", "test"], (
        "'type' must be one of ['superpoint', 'primitive', 'test']"
    )

    metrics_by_epoch = {}
    current_epoch = np.nan
    reading_metric = False
    current_metric_line = 0
    metric_keys = None

    with open(log_path) as fp:
        log_lines = fp.readlines()
        for line in log_lines:
            line = line.strip()

            if reading_metric:
                current_metric_line += 1
                if current_metric_line == metric_lines[0]:
                    metric_keys = line.split(" ")
                    metric_keys = [
                        key.strip() for key in metric_keys if key.strip() != ""
                    ]
                elif current_metric_line == metric_lines[1]:
                    try:
                        metrics = line.split(" ")
                        metrics = [
                            float(metric.strip())
                            for metric in metrics
                            if metric.strip() != ""
                        ]
                    except Exception:
                        pass  # Failed to parse metrics (most likely due to incorrect formatting)
                    if len(metric_keys) == len(metrics):
                        contains_epoch = metrics_by_epoch.get(current_epoch, False)
                        if not keep_older_duplicate or (
                            not contains_epoch and keep_older_duplicate
                        ):
                            metrics_by_epoch[current_epoch] = dict(
                                zip(metric_keys, metrics)
                            )
                    else:
                        warnings.warn(
                            f"Number of metrics for epoch {current_epoch} does not match the number of metric names. Ignoring metrics.",
                            RuntimeWarning,
                        )
                    # End of metric lines reached, reset variables
                    reading_metric = False
                    current_metric_line = 0
                    metric_keys = None
            else:
                split_line = line.split(" | ")
                if len(split_line) > 1:
                    try:
                        current_epoch = int(split_line[0].split("epoch")[1].strip())
                    except Exception:
                        pass  # Line did not contain epoch information (at least not in the correct format)
                else:
                    split_line = line.lower().split(type + " accuracy")
                    if len(split_line) > 1:
                        # The next lines will contain accuracy metric information
                        reading_metric = True

    # Extract metrics in a sorted order
    sorted_epochs = list(metrics_by_epoch.keys())
    sorted_epochs.sort()
    data_metric_names = []
    # Extract all possible metric names
    for epoch in sorted_epochs:
        keys = list(metrics_by_epoch[epoch].keys())
        for key in keys:
            if key not in data_metric_names:
                data_metric_names.append(key)
    # If metric_names provided, filter out undesired metrics
    if metric_names is not None:
        data_metric_names = [name for name in metric_names if name in metric_names]
    metric_names = data_metric_names

    # Plot metrics
    fig = plt.figure(figsize=(10, 8), dpi=200)
    for name in metric_names:
        values = [metrics_by_epoch[epoch].get(name, np.nan) for epoch in sorted_epochs]
        plt.plot(sorted_epochs, values)

    plt.xlabel("Epoch")
    plt.title("GrowSP accuracy metrics")
    plt.legend(metric_names)
    plt.show()

    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pass
        filename = os.path.join(save_path, _generate_filename(log_path, "loss"))
        fig.savefig(filename)


def _generate_filename(
    path: str, default_name: str, ckpt_dirname: str = "ckpt", log_dirname: str = "logs"
):
    filename = default_name
    path = Path(path)
    path_parts = list(path.parts)
    # Remove checkpoint and log directories from the path to create a filename
    path_parts = reversed(
        list(takewhile(lambda x: x != ckpt_dirname, reversed(path_parts)))
    )
    path_parts = list(takewhile(lambda x: x != log_dirname, path_parts))
    if len(path_parts) > 0:
        filename = "_".join(path_parts)

    return filename
