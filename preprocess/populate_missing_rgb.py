import time
import os
import glob
import warnings
import argparse

import numpy as np

from pathlib import Path
from easydict import EasyDict
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import cKDTree
from scipy.stats import mode

from util import get_read_fn
from util.misc import filter_paths_by_id

# Warnings silenced to avoid warning spam when computing nanmean over an array with only nan values
warnings.filterwarnings("ignore")


def populate_missing_rgb(args: argparse.Namespace) -> None:
    """Populate missing rgb values (i.e. any rgb values that are 0) See comments on _populate_missing_rgb_one_pc for more information.
    Here rgb is assumed to contain reflectance, i.e. 0 means that the data is missing. If your rgb values are actual color information
    this function is probably not for you

    Args:
        args (argparse.Namespace): args passed to main
    """
    dataset_config = args.dataset_config
    dataset_config.verbose = args.verbose
    # Find the path of all .las/.ply files in the input directory
    path_list = sorted(
        glob.glob(
            os.path.join(
                dataset_config.UNPOPULATED_PATH, "*" + dataset_config.FILENAME_EXTENSION
            )
        )
    )
    path_list, _ = filter_paths_by_id(path_list, dataset_config.IDS)
    if len(path_list) < 1:
        warnings.warn(
            f"no {dataset_config.FILENAME_EXTENSION} files found in '{dataset_config.UNPOPULATED_PATH}', cannot construct superpoints!",
            RuntimeWarning,
        )
        return
    config_list = [dataset_config] * len(path_list)
    filename_list = [Path(path).name for path in path_list]
    read_fn = get_read_fn(dataset_config.FILENAME_EXTENSION)
    fn_list = [read_fn] * len(path_list)
    if not os.path.exists(dataset_config.INPUT_PATH):
        os.makedirs(dataset_config.INPUT_PATH)
    pool = ProcessPoolExecutor(max_workers=8)
    _ = list(
        pool.map(_populate_missing_rgb_one_pc, config_list, filename_list, fn_list)
    )


def _populate_missing_rgb_one_pc(config: EasyDict, filename: str, read_fn):
    """Populate the missing rgb values of once poit cloud. First, each missing rgb value is assigned the mean of the
    rgb values within the superpoint it belongs to. This can not be done, if e.g. all green values are missing within a superpoint.
    Therefore, after iterating through the points once and populating missing values where possible, we perform a second iteration.
    On the second iteration, we find the nearest superpoint with populated rgb values for all points that are within superpoints where
    rgb values could not be populated. For each unpopulated superpoint, we then choose the nearest populated superpoint by a majority vote
    (i.e. which superpoint was most commonly the nearest neighbor among all points within the current superpoint) and assign the mean rgb
    value of that superpoint to the unpopulated superpoint

    Args:
        config (EasyDict): dataset config
        filename (str): name of the current point cloud file to process
        read_fn (function): function for reading the point cloud from file
    """
    start_time = time.time()
    input_path = os.path.join(config.UNPOPULATED_PATH, filename)
    init_sp_path = os.path.join(
        config.SP_PATH, filename[: -len(config.FILENAME_EXTENSION)] + "_superpoints.npy"
    )
    output_path = os.path.join(config.INPUT_PATH, filename)
    try:
        pc = read_fn(input_path)
        init_sp = np.load(init_sp_path)
    except FileNotFoundError:
        return

    unq_sp = np.unique(init_sp)
    # Array where element at index i is True, if the superpoint at index i of unq_sp is missing all values for one or more colors
    missing_all = np.zeros_like(unq_sp, dtype=bool)
    # Mask over the initial superpoint labels that have all rgb values after iterating through once
    nonzero_mask = np.zeros_like(init_sp, dtype=bool)
    rgb = np.vstack((pc.red, pc.green, pc.blue)).T
    new_rgb = np.zeros_like(rgb)  # Array for saving the new rgb values

    for i, sp in enumerate(unq_sp):
        if sp != -1:
            sp_mask = init_sp == sp
            sp_rgb = (
                rgb[sp_mask, :].copy().astype(np.float32)
            )  # rgb values in current superpoint
            zero_mask = sp_rgb == 0  # Mask of missing values
            sp_rgb[zero_mask] = np.nan  # Set missing values to nan
            rgb_mean = np.nanmean(sp_rgb, axis=0)  # Compute mean
            if np.any(np.isnan(rgb_mean)):
                missing_all[i] = True
            else:
                nonzero_mask = nonzero_mask | sp_mask
            rgb_mean = np.tile(rgb_mean, (sp_rgb.shape[0], 1))
            sp_rgb[zero_mask] = rgb_mean[zero_mask]
            new_rgb[sp_mask] = sp_rgb

    # Compute the nearest neighbor of all points among the superpoints that no longer have any missing rgb values
    kdtree = cKDTree(pc.xyz[nonzero_mask])
    _, nearest_ind = kdtree.query(pc.xyz, k=1)
    try:
        # Label of the nearest superpoint for all points with missing rgb values
        nearest_sp = init_sp[nonzero_mask][nearest_ind]
    except IndexError:
        # In this case one or more of the rgb fields in the current cylinder is missing all values. As such,
        # the cylinder can not be populated. Such plots should most likely not be used at all
        print(
            f"{filename} missing all values for one or more of the rgb fields. Populating not possible!"
        )
        return

    for i, sp in enumerate(unq_sp):
        if missing_all[
            i
        ]:  # Only compute new rgb values if the superpoint is missing rgb values after the first iteration
            sp_mask = init_sp == sp
            sp_rgb = new_rgb[sp_mask, :].copy().astype(np.float32)
            zero_mask = sp_rgb == 0
            # Nearest superpoint with all colors for each point in the current superpoint
            nearest_sp_list = nearest_sp[sp_mask]
            # Find the most common superpoint label among the nearest neighbors, then compute the mean rgb values of that superpoint
            mode_sp = mode(nearest_sp_list, keepdims=True)[0][0]
            mode_sp_mask = init_sp == mode_sp
            mode_sp_rgb = new_rgb[mode_sp_mask, :].copy().astype(np.float32)
            rgb_mean = np.mean(mode_sp_rgb, axis=0)
            # Update rgb values
            rgb_mean = np.tile(rgb_mean, (sp_rgb.shape[0], 1))
            sp_rgb[zero_mask] = rgb_mean[zero_mask]
            new_rgb[sp_mask] = sp_rgb

    # Assign the newly populated rgb values to the original point cloud
    pc.red = new_rgb[:, 0]
    pc.green = new_rgb[:, 1]
    pc.blue = new_rgb[:, 2]
    # Save populated point cloud
    pc.write(output_path)

    if config.verbose:
        print(f"Completed '{filename}' in: {(time.time() - start_time):.3f} s")
