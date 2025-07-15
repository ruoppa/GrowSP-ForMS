import os
import random
import laspy
import torch

import numpy as np

from enum import Enum, unique
from typing import List, Tuple, Union
from pathlib import Path
from collections import abc


@unique
class Mode(Enum):
    """Enum class for dataloader mode. Supported modes are as follows:

        train (str): training mode
        cluster (str): clustering mode
        eval (str): evaluation mode
        test (str): otherwise same as eval, but use test data
        eval_unlabeled (str): evaluation for unlabeled data

    Other modes can be added manually, but the string representation should be unique
    """

    train = "train"
    cluster = "cluster"
    eval = "eval"
    test = "test"
    eval_unlabeled = "eval_unlabeled"


@unique
class data_type(Enum):
    """Enum class for dataloader data type. Supported types are as follows

    sparse (str): sparse data
    dense (str): dense data
    """

    sparse = "sparse"
    dense = "dense"


def showwarning(
    message: str, category=RuntimeWarning, filename="", lineno=-1, *args, **kwargs
) -> None:
    """Custom warning function to disable showing the source line when warnings are displayed"""
    print(f"{category.__name__}: {message}")


def filter_files_by_id_and_num(
    file_list: List[str],
    id_list: List[str],
    id_dict: dict,
    remove: bool = False,
    ignore_files: bool = False,
) -> List[str]:
    """Filter a list of files by an id/number combo in the filename

    Args:
        file_list (List[str]): list of files to files
        id_list (List[str]): list of plot ids to consider
        id_dict (dict): dict where keys are permitted plot ids and values are lists of permitted numbers corresponding to that id
        remove (bool): If True, remove the items listed in id_dict as opposed to keeping them. Defaults to False.
        ignore_files (bool): If True, filter out files where the last thing in the name is 'ignore'

    Raises:
        ValueError: if filename does not adhere to the required format

    Returns:
        List[str]: list of filtered files
    """
    filtered_file_list = []
    for filename in file_list:
        split_filename = filename.split("_")
        plot_id_str = split_filename[1]
        num_str = split_filename[3]
        try:
            plot_id = int(plot_id_str)
        except ValueError as e:
            raise ValueError(
                f"Invalid plot id '{plot_id_str}' cannot be converted to int!"
            ) from e
        if plot_id not in id_list:
            continue
        try:
            num = int(num_str)
        except ValueError as e:
            raise ValueError(
                f"Invalid file number '{num_str}' cannot be converted to int!"
            ) from e
        num_list = id_dict.get(plot_id_str, [])
        permitted = np.isin(num, num_list)
        ignore = ignore_files and split_filename[-1] == "ignore"
        if np.logical_xor(permitted, remove) and not ignore:
            filtered_file_list.append(filename)

    return filtered_file_list


def filter_paths_by_id(
    path_list: List[str], id_list: List[int]
) -> Tuple[List[str], List[str]]:
    """Filter a list of paths by an id in the filename

    Args:
        path_list (List[str]): list of paths to .las files
        id_list (List[int]): list of permitted plot ids. Paths to files with id not in this list are filtered out

    Raises:
        ValueError: if filename does not contain an integer id in the correct spot

    Returns:
        Tuple[List[str], List[str]]: list of filtered paths and list of filename templates for each path (used when saving preprocessing results)
    """
    filtered_path_list, filename_template_list = [], []
    for path in path_list:
        file_prefix, plot_id = validate_plot_filename(path)
        if plot_id not in id_list:
            continue  # Paths where the id is not in the list of permitted ids are ignored
        filtered_path_list.append(path)
        filename_template_list.append(
            file_prefix + "_" + str(plot_id) + "_cylinder_{:n}.las"
        )

    return filtered_path_list, filename_template_list


def get_gt_label(
    path: str, id_list: List[int], pc: laspy.LasData
) -> Union[np.ndarray, None]:
    _, plot_id = validate_plot_filename(path)
    if plot_id in id_list:
        return pc.classification
    return None


def get_gt_labels(
    path_list: List[str], id_list: List[int], flatten: bool = True
) -> np.ndarray:
    """Get ground truth labels from .las files

    Args:
        path_list (List[str]): list of paths to .las files
        id_list (List[int]): list of permitted plot ids. Paths to files with id not in this list are ignored
        flatten (bool, optional): if True, flatten the labels into a single array. Otherwise a list of labels for each .las files is returned. Defaults to True.

    Returns:
        np.ndarray: ground truth labels
    """
    gt_labels = []
    for path in path_list:
        _, plot_id = validate_plot_filename(path)
        if plot_id in id_list:
            pc = laspy.read(path)
            gt_labels.append(pc.classification)
    if flatten:
        gt_labels = np.concatenate(gt_labels)
    return gt_labels


def validate_plot_filename(path: str) -> Tuple[str, int]:
    """Validate that the name of a .las file the given path leads to follows the required format.
    The format is as follows: <word>_<id>_*.las or <word>_<id>.las, e.g. someplot_1005_processed.las
    or simply someplot_1005.las

    Args:
        path (str): path to a .las file (doesn't validate whether it's actually a .las file)

    Raises:
        ValueError: if filename does not contain an integer id in the correct spot

    Returns:
        Tuple[str, int]: filename prefix (eg word prior to the id) as str and plot id as int
    """
    plot_filename = Path(path).name
    split_filename = plot_filename.split("_")
    plot_id_str = split_filename[1]
    try:
        plot_id = int(plot_id_str)
    except ValueError as e:
        try:
            plot_id = int(plot_id_str[:-4])
        except ValueError:
            raise ValueError(
                f"Invalid plot id '{plot_id_str}' cannot be converted to int!"
            ) from e

    return split_filename[0], plot_id


def prompt_bool(msg: str) -> bool:
    """Prompt the user to answer a yes/no question. The function prompts the user until a valid answer is given.

    Args:
        msg (str): the question the user should answer with yes or no

    Returns:
        bool: True if yes, False if no.
    """
    while True:
        answer = input(f"{msg} [y/n]: ")
        answer = answer.lower().strip()
        if answer == "y" or answer == "yes":
            return True
        elif answer == "n" or answer == "no":
            return False


def is_seq_of(seq, expected_type, seq_type=None) -> bool:
    """Check whether 'seq' is a sequence of some type

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: True if the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def concat_point_clouds(pc_a: laspy.LasData, pc_b: laspy.LasData) -> laspy.LasData:
    """Concatenate two laspy point cloud objects. NOTE: It is quite possible that this function will only work with .las format 7, I haven't tested
    any other formats. However, making it work with other formats should be quite straightforward

    Args:
        pc_a, pc_b (laspy.LasData): point cloud objects, assumed to be of the same .las point format, e.g.

    Returns:
        laspy.LasData: point cloud object created by concatenating the two point clouds given as input
    """
    # The function also works for point clouds that are simply numpy arrays (assuming the dimension
    # of the arrays are correct)
    if (type(pc_a) is np.ndarray) and (type(pc_b) is np.ndarray):
        pc_ab = np.concatenate([pc_a, pc_b])
    else:
        # Compute the difference of offsets between the two point clouds that are merged, the subtract
        # the difference from the xyz-coordinates of point cloud b. This ensures that the xyz-coordinates
        # of point cloud b remain unchanged once the offset is changed to that of point cloud a
        offset_diffs = pc_a.points.offsets - pc_b.points.offsets
        pc_b.xyz -= offset_diffs
        # Begin by pulling out the ScaleAwarePointRecord objects from each lasdata object. These objects contain all of the
        # non-header data, e.g. point coordinate, intensities, scan_angle, rgb and so on
        points_a = pc_a.points
        points_b = pc_b.points
        # Create a new lasdata object. Requires a header as an argument. Assuming that both point cloud objects are in the same
        # format, we can use either header
        pc_ab = laspy.LasData(pc_a.header)
        # Concatenate all of the data from pc_a and pc_b
        array = np.concatenate([points_a.array, points_b.array])
        # By default offsets, scales, point format and sub field dict are pulled from point cloud a. For this to work we again rely on the
        # assumption that the point clouds are in the same format
        # Create a new ScaleAwarePointRecord object from the concatenated data
        points_ab = laspy.point.record.ScaleAwarePointRecord(
            array, points_a.point_format, points_a.scales, points_a.offsets
        )
        # Add points to new lasdata object and update header
        pc_ab.points = points_ab
        pc_ab.update_header()
    return pc_ab


def set_seed(seed):
    """
    Set seed for every library we use tha has randomness (if possible)
    Unfortunately, backward() of [interpolate] functional seems to be never deterministic.

    Below are related threads:
    https://github.com/pytorch/pytorch/issues/7068
    https://discuss.pytorch.org/t/non-deterministic-behavior-of-pytorch-upsample-interpolate/42842?u=sbelharbi
    """
    # Use random seed.
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
