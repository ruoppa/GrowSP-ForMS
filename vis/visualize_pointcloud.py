import laspy
import os

import numpy as np

from util import PlyData
from vis import SUPERPOINT_COLORMAP
from .vis_util import get_evoms_rgb_array
from typing import Union


def create_superpoint_vis(
    points: Union[laspy.LasData, PlyData],
    labels: np.ndarray,
    path: str,
    name: str,
    return_pc: bool = False,
) -> Union[None, laspy.LasData, PlyData]:
    """Create a .las or .ply pointcloud with the superpoints for visualization purposes. Note that this will overwrite any rgb values
    in the original point clouds with superpoint label colors.

    Args:
        points (Union[laspy.LasData]): point cloud LasData or PlyData object
        labels (np.ndarray): [n x 1] array of superpoint labels
        path (str): path to the directory where the resulting pointcloud should be saved
        name (str): name of the file to save the result in (assumed to contain the filename extension)
        return_las (bool, optional): if True, return the modified point cloud object. Defaults to False.
    """
    pc = points
    points = points.xyz

    if not os.path.exists(path):
        os.makedirs(path)
    colors = np.zeros_like(points)
    for p in range(colors.shape[0]):
        colors[p] = 255 * (SUPERPOINT_COLORMAP[labels[p].astype(np.int32)])[:3]
    colors = colors.astype(np.uint8)

    dim_names = [_.name for _ in list(pc.header.point_format.standard_dimensions)]
    for i, c in enumerate(["red", "green", "blue"]):
        if c not in dim_names:
            pc.add_extra_dim(laspy.ExtraBytesParams(name=c, type=np.uint8))
        pc[c] = colors[:, i]
    pc.header.global_encoding.wkt = True
    pc.write(os.path.join(path, name))
    if return_pc:
        return pc


def create_evoms_prediction_vis(
    pc: Union[laspy.LasData, PlyData],
    pred_labels: np.ndarray,
    path: str,
    name: str,
    show_errors: bool = True,
) -> None:
    """Create a pointcloud visualization for an evoms point cloud based on the predicted labels

    Args:
        points (Union[laspy.LasData, PlyData]): point cloud (with )
        pred_labels (np.ndarray): predicted labels
        path (str): path to the directory where the visualization point cloud should be saved
        name (str): name of the visualization point cloud
        show_errors (bool, optional): Show points with incorrectly predicted labels with distinct colors. Note that this assumes the 'classification' field in 'points' contains the ground truth labels. Defaults to True.
    """
    pc.classification[pc.classification == 2] = (
        0  # merge 'understory' class from ground truth with the 'foliage' class
    )
    if show_errors:
        pred_labels_w_err = pc.classification.copy()
        # NOTE: this assumes that pc.classification contains the ground truth
        wood_fn_mask = (pc.classification == 1) & (
            pred_labels == 0
        )  # wood points where prediction was foliage
        foliage_fn_mask = (pc.classification == 0) & (
            pred_labels == 1
        )  # foliage points where prediction was wood
        pred_labels_w_err[wood_fn_mask] = 2
        pred_labels_w_err[foliage_fn_mask] = 3
        colors = get_evoms_rgb_array(pred_labels_w_err)
    else:
        colors = get_evoms_rgb_array(pred_labels)

    dim_names = [_.name for _ in list(pc.header.point_format.standard_dimensions)]
    for i, c in enumerate(["red", "green", "blue"]):
        if c not in dim_names:
            try:
                pc[c]
            except AttributeError:
                pc.add_extra_dim(laspy.ExtraBytesParams(name=c, type=np.uint8))
        pc[c] = colors[:, i]
    pc.header.global_encoding.wkt = True
    pc.add_extra_dim(laspy.ExtraBytesParams("pred", type=pc.classification.dtype))
    pc.pred = pred_labels  # This way we can look at points in individual classes separately in cloudcompare
    if show_errors:
        pc.add_extra_dim(
            laspy.ExtraBytesParams("pred_w_err", type=pc.classification.dtype)
        )
        pc.pred_w_err = pred_labels_w_err
    pc.write(os.path.join(path, name))
