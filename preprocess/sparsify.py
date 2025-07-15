import os, random
import laspy

import numpy as np

from scipy.spatial import ConvexHull
from typing import List
from pathlib import Path
# Implementation of sparsification mostly copied from the SegmentAnyTree repository: https://github.com/SmartForest-no/SegmentAnyTree/blob/main/nibio_sparsify/sparsify_las_based_sq_m.py


def sparsify_all(data_path_list: List[str], target_density_list: List[int], seed: int = 33):
    """Generate sparsified version of all point clouds in data_path_list for all densities in target_density_list

    Args:
        data_path_list (List[str]): list of paths to point cloud objects which should be sparsified
        target_density_list (List[int]): list of integers specifying all target densities (points/m^2)
        seed (int, optional): seed for randomness. Defaults to 33.
    """
    rng = np.random.default_rng(seed = seed)
    path_obj = Path(data_path_list[0])
    # Extremely janky, but it works I guess...
    target_path_template_list = [
        os.path.join(
            *path_obj.parts[:-3], path_obj.parts[-3] + f"_{target_density}", *path_obj.parts[-2:-1]
        ) for target_density in target_density_list
    ]
    # Ensure that target directories exist
    for target_path in target_path_template_list:
        if not os.path.exists(target_path):
            os.makedirs(target_path)
    for data_path in data_path_list:
        pc = laspy.read(data_path)
        current_density = _compute_pc_density(pc)
        path_obj = Path(data_path)
        for target_density, target_path_template in zip(target_density_list, target_path_template_list):
            if current_density <= target_density:
                print(f"Density of {path_obj.name} lower than target, using original point cloud ({current_density} <= {target_density})")
                sparsified_pc = pc
            else:
                sparsified_pc = sparsify(pc, target_density, current_density, rng)
            target_path = os.path.join(target_path_template, path_obj.name)
            # Save sparsfied point cloud
            sparsified_pc.write(target_path)


def get_mean_point_density(data_path_list: List[str]) -> float:
    """Find the average point density of the point clouds in the given data paths

    Args:
        data_path_list (List[str]): list of paths to point cloud objects

    Returns:
        float: average density (points/m^2) of the given point cloud objects
    """
    density_list = []
    for data_path in data_path_list:
        pc = laspy.read(data_path)
        current_density = _compute_pc_density(pc)
        density_list.append(current_density)
    mean_density = np.mean(density_list)

    return mean_density


def sparsify(pc: laspy.LasData, target_density: float, current_density: float, rng: np.random.Generator) -> laspy.LasData:
    """Sparsify a point cloud object to a given target density (points/m^2)

    Args:
        pc (laspy.LasData): point cloud object.
        target_density (float): target density of sparsified point cloud (points/m^2).
        current_density (float): current density of the point cloud object, output of _compute_pc_density().
        rng (np.random.Generator): numpy random generator object for random sampling points

    Returns:
        (laspy.LasData): sparsified point cloud
    """
    x = pc.x
    keep_count = int(len(x) * (target_density / current_density))
    sampled_indices = random.sample(range(len(x)), keep_count)
    sampled_indices = rng.choice(np.arange(0, len(x)), keep_count, replace = False)
    sparsified_pc = pc[sampled_indices]
    
    return sparsified_pc


def _compute_pc_density(pc: laspy.LasData) -> float:
    """Compute point density (in points/m^2) of the given point cloud (assumes point cloud coordinates are in meters)

    Args:
        pc (laspy.LasData): point cloud object

    Returns:
        (float): density of the point cloud in points/m^2
    """
    pc_xy = pc.xyz[:, :2]
    ch = ConvexHull(pc_xy)
    area = ch.volume # When points are 2D, the volume property returns the area when using scipy ConvexHull
    point_count = len(pc_xy)
    density = point_count / area

    return density