import pclpy
import os
import glob
import argparse
import time
import logging

import numpy as np
import numpy.matlib
import MinkowskiEngine as ME

from .cut_pursuit import libcp
from pclpy import pcl
from pathlib import Path
from easydict import EasyDict
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.stats import mode
from scipy.spatial import cKDTree
from typing import Union, Tuple, List
from jakteristics import compute_features

from util import get_read_fn
from util.logger import print_log
from util.config_yaml import cfg_from_yaml_file
from vis.visualize_pointcloud import create_superpoint_vis
from util.accuracy import (
    sp_labels_to_gt,
    get_accuracy,
    get_superpoint_boundary_accuracy,
    get_formatted_acc_str_w_boundary_metrics,
)
from util.misc import filter_paths_by_id, get_gt_label


def create_superpoints(
    args: argparse.Namespace, logger: logging.Logger = None, config: EasyDict = None
):
    """Construct initial superpoints from  point clouds based on config and arguments. The function is intended for processing the
    preprocessed data produced by preprocess_data()

    Args:
        args (argparse.Namespace): arguments passed to main.py
        logger (logging.Logger): logger object, only necessary in grid search mode. Defaults to None.
        config (EasyDict): config object. If none provided, the function reads the config from file. Defaults to None.
    """
    dataset_config = args.dataset_config
    if config is None:
        config = cfg_from_yaml_file(args.config_path)  # Get config
    config.verbose = args.verbose
    start_time = time.time()
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
        print(
            f"no {dataset_config.FILENAME_EXTENSION} files found in '{dataset_config.UNPOPULATED_PATH}', cannot construct superpoints!"
        )
        return
    if config.verbose:
        print_log(
            f"Constructing initial superpoints with method '{config.METHOD}'...", logger
        )
    if not os.path.exists(dataset_config.SP_PATH):
        os.makedirs(dataset_config.SP_PATH)
    vis_path = os.path.join(dataset_config.SP_PATH, "visualize")
    if config.VISUALIZE and not os.path.exists(vis_path):
        os.makedirs(vis_path)
    config.dataset_config = dataset_config
    read_fn = get_read_fn(dataset_config.FILENAME_EXTENSION)
    result = []
    for path in path_list:
        res = _create_superpoints(path, config, read_fn)
        result.append(res)

    if config.verbose:
        print_log(
            f"Finish constructing initial superpoints in {(time.time() - start_time):.3f} s",
            logger,
        )
        print_log("Computing superpoint accuracy metrics...", logger)
        start_time = time.time()
        gt_labels, sp_labels, sp_gt_labels, source, target = [], [], [], [], []
        total = 0
        counter = 0
        for gt, sp, sp_gt, n, src, tgt in result:
            if gt is not None:
                total += n
                counter += 1
                gt_labels.append(gt)
                sp_labels.append(sp)
                sp_gt_labels.append(sp_gt)
                source.append(src)
                target.append(tgt)
        # Compute superpoint boundary accuracy metrics
        br, bp = get_superpoint_boundary_accuracy(gt_labels, sp_labels, source, target)
        gt_labels, sp_gt_labels = (
            np.concatenate(gt_labels),
            np.concatenate(sp_gt_labels),
        )
        o_acc, m_acc, m_iou, iou_array, _ = get_accuracy(
            dataset_config.N_CLASSES, gt_labels, sp_gt_labels
        )
        formatted_acc_str = get_formatted_acc_str_w_boundary_metrics(
            o_acc,
            m_acc,
            m_iou,
            iou_array,
            br,
            bp,
            dataset_config.N_CLASSES,
            dataset_config.LABELS,
            "Superpoint accuracy",
        )
        print_log(
            f"Finish computing accuracy metrics in {(time.time() - start_time):.3f} s",
            logger,
        )
        print_log(
            f"Average number of superpoints / point cloud: {(total / counter):.3f}",
            logger,
        )
        print_log(formatted_acc_str, logger)


def _create_superpoints(
    path: str, config: EasyDict, read_fn, save: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """Handles the initial superpoint construction for one individual point cloud

    Args:
        path (str): path to the point cloud
        config (EasyDict): initial superpoint config file as an EasyDict object
        read_fn (function): function that takes as an argument a path to pointcloud and returns either a LasData or a PlyData object
        save (bool): if True, save results. Defaults to True.

    Raises:
        ValueError: if no superpoint constructor matches the mode specified in config

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]: ground truth labels, superpoint labels, groudn truth labels for each superpoint based on majority voting, number of superpoints, neighborhood graph source and target indices
    """
    start_time = time.time()
    pc = read_fn(path)
    pc.classification[pc.classification == config.dataset_config.IGNORE_LABEL] = -1
    path = Path(path)
    if config.METHOD == "vccs":
        sp_labels, n_superpoints = create_vccs_superpoints(
            pc.xyz.copy(),
            np.vstack((pc.red, pc.green, pc.blue)).T.copy(),
            pc.classification.copy(),
            config.VOXEL_SIZE,
            config.W_XYZ,
            config.W_NORMALS,
            config.W_RGB,
            return_n_sp=True,
        )
    elif config.METHOD == "cutpursuit":
        sp_labels, n_superpoints = create_cutpursuit_superpoints(
            pc,
            np.vstack((pc.red, pc.green, pc.blue)).T.copy(),
            config.FEATURE_NAMES,
            config.N_NEIGHBORS,
            config.SEARCH_RADIUS,
            config.MIN_POINTS,
            config.LAMBDA,
            config.W_RGB,
            config.W_GEOF,
            return_n_sp=True,
            apply_merging=config.APPLY_MERGING,
        )
    else:
        raise ValueError(
            f"No superpoint constructor found for method '{config.METHOD}'"
        )
    if save and sp_labels is not None:
        # Save superpoints in .npy file
        np.save(
            os.path.join(
                config.dataset_config.SP_PATH, path.name[:-4] + "_superpoints.npy"
            ),
            sp_labels,
        )
        if config.VISUALIZE:
            vis_path = os.path.join(config.dataset_config.SP_PATH, "visualize")
            create_superpoint_vis(
                pc, sp_labels, vis_path, path.name
            )  # Save .las/.ply file with superpoint labels if required

    if config.verbose:
        if config.dataset_config.NAME == "EvoMS":
            gt_labels = get_gt_label(path, config.dataset_config.LABELED_IDS, pc)
        elif config.dataset_config.NAME == "S3DIS":
            gt_labels = pc.classification  # in S3DIS, all data is labeled
        else:
            # TODO: Essentially all you need to do is implement a function that returns a numpy array of class labels for the current
            # point cloud if such labels are available and None otherwise (should be quite straightforward to do)
            raise NotImplementedError(
                f"Fetching class labels not implemented for dataset '{config.dataset_config.NAME}'!"
            )
        if gt_labels is not None and config.dataset_config.NAME == "EvoMS":
            # Cast understory labels to foliage
            gt_labels[gt_labels == 2] = 0
        sp_gt_labels = sp_labels_to_gt(gt_labels, sp_labels)
        print(f"Completed '{path.name}' in: {(time.time() - start_time):.3f} s")
        if gt_labels is not None:
            # Fetch neighborhood graph for computing boundary recall and boundary precision
            source, target, _ = _get_knn(pc.xyz, config.N_NEIGHBORS)
        else:
            source, target = None, None

        return gt_labels, sp_labels, sp_gt_labels, n_superpoints, source, target


def create_vccs_superpoints(
    points: np.ndarray,
    rgb: np.ndarray,
    labels: np.ndarray,
    voxel_size: float = 0.05,
    w_xyz: float = 0.4,
    w_normals: float = 1,
    w_rgb: float = 0.2,
    return_n_sp: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
    """Create superpoints using VCCS and region growing. This is the superpoint constructor used in the original GrowSP paper

    Args:
        points: (np.ndarray): [n x 3] array of xyz-coordinates
        rgb (np.ndarray, optional):[n x 3] array of rgb values. Defaults to None.
        voxel_size (float, optional): voxel size. Defaults to 0.05.
        w_xyz (float, optional): weight for coordinates in supervoxel clustering. Defaults to 0.4.
        w_normals (float, optional): weight for normals in supervoxel clustering. Defaults to 1.
        w_rgb (float, optional): weight for rgb values in supervoxel clustering. Defaults to 0.2.
        return_n_sp (bool, optional): if True, return the number of superpoints found. Defaults to False.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, int]]: superpoint labels and number of superpoints if return_n_sp is set to True
    """
    # Quantize
    points -= points.mean(0).copy()
    scale = 1 / voxel_size
    points = np.floor(points * scale)
    points, rgb, labels, _, inverse_map = ME.utils.sparse_quantize(
        np.ascontiguousarray(points),
        rgb,
        labels=labels,
        ignore_label=-1,
        return_index=True,
        return_inverse=True,
    )
    points = points.numpy().astype(np.float32)

    # VCCS
    voxel_labels = _supervoxel_clustering(points, rgb, w_xyz, w_normals, w_rgb)

    # Region growing
    # returns clusters from region growing algorithm
    clusters = _region_growing_simple(points)
    # array for mapping region numbers to correct index (original voxel index)
    region_idx = -np.ones_like(labels, dtype=np.float32)
    for region in range(len(clusters)):
        for point_idx in clusters[region].indices:
            region_idx[point_idx] = (
                region  # save at the original index region number/identifier
            )

    # Merging
    merged = -np.ones_like(labels, dtype=np.float64)
    voxel_labels[voxel_labels != -1] += len(clusters)
    for label in np.unique(voxel_labels):
        sp_mask = label == voxel_labels
        # Count which regions appear in the current superpoint
        sp_to_region = region_idx[sp_mask]
        dominant_region = mode(sp_to_region, keepdims=True)[0][0]
        # If majority (>50 %) of the region labels within the vccs cluster are the same, the entire vccs cluster is labeled
        # with the label in question. (e.g. >50% has label 'x' -> all voxels labeled with 'x')
        if (dominant_region == sp_to_region).sum() > sp_to_region.shape[0] * 0.5:
            merged[sp_mask] = (
                dominant_region  # replace vccs labels with region growing labels
            )
        else:
            merged[sp_mask] = (
                label  # if no dominant region inside vccs cluster, keep the original labels
            )

    # Construct continuous superpoint labels
    sp_labels = -np.ones_like(merged, dtype=np.int32)
    count_num = 0
    for m in np.unique(merged):
        if m != -1:
            sp_labels[merged == m] = count_num
            count_num += 1

    # Create a label for all points that were not assigned to any superpoint (label -1) using DBSCAN
    try:
        unassigned_points_mask = sp_labels == -1
        unassigned_points_coords = points[unassigned_points_mask]
        # Apply DBSCAN to the unassigned points
        dbscan = DBSCAN(eps=0.2, min_samples=1)
        dbscan_labels = dbscan.fit_predict(unassigned_points_coords)
        # Update the voxel_idx for the unassigned points with DBSCAN labels
        sp_labels[unassigned_points_mask] = dbscan_labels + (
            len(np.unique(sp_labels)) - 1
        )
    except Exception:
        pass

    # Map to original shape
    sp_labels = sp_labels[inverse_map]
    n_superpoints = len(np.unique(sp_labels))

    if return_n_sp:
        return sp_labels, n_superpoints
    else:
        return sp_labels


def _supervoxel_clustering(
    coords: np.ndarray,
    rgb: np.ndarray = None,
    w_xyz: float = 0.4,
    w_normals: float = 1,
    w_rgb: float = 0.2,
) -> np.ndarray:
    """Perform supervoxel clustering for the given points

    Args:
        points: (np.ndarray): [n x 3] array of xyz-coordinates
        rgb (np.ndarray, optional):[n x 3] array of rgb values. Defaults to None.
        w_xyz (float, optional): weight for coordinates in supervoxel clustering. Defaults to 0.4.
        w_normals (float, optional): weight for normals in supervoxel clustering. Defaults to 1.
        w_rgb (float, optional): weight for rgb values in supervoxel clustering. Defaults to 0.2.

    Returns:
        np.ndarray: [n x 1] array voxel labels
    """
    pc = pcl.PointCloud.PointXYZRGBA(coords, rgb)
    normals = pc.compute_normals(radius=3, num_threads=2)
    vox = pcl.segmentation.SupervoxelClustering.PointXYZRGBA(
        voxel_resolution=1, seed_resolution=10
    )
    vox.setInputCloud(pc)
    vox.setNormalCloud(normals)
    vox.setSpatialImportance(w_xyz)
    vox.setNormalImportance(w_normals)
    vox.setColorImportance(w_rgb)
    output = pcl.vectors.map_uint32t_PointXYZRGBA()
    vox.extract(output)
    sp = vox.getLabeledCloud()
    sp_labels = sp.label.astype(np.float32)
    return sp_labels - 1.0


def _region_growing_simple(points: np.ndarray) -> np.ndarray:
    """Apply region growing algorithm for the given points

    Args:
        points (np.ndarray): [n x 3] array of xyz-coordinates

    Returns:
        np.ndarray: [n x 1] array of region labels
    """
    pc = pcl.PointCloud.PointXYZ(points)
    normals = pc.compute_normals(radius=3, num_threads=2)
    clusters = pclpy.region_growing(
        pc,
        normals=normals,
        min_size=1,
        max_size=100000,
        n_neighbours=15,
        smooth_threshold=3,
        curvature_threshold=1,
        residual_threshold=1,
    )
    return clusters


def create_cutpursuit_superpoints(
    pc: np.ndarray,
    rgb: np.ndarray,
    feature_names: List[str] = ["verticality", "linearity", "PCA1"],
    n_neighbors: int = 10,
    search_radius: float = 0.35,
    min_points: int = 5,
    cp_lambda: float = 0.08,
    w_rgb: float = 8.0,
    w_geof: float = 10.0,
    return_n_sp: bool = False,
    apply_merging: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
    """Create superpoints using the L0 cut pursuit algorithm on a graph constructed from the point cloud. The graph is constructed
    such that edges are added between the k nearest neighbor nodes. The node values are set to geometric features computed from the
    point cloud (and possibly rgb). NOTE: currently using rgb for node values is commented out as it was not used in the paper

    Args:
        points (np.ndarray): xyz-coordinates
        rgb (np.ndarray): rgb values
        feature_names (List[str], optional): Names of geometric features to use for nodes. Defaults to ["verticality", "linearity", "PCA1"].
        n_neighbors (int, optional): Number of nearest neighbors used when creating the graph. Defaults to 10.
        search_radius (float, optional): Search radius when computing the geometric features. Defaults to 0.35.
        min_points (int, optional): Absolute minimum number of points that a superpoint must contain to not be considered "small".
            Note that the function uses an adaptive threshold based on the number of superpoints in the point lcoud. Defaults to 5.
        cp_lambda (float, optional): Value of lambda for the cut pursuit algorithm. Defaults to 0.08.
        w_rgb (float, optional): Weight given to rgb values in the nodes. Defaults to 8..
        w_geof (float, optional): Weight given to geometric features in the nodes. Defaults to 10..
        return_n_sp (bool, optional): If True, return the number of superpoints in addition to the superpoint labels. Defaults to False.
        apply_merging (bool, optional): If True, apply the simple superpoint merging proposed in the paper to reduce the number of superpoints. Defaults to True.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, int]]: superpoint label for each point and number of superpoints if return_n_sp is set to True.
    """
    points = pc.xyz
    if len(points) <= min_points:
        return None, 0  # Superpoints cannot be generated, point cloud too small

    # Scale rgb values to the interval [0, 1]
    rgb_scale = float(np.iinfo(rgb.dtype).max)
    rgb = rgb.astype(np.float32)
    rgb /= rgb_scale
    # Find nearest neighbors
    source, target, distances = _get_knn(points, n_neighbors)
    # Compute geometric features
    geof = compute_features(
        points, search_radius=search_radius, feature_names=feature_names
    )
    geof[np.isnan(geof)] = 0
    # Weight between neighbors is the inverse of their distance
    edge_weight = np.array(1.0 / distances, dtype=np.float32)
    # NOTE: every array is converted to float32, since cut pursuit does not support float64 (don't know why)
    # Feature vector is a weighted combination of rgb and geometric features
    features = (w_geof * geof).astype(
        np.float32
    )  # np.hstack((rgb * w_rgb, geof * w_geof)).astype(np.float32)
    # Apply L0-cut pursuit
    _, sp_labels = libcp.cutpursuit(features, source, target, edge_weight, cp_lambda)
    if apply_merging:
        # Merge small superpoints
        min_points = _get_adaptive_threshold(sp_labels, min_points)
        singular_sp_mask, small_sp_mask = _get_small_sp_mask(sp_labels, min_points)
        labeled_mask = ~(
            singular_sp_mask | small_sp_mask
        )  # Large superpoints are considered labeled, small are unlabeled
        # If no superpoints are considered labeled, decrease min_points until at least some are
        while np.sum(labeled_mask) < 1:
            min_points -= 1
            singular_sp_mask, small_sp_mask = _get_small_sp_mask(sp_labels, min_points)
            labeled_mask = ~(singular_sp_mask | small_sp_mask)
        singular_sp_labels = _cluster_singular_sp(
            points[singular_sp_mask], 0.3, ["verticality", "linearity", "PCA1"]
        )
        small_sp_labels = sp_labels[small_sp_mask]
        # Map small and singular masks to only unlabeled points
        singular_sp_mask = singular_sp_mask[~labeled_mask]
        small_sp_mask = small_sp_mask[~labeled_mask]
        # Map small and sigular superpoint to the nearest large superpoints
        sp_labels = _map_to_nearest_superpoint(
            sp_labels[labeled_mask],
            points,
            labeled_mask,
            singular_sp_mask,
            small_sp_mask,
            singular_sp_labels,
            small_sp_labels,
        )

    if return_n_sp:
        return sp_labels, len(np.unique(sp_labels))
    else:
        return sp_labels


def _get_small_sp_mask(
    sp_labels: np.ndarray, min_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Get boolean masks that point to singular (less than three points) and small (less than min_points) superpoints

    Args:
        sp_labels (np.ndarray): superpoint labels from the graph search
        min_points (int): minimum points in a superpoints. Non-singular superpoints with less points are considered small

    Returns:
        Tuple[np.ndarray, np.ndarray]: boolean array pointing to points that are part of singular superpoints and boolean array array pointing to points that are part of small (but not singular) superpoints
    """
    sp_sizes = np.bincount(sp_labels)  # Count superpoint sizes
    singular_sp = np.where(sp_sizes < 3)  # Superpoints with at most two points
    singular_sp_mask = np.isin(sp_labels, singular_sp)
    small_sp = np.where((2 < sp_sizes) & (sp_sizes < min_points))
    small_sp_mask = np.isin(sp_labels, small_sp)

    return singular_sp_mask, small_sp_mask


def _cluster_singular_sp(
    points: np.ndarray,
    search_radius: float = 0.3,
    feature_names: List[str] = ["verticality"],
    eps: float = 0.2,
) -> np.ndarray:
    """Cluster points that are part of singular superpoints using DBSCAN. xyz-coordinates and chosen features are used
    for clustering

    Args:
        points (np.ndarray): points (that are part of singular superpoints)
        search_radius (float, optional): search radius to use for feature computations. Defaults to 0.3.
        feature_names (List[str], optional): features to use in clustering. Defaults to ["verticality"].
        eps (float, optional): eps for DBSCAN. Defaults to 0.2.

    Returns:
        np.ndarray: labels for each cluster (points with no cluster are labeled -1)
    """
    features = compute_features(
        points, search_radius=search_radius, feature_names=feature_names
    )
    features[np.isnan(features)] = 0
    points = np.concatenate((points, features), axis=1)
    labels = DBSCAN(eps=eps).fit_predict(points)

    return labels


def _map_to_nearest_superpoint(
    superpoint_labels: np.ndarray,
    points: np.ndarray,
    labeled_mask: np.ndarray,
    singular_sp_mask: np.ndarray,
    small_sp_mask: np.ndarray,
    singular_sp_labels: np.ndarray,
    small_sp_labels: np.ndarray,
) -> np.ndarray:
    """Map small and singular superpoints to the nearest large superpoints

    Args:
        superpoint_labels (np.ndarray): superpoint labels of the large superpoints
        points (np.ndarray): all xyz-coordinates
        labeled_mask (np.ndarray): boolean mask of the labeled points (i.e. points that are part of the large superpoints)
        singular_sp_mask (np.ndarray): boolean mask of points in singular superpoints (less than 3 points)
        small_sp_mask (np.ndarray): boolean mask of points in small superpoints (at least 3 and less than min_points)
        singular_sp_labels (np.ndarray): labels of the singular points (output of _cluster_singular_sp())
        small_sp_labels (np.ndarray): superpoint labels of the small superpoints

    Returns:
        np.ndarray: superpoint labels for all points, such that small and singular superpoints have been mapped to the larger superpoints
    """
    sp_labels = np.zeros_like(
        labeled_mask, dtype=np.int64
    )  # Array for the final superpoint labels
    # For each unlabeled point, find the closest label point (i.e. point that is part of a superpoint)
    kdtree = cKDTree(points[labeled_mask])
    _, nearest_ind = kdtree.query(points[~labeled_mask], k=1)
    nearest_ind[nearest_ind == superpoint_labels.shape[0]] = 0
    # For each unlabeled point, give it the label of the closest superpoint
    new_labels = superpoint_labels[nearest_ind]

    # For each unlabeled point that was part of a small superpoint, give it the label that most frequently occurs among
    # the points in the same small superpoint
    for sp in np.unique(small_sp_labels):
        if sp != -1:
            mask = small_sp_labels == sp
            new_labels[small_sp_mask][mask] = mode(
                new_labels[small_sp_mask][mask], keepdims=True
            )[0][0]
    # For each unlabeled point that was part of a singular superpoint and was assigned to a cluster, give it the label that
    # most frequently occurs among the points in the same cluster
    for sp in np.unique(singular_sp_labels):
        if sp != -1:
            mask = singular_sp_labels == sp
            new_labels[singular_sp_mask][mask] = mode(
                new_labels[singular_sp_mask][mask], keepdims=True
            )[0][0]

    sp_labels[labeled_mask] = superpoint_labels
    sp_labels[~labeled_mask] = new_labels
    # Map labels to a continuous range
    unq_labels = np.unique(sp_labels)
    sp_labels = np.searchsorted(unq_labels, sp_labels)

    return sp_labels


def _get_knn(
    points: np.ndarray, n_neighbors: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the k nearest neighbors for each point

    Args:
        points (np.ndarray): xyz-coordinates
        n_neighbors (int, optional): number of neighbors to find. Defaults to 10.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: indices of source point, indices of target point and distances (each source-target index pair forms a neighbor)
    """
    # Find the nearest neighbors for each point
    n_points = points.shape[0]
    if n_points <= n_neighbors:
        n_neighbors = n_points - 1
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="kd_tree").fit(points)
    distances, neighbors = knn.kneighbors(points)
    del knn
    neighbors = neighbors[:, 1:]
    distances = distances[:, 1:]
    source = (
        np.matlib.repmat(range(0, n_points), n_neighbors, 1)
        .flatten(order="F")
        .astype(np.uint32)
    )
    target = np.transpose(neighbors.flatten(order="C")).astype(np.uint32)
    distances = distances.flatten()

    # Remove possible remaining self edges
    self_edge_mask = source == target
    distances = distances[~self_edge_mask]
    source = source[~self_edge_mask]
    target = target[~self_edge_mask]
    # Some points may have zero distance between them despite being different points (this is generally due to floating point precission being too low)
    # We replace any distances that are zero with a very low distance
    zero_dist_mask = distances == 0
    distances[zero_dist_mask] = 1e-6

    return source, target, distances


def _get_adaptive_threshold(
    sp_labels: np.ndarray, min_points: int, max_superpoints: int = 2200
) -> int:
    """Find an adaptive threshold for the minimum number of points a superpoint must contain to not
    be considered "small". The threshold is adapted such that the number of superpoints remains under
    a certain threshold. We begin from a certain minimum value for the threshold (user defined) and iteratively
    increase the threshold until the number of superpoints for the given threshold is below max_points

    Args:
        sp_labels (np.ndarray): superpoint labels for each point
        min_points (int): initial minimum number of points a superpoint must contain to not be considered small
        max_superpoints (int, optional): Maximum number of superpoints per point cloud. Defaults to 2200.

    Returns:
        int: threshold for the minimum number of points a superpoint must contain to not be considered "small"
    """
    _, counts = np.unique(sp_labels, return_counts=True)
    n_superpoints = len(counts)
    n_small_superpoints = np.sum(counts <= min_points)
    while (n_superpoints - n_small_superpoints) > max_superpoints:
        min_points += 1
        n_small_superpoints = np.sum(counts <= min_points)

    return min_points
