import os
import glob
import argparse
import math
import laspy
import smallestenclosingcircle

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from typing import List, Tuple, Dict
from tqdm import tqdm
from scipy.spatial import ConvexHull
from easydict import EasyDict
from jakteristics import compute_features

from util.config_yaml import cfg_from_yaml_file
from util.misc import filter_paths_by_id, concat_point_clouds

# Constants
QUARTER_ROTATION = np.pi / 2
ONE_SIXTH_ROTATION = np.pi / 3


def preprocess_data(args: argparse.Namespace) -> None:
    """Preprocess raw point clouds. A preprocessing method should be implemented sepparately for each dataset

    Args:
        args (argparse.Namespace): arguments passed to main.py
    """
    dataset_config = args.dataset_config
    config = cfg_from_yaml_file(args.config_path)  # Get config

    if dataset_config.NAME == "EvoMS":
        path_list = glob.glob(
            os.path.join(
                dataset_config.RAW_PATH, "*" + dataset_config.FILENAME_EXTENSION
            )
        )
        preprocess_forest_plots(path_list, args, config)
    else:
        # ADD PREPROCESSING FOR YOUR OWN DATASET HERE
        raise NotImplementedError(
            f"No preprocessing implemented for dataset '{dataset_config.NAME}'!"
        )


def preprocess_forest_plots(
    path_list: List[str], args: argparse.Namespace, config: EasyDict
) -> None:
    """Preprocess raw forest plot point clouds based on config and arguments. The function cuts each raw point cloud into
    smaller cylindrical point clouds and then saves them in a directory determined by the config file

    Args:
        path_list (List[str]): list of paths to forest plot point clouds
        args (argparse.Namespace): arguments passed to main.py
        config (EasyDict): preprocessing config from yaml file
    """
    dataset_config = args.dataset_config
    id_list = dataset_config.IDS

    if not config.PROCESS_ALL:
        path_list, filename_template_list = filter_paths_by_id(path_list, id_list)
    if len(path_list) < 1:
        print(
            f"no .las files with permitted id found in '{dataset_config.RAW_PATH}', nothing to preprocess!"
        )
        return
    if not os.path.exists(dataset_config.UNPOPULATED_PATH):
        os.makedirs(dataset_config.UNPOPULATED_PATH)
    if args.verbose:
        path_list = tqdm(
            zip(path_list, filename_template_list),
            desc="Processing plots",
            total=len(path_list),
        )
    else:
        path_list = zip(path_list, filename_template_list)
    for path, filename_template in path_list:
        pc = laspy.read(path)
        # Normalize point cloud
        # Need to change scales, otherwise overflow error occurs
        pc.change_scaling([0.01, 0.01, 0.001], offsets=[0, 0, 0])
        x_min = np.min(pc.X)
        y_min = np.min(pc.Y)
        pc.X -= x_min
        pc.Y -= y_min

        if config.CIRCULAR_AREA:
            # Find smallest enclosing circle for data
            x, y, area_radius = min_enclosing_circle(pc)
            # Create circles for cylinders
            circle_centers, overlap_dict = create_circles(
                radius=config.RADIUS,
                circular_area=True,
                center=[x, y],
                area_radius=area_radius,
            )
        else:
            x_min, y_min, _ = np.min(pc.xyz, axis=0)
            x_max, y_max, _ = np.max(pc.xyz, axis=0)
            # Create circles for cylinders
            circle_centers, overlap_dict = create_circles(
                np.ndarray([x_min, x_max]),
                np.ndarray([y_min, y_max]),
                radius=config.RADIUS,
            )

        # Cut cylinders from point cloud and save them in separate files
        cut_cylinders(
            pc,
            circle_centers,
            overlap_dict,
            config.RADIUS,
            dataset_config.UNPOPULATED_PATH,
            filename_template,
            config.COMPUTE_NORMALS,
            config.COMPUTE_GEOMETRIC,
            config.GEOF_RADIUS,
            config.FEATURE_NAMES,
            config.SMOOTH_FEATURES,
            config.N_NEIGHBORS,
        )


def create_circles(
    x_dim: np.ndarray = None,
    y_dim: np.ndarray = None,
    radius: float = 1,
    circular_area: bool = False,
    center: np.ndarray = None,
    area_radius: float = None,
) -> Tuple[List[List[int]], Dict[int, List[int]]]:
    """Create circles in a pattern that covers the given area, such that the circles have minimal overlap. For a short
    explanation of the pattern, see: https://en.wikipedia.org/wiki/Overlapping_circles_grid#Other_variations

    Args:
        x_dim (np.ndarray, optional):  minimum and maximum values of x-coordinates in the area to cover [x_min, x_max]. Defaults to None.
        y_dim (np.ndarray, optional): minimum and maximum values of y-coordinates in the area to cover [y_min, y_max]. Defaults to None.
        radius (float, optional): radius used for the circles. Defaults to 1.
        circular_area (bool, optional): If True, cover circle instead of corners. Defaults to False.
        center (np.ndarray, optional): center of circular area to cover (if circular_area = True). Defaults to None.
        area_radius (float, optional): radius of circular area to cover (if circular_area = True). Defaults to None.

    Returns:
        (Tuple[List[List[int]], Dict[int, List[int]]]): list of circle center coordinates, and dict with indices of overlapping circles for each center
    """
    if circular_area:
        assert center is not None, "'circular_area' is True, but 'center' is None"
        assert area_radius is not None, (
            "'circular_area' is True, but 'area_radius' is None"
        )
    else:
        assert x_dim is not None, "'x_dim' is None"
        assert y_dim is not None, "'y_dim' is None"

    if circular_area:
        x_center, y_center = center
    else:
        x_min, x_max = x_dim[0], x_dim[1]
        y_min, y_max = y_dim[0], y_dim[1]
        x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
    center_dist = radius * np.sqrt(3)  # Distance between centers of adjacent circles
    circle_centers = [
        [x_center, y_center]
    ]  # First circle center in the middle of the plot
    iteration = 1
    range_start = 0
    range_end = 1
    while not _area_covered(
        circle_centers[range_start:range_end],
        x_dim,
        y_dim,
        radius,
        circular_area,
        center,
        area_radius,
    ):
        num_circles = 6 * iteration  # Number of circles to create on this iteration
        # Center of the initial circle in this iteration is directly above the original first circle
        x_init, y_init = x_center, y_center + iteration * center_dist
        circle_centers.append([x_init, y_init])
        angle = QUARTER_ROTATION - ONE_SIXTH_ROTATION  # Initial direction
        for i in range(1, num_circles):
            # For the math here, see e.g. https://math.stackexchange.com/questions/143932/calculate-point-given-x-y-angle-and-distance
            x_prev, y_prev = circle_centers[-1]
            x_next = x_prev + center_dist * np.cos(-angle)
            y_next = y_prev + center_dist * np.sin(-angle)
            circle_centers.append([x_next, y_next])
            # Change direction at the corners
            if i % iteration == 0:
                angle += ONE_SIXTH_ROTATION
        iteration += 1
        if circular_area:
            range_start = len(circle_centers) - num_circles
            range_end = range_start + iteration
        else:
            range_end += num_circles

    overlap_dict = {}
    num_circles = len(circle_centers)
    # For each circle, find the circles that overlap with it and save their indices in a dictionary
    for i in range(num_circles):
        x_center_i, y_center_i = circle_centers[i]
        overlap_indices = []
        for j in range(num_circles):
            if i == j:
                continue
            x_center_j, y_center_j = circle_centers[j]
            dist_ij = np.linalg.norm(
                [x_center_i - x_center_j, y_center_i - y_center_j], 2
            )
            if math.isclose(center_dist, dist_ij):
                # Circles overlap
                overlap_indices.append(j)
        # Save indices of all overlapping circles in a dictionary with the current circle index as key
        overlap_dict[i] = overlap_indices

    return circle_centers, overlap_dict


def _area_covered(
    circle_centers: list,
    x_dim: np.ndarray,
    y_dim: np.ndarray,
    radius: float,
    circular_area: bool,
    center: np.ndarray,
    area_radius: float,
) -> bool:
    """Check if the given circles cover the area defined by x_dim and y_dim

    Args:
        circle_centers (list): list of circle center coordinates
        x_dim (np.ndarray): min and max x-coordinates
        y_dim (np.ndarray): min and max y-coordinates
        radius (float): radius of circles
        circular_area (bool): If True, cover circle instead of corners
        center (np.ndarray): center of circular area to cover (if circular_area = True)
        area_radius (float): radius of circular area to cover (if circular_area = True)

    Returns:
        (bool): True if area covered, False otherwise
    """
    if circular_area:
        # If the maximum distance from the area center to the furthest intersection points
        # of each pair of circles in the list is >= the radius of the area, the circles cover
        # said area
        num_circles = len(circle_centers)
        if num_circles < 2:
            return (
                radius >= area_radius
            )  # Cannot compute intersection for only one circle
        center_pairs = zip(
            circle_centers[0 : (num_circles - 1)], circle_centers[1:num_circles]
        )
        for c0, c1 in center_pairs:
            max_dist = _max_intersect_dist(center, c0, c1, radius)
            if max_dist < area_radius:
                return False
        return True
    else:
        for x in x_dim:
            for y in y_dim:
                covered = False
                for c in circle_centers:
                    dist = np.linalg.norm([x - c[0], y - c[1]], 2)
                    if dist <= radius:
                        covered = True
                if not covered:
                    return False
        return True


def _max_intersect_dist(
    c_area: np.ndarray, c0: np.ndarray, c1: np.ndarray, radius: float
) -> float:
    """Compute the maximum distance from the center of the area to cover, to some intersection point of
    two circles. For math, see e.g.: http://paulbourke.net/geometry/circlesphere/ (Section titled "intersection of two circles")
    NOTE: we assume that the two circles have equal radius and always intersect

    Args:
        c_area (np.ndarray): center point of area to cover
        c0 (np.ndarray): center of circle 1
        c1 (np.ndarray): center of circle 2
        radius (float): radius of circles 1 and 2

    Returns:
        float: maximum distance from c_area to some intersection point of c0 and c1
    """
    xc, yc = c_area
    x0, y0 = c0
    x1, y1 = c1
    dist = np.linalg.norm([x0 - x1, y0 - y1], 2)
    a = dist / 2
    h = np.sqrt(radius**2 - a**2)
    x2 = x0 + a * (x1 - x0) / dist
    y2 = y0 + a * (y1 - y0) / dist
    # x3, y3 and x4, y4 are the two intersection points
    x3 = x2 + h * (y1 - y0) / dist
    x4 = x2 - h * (y1 - y0) / dist
    y3 = y2 - h * (x1 - x0) / dist
    y4 = y2 + h * (x1 - x0) / dist
    # Find maximum distance from intersection point to area center
    dist_xy3 = np.linalg.norm([xc - x3, yc - y3], 2)
    dist_xy4 = np.linalg.norm([xc - x4, yc - y4], 2)
    max_dist = np.max([dist_xy3, dist_xy4])

    return max_dist


def _plot_circles(
    circle_centers: list,
    x_dim: np.ndarray = None,
    y_dim: np.ndarray = None,
    radius: float = 1,
    circular_area: bool = False,
    center: np.ndarray = None,
    area_radius: float = None,
    display_area: bool = False,
) -> None:
    """Plot the circles defined by circle_centers and radius

    Args:
        circle_centers (list): list of circle center coordinates
        x_dim (np.ndarray, optional): min and max x-coordinates
        y_dim (np.ndarray, optional): min and max y-coordinates
        radius (float, optional): radius used for the circles. Defaults to 1.
        circular_area (bool, optional): If True, cover circle instead of corners. Defaults to False.
        center (np.ndarray, optional): center of circular area to cover (if circular_area = True). Defaults to None.
        area_radius (float, optional): radius of circular area to cover (if circular_area = True). Defaults to None.
    """
    _, ax = plt.subplots()
    if display_area:
        if circular_area:
            assert center is not None, (
                "'display_area' and 'circular_area' are True, but 'center' is None"
            )
            assert area_radius is not None, (
                "'display_area' and 'circular_area' are True, but 'area_radius' is None"
            )
            area_circle = plt.Circle(center, area_radius, fill=False, color="red")
            ax.add_patch(area_circle)
            plt.legend(["Area"])
        else:
            assert x_dim is not None, "'display_area' is True but 'x_dim' is None"
            assert y_dim is not None, "'display_area' is True but 'y_dim' is None"
            x_corners = [x for x in x_dim for _ in y_dim]
            y_corners = [y for _ in x_dim for y in y_dim]
            plt.scatter(x_corners, y_corners, color="red")
            plt.legend(["Area corner points"])

    for c in circle_centers:
        circle = plt.Circle(c, radius, fill=False)
        ax.add_patch(circle)

    # Automatic scaling does not work very well for circles for one reason or the other, so we find and set the limits manually
    x_coords = [c[0] for c in circle_centers]
    y_coords = [c[1] for c in circle_centers]
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    margin = 2 * radius
    ax.set_xlim(x_min - 2 * margin, x_max + 2 * margin)
    ax.set_ylim(y_min - 2 * margin, y_max + 2 * margin)
    ax.set_aspect("equal", "box")

    plt.show()


def min_enclosing_circle(pc: laspy.LasData) -> Tuple[float, float, float]:
    """Find the smallest enclosing circle for a point cloud

    Args:
        pc (laspy.LasData): point cloud of a circular forest plot

    Returns:
        Tuple[float, float, float]: center and radius of the approximate smallest enclosing circle
    """
    hull = ConvexHull(
        pc.xyz[:, :2]
    )  # Reduce number of points by fitting a convex hull to data
    vertices = pc.xyz[:, :2][hull.vertices]
    x, y, radius = smallestenclosingcircle.make_circle(vertices)
    return x, y, radius


def cut_cylinders(
    pc: laspy.LasData,
    circle_centers: list,
    overlap_dict: dict,
    radius: float,
    path: str,
    filename_template: str,
    compute_normals: bool,
    compute_geometric: bool,
    geof_radius: float = None,
    feature_names: List[str] = [],
    smooth_features: bool = False,
    n_neighbors: List[int] = [],
) -> None:
    """Cut cylinders from the given point clouds based on the list of circle centers and radius. Save resulting
    cylindrical point clouds in given path. NOTE: x- and y-coordinates within cylinders are normalized in the process, by
    subtracting the corresponding mean coordinate from both

    Args:
        pc (laspy.LasData): point cloud of a circular forest plot
        circle_centers (list): list of circle centers
        overlap_dict (dict): dictionary where key i contains a list of indices of all circle centers that overlap with the center at index i of circle_centers
        radius (float): radius of circles
        path (str): path the cylinders are saved in
        filename_template (str): template for cylinder point cloud filenames
        compute_normals (bool): if True, compute normal vectors and add them to the point cloud as extra dimensions
        compute_geometric (bool): if True, compute geometric features and add them to the point cloud as extra dimensions
        geof_radius (float): the radius used for computing the geometric features. Defaults to None.
        feature_names (List[str]): names of geometric features to compute. Defaults to [].
    """
    cylinder_counter = 1
    pc.update_header()
    if compute_geometric:
        # Compute geometric features on the full point cloud to avoid loss of information
        if smooth_features:
            geof = np.zeros(
                (pc.header.point_count, len(feature_names)), dtype=np.float32
            )
            for k in n_neighbors:
                geof_k = compute_features(
                    pc.xyz,
                    search_radius=geof_radius,
                    feature_names=feature_names,
                    max_k_neighbors=k,
                )
                geof = geof + geof_k
            geof = geof / len(n_neighbors)
        else:
            geof = compute_features(
                pc.xyz, search_radius=geof_radius, feature_names=feature_names
            )
        # Default to middle value in range [0, 1] when geometric feature cannot be computed
        geof[np.isnan(geof)] = 0.5
        extra_dims = [
            laspy.ExtraBytesParams(name=name, type=np.float32) for name in feature_names
        ]
        pc.add_extra_dims(extra_dims)
        for dim, name in enumerate(feature_names):
            pc[name] = geof[:, dim].astype(np.float32)
    if compute_normals:
        dim_names = ["x_normals", "y_normals", "z_normals"]
        extra_dims = [
            laspy.ExtraBytesParams(name=name, type=np.float64) for name in dim_names
        ]
        pc.add_extra_dims(extra_dims)
    # Add extra dimensions for labeling overlap points
    pc.add_extra_dim(laspy.ExtraBytesParams(name="overlap_point", type=np.uint8))
    pc.overlap_point = np.zeros_like(pc.classification, dtype=np.uint8)
    pc.add_extra_dim(laspy.ExtraBytesParams(name="overlap_point_id", type=np.uint64))
    pc.overlap_point = np.zeros_like(pc.classification, dtype=np.uint64)
    # Dict for saving overlaping points for each overlap (order of points in cylinders is not necessarily the
    # same, so the easiest way of giving the same id to the same point within an overlaping region of two cylinders
    # is to save the overlapping points the first time when they're given ids and then concat those points to the
    # other cylinder that contains the same overlapping points)
    overlap_points_dict = {}
    current_max_id = 0

    num_centers = len(circle_centers)
    for i in range(num_centers):
        center = circle_centers[i]
        dist = np.linalg.norm(pc.xyz[:, :2] - center, 2, axis=1)
        cylinder_mask = dist <= radius
        cylinder_pc = pc[cylinder_mask]
        if cylinder_pc.header.point_count <= 0:
            # Some cylinders (especially near the edges of the plot) may be empty
            return
        if compute_normals:
            pc_o3d = o3d.geometry.PointCloud()
            pc_o3d.points = o3d.utility.Vector3dVector(cylinder_pc.xyz)
            pc_o3d.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3, max_nn=30)
            )
            normals = np.asarray(pc_o3d.normals)
            dim_names = ["x_normals", "y_normals", "z_normals"]
            for dim, name in enumerate(dim_names):
                cylinder_pc[name] = normals[:, dim]

        # Find overlap points and label them
        overlap_center_indices = overlap_dict[i]
        for j in overlap_center_indices:
            overlap_center = circle_centers[j]
            dist = np.linalg.norm(cylinder_pc.xyz[:, :2] - overlap_center, 2, axis=1)
            overlap_mask = dist <= radius
            cylinder_pc.overlap_point[overlap_mask] = 1
            # Try to fetch overlapping points
            overlap_key = [i, j]
            overlap_key.sort()
            overlap_key = tuple(overlap_key)
            overlap_points = overlap_points_dict.get(overlap_key)
            if overlap_points is None:
                # No overlapping points saved yet, create them
                overlap_points = cylinder_pc[overlap_mask]
                n_overlap_points = overlap_points.header.point_count
                new_max_id = current_max_id + n_overlap_points
                # arange interval end non-inclusive so add 1
                overlap_point_id = np.arange(current_max_id + 1, new_max_id + 1, 1)
                current_max_id = new_max_id
                # Add overlap point ids to cylinder and save the overlap points in dict
                cylinder_pc.overlap_point_id[overlap_mask] = overlap_point_id
                overlap_points = cylinder_pc[overlap_mask]
                overlap_points_dict[overlap_key] = overlap_points
            else:
                # Overlapping points saved, remove from current cylinder and concat the fetched overlap points
                cylinder_pc = cylinder_pc[~overlap_mask]
                cylinder_pc = concat_point_clouds(cylinder_pc, overlap_points)

        filename = filename_template.format(cylinder_counter)
        # laspy breaks the files for some reason if we dont set this flag manually (e.g. can't open in CloudCompare)
        cylinder_pc.header.global_encoding.wkt = True
        cylinder_pc.write(os.path.join(path, filename))
        cylinder_counter += 1
