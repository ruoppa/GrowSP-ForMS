import numpy as np

from scipy.linalg import expm, norm
from typing import Tuple


def _M(axis: np.ndarray, theta: float) -> np.ndarray:
    """Construct a rotation matrix for the given axis and angle

    Args:
        axis (np.ndarray): axis to rotate over. [1 x 3] array where one value should be 1 and others 0. The firs element represents the x-axis etc.
        theta (float): angle of rotation (in radians)

    Returns:
        np.ndarray: rotation matrix
    """
    # Technique for constructing rotation matrices with small rotations. It first constructs a skew-symmetric matrix (A^T=-A) and
    # then forms an orthogonal rotation matrix by taking its exponential
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


class shift_coords:
    """Shift coordinates by adding random value between 0 and 1 multiplied by 'shift_ratio' to each coordinate. Shift values are different
    for each axis, but equal within coordinates of one axis (i.e. same value for all x-coordinates, which is different from the value for
    y-coordinates)
    """

    def __init__(self, shift_ratio: float):
        self.ratio = shift_ratio

    def __call__(self, coords: np.ndarray):
        shift = np.random.uniform(0, 1, 3) * self.ratio
        return coords + shift


class rotate_coords:
    """Rotate the coordinates using a rotation matrix. Maximum rotation is constrained with 'rotation_bound'"""

    def __init__(
        self,
        rotation_bound=(
            (-np.pi / 32, np.pi / 32),
            (-np.pi / 32, np.pi / 32),
            (-np.pi, np.pi),
        ),
    ):
        self.rotation_bound = rotation_bound

    def __call__(self, coords: np.ndarray):
        rotation_matrices = []
        for axis_ind, bound in enumerate(self.rotation_bound):
            theta = 0
            axis = np.zeros(3)
            axis[axis_ind] = 1
            if bound is not None:
                theta = np.random.uniform(*bound)  # Random number between the bounds
            rotation_matrices.append(_M(axis, theta))
        # Use random order
        np.random.shuffle(rotation_matrices)
        rotation_matrix = (
            rotation_matrices[0] @ rotation_matrices[1] @ rotation_matrices[2]
        )
        return coords.dot(rotation_matrix)


class scale_coords:
    """Scale the coordinates by multiplying with a random value within 'scale_bound'"""

    def __init__(self, scale_bound: Tuple[float, float] = (0.8, 1.25)):
        self.scale_bound = scale_bound

    def __call__(self, coords: np.ndarray):
        scale = np.random.uniform(*self.scale_bound)
        return coords * scale
