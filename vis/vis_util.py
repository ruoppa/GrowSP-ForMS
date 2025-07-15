import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# rgb colors for each class in the EvoMS dataset
_EVOMS_COLORMAP = {
    0: mcolors.to_rgb(mcolors.CSS4_COLORS["limegreen"]),  # foliage
    1: mcolors.to_rgb(mcolors.CSS4_COLORS["orangered"]),  # wood
    2: mcolors.to_rgb(mcolors.CSS4_COLORS["magenta"]),  # wood predicted as foliage
    3: mcolors.to_rgb(mcolors.CSS4_COLORS["aqua"]),  # foliage predicted as wood
}


def get_sp_colormap(cnum: int = 2000) -> np.array:
    """Generate a colormap for visualizing superpoints. The resulting colormap will contain all colors from the
    premade matplotlib colormaps Set1, Set2 and Set3 cnum times in a row

    Args:
        cnum (int, optional): Length of the gengerated colormap is cnum * 29 + 1. Defaults to 2000.

    Returns:
        np.array: colormap as a numpy array. Each row of the array is a color.
    """
    colormap = []
    for _ in range(cnum):
        for k in range(12):
            colormap.append(plt.cm.Set3(k))
        for k in range(9):
            colormap.append(plt.cm.Set1(k))
        for k in range(8):
            colormap.append(plt.cm.Set2(k))
    colormap.append((0, 0, 0, 0))
    colormap = np.array(colormap)
    return colormap


def get_evoms_rgb_array(labels: np.ndarray) -> np.ndarray:
    """Return an rgb array based on (predicted) class labels for the EvoMS dataset

    Args:
        labels (np.ndarray): array of labels (value between 0 and 1)

    Returns:
        np.ndarray: [n x 3] array of rgb values, where all points within the same class have the same rgb value
    """
    rgb = [list(_EVOMS_COLORMAP[point]) for point in labels]
    return np.array(rgb) * 255
