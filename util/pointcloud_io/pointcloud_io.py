import laspy

import numpy as np
import pandas as pd

from easydict import EasyDict
from typing import Union
from util.pointcloud_io import ply_io


def get_read_fn(filename_extension: str):
    if filename_extension == ".las" or filename_extension == ".laz":
        return laspy.read
    elif filename_extension == ".ply":
        return _read_ply
    else:
        raise ValueError(
            f"File format '{filename_extension}' is not supported. Supported formats are ['.las', '.laz', '.ply']"
        )


def _read_ply(path: str, class_field_name: str = "class") -> "PlyData":
    """Wrapper function for read_ply() that transforms the numpy struct array into a PlyData object

    Args:
        path (str): path to .ply pointcloud
        class_field_name (str, optional): name of the ground truth data field in the point cloud. Defaults to "class".

    Returns:
        PlyData: read point cloud as PlyData object
    """
    # Cast the ground truth field to int8
    pc = ply_io.read_ply(path)
    new_type = pc.dtype.descr
    for i, x in enumerate(new_type):
        if x[0] == class_field_name:
            new_type[i] = (class_field_name, "i1")
            break
    pc = pc.astype(np.dtype(new_type))
    return PlyData(pc, class_field_name)


"""This class is a sort of wrapper for the numpy array returned by read_ply, that supports all of the laspy operations
performed in the code. For quite a few of the functions the class simply does nothing, since in the case of .ply files we
have no need for doing anything with operations such as update_header(). The single advantage this class brings is that
it eliminates the need for multiple if/else clauses that would otherwise be required whenever we call a function or access
a property that is specific to laspy
"""


class PlyData:
    def __init__(
        self,
        data: np.ndarray,
        class_field_name: str = "class",
        extra_dimensions: pd.DataFrame = None,
    ) -> None:
        self.data = data
        self._num_datapoints = data.size
        self.class_field_name = class_field_name
        self.header = EasyDict(
            {
                "global_encoding": {"wkt": True},
                "point_format": {
                    "standard_dimensions": [
                        {"name": dim} for dim in self.data.dtype.names
                    ]
                },
            }
        )
        """Merging numpy struct arrays is incredibly slow (i.e. adding any extra dimensions to 'data' is very slow). As such, we
        store any extra dimensions in a pandas dataframe. While it does consume more memory, it is also much more efficient, and
        thankfully extra dimensions only need to be added during preprocessing
        """
        if extra_dimensions is None:
            self._extra_dimensions = pd.DataFrame(index=range(self._num_datapoints))
        else:
            assert len(extra_dimensions.index) == self._num_datapoints, (
                "Number of datapoints in 'extra_dimensions' must match the number of datapoints in 'data'"
                f"[{len(extra_dimensions.index)} != {self._num_datapoints}]"
            )
            self._extra_dimensions = extra_dimensions.reset_index(drop=True)
        self.X, self.Y, self.Z = self.x, self.y, self.z

    @property
    def x(self):
        return self.data["x"]

    @x.setter
    def x(self, value):
        self.data["x"] = value

    @property
    def y(self):
        return self.data["y"]

    @y.setter
    def y(self, value):
        self.data["y"] = value

    @property
    def z(self):
        return self.data["z"]

    @z.setter
    def z(self, value):
        self.data["z"] = value

    @property
    def red(self):
        return self.data["red"]

    @red.setter
    def red(self, value):
        self.data["red"] = value

    @property
    def green(self):
        return self.data["green"]

    @green.setter
    def green(self, value):
        self.green["green"] = value

    @property
    def blue(self):
        return self.data["blue"]

    @blue.setter
    def blue(self, value):
        self.data["blue"] = value

    @property
    def classification(self):
        return self.data[self.class_field_name]

    @classification.setter
    def classification(self, value):
        self.data[self.class_field_name] = value

    @property
    def xyz(self):
        return np.vstack([self.x, self.y, self.z]).T

    def __getitem__(self, index: Union[int, np.ndarray, str]):
        """Return a portion of the PlyData object indicated by indices. The returned object is a copy, i.e. any operations
        done to it do not affect the original PlyData object. Alternatively, if the index is a string, return the corresponding
        field in the data

        Args:
            index (Union[int, np.ndarray, str]): indices of PlyData object to return or the name of the field which to return

        Returns:
            Any: new PlyData or field
        """
        if isinstance(index, int):
            index = [index]
        if isinstance(index, str):
            if index in self.data.dtype.names:
                return self.data[index]
            else:
                return self._extra_dimensions[index].to_numpy()
        else:
            new_data = self.data[index].copy()
            new_extra_dim = self._extra_dimensions.iloc[index].copy()
            return PlyData(new_data, self.class_field_name, new_extra_dim)

    def __setitem__(self, index: Union[int, np.ndarray, str], value) -> None:
        if isinstance(index, str):
            if index in self.data.dtype.names:
                self.data[index] = value
            elif index in self._extra_dimensions.columns:
                self._extra_dimensions[index] = value.astype(
                    self._extra_dimensions[index].dtype
                )
            else:
                raise KeyError(f"No field of name '{index}'!")
        else:
            self.data[index] = value

    def __len__(self):
        return self._num_datapoints

    def __getattr__(self, __name: str):
        # If we try to access an attribute that is not a property of the class, try finding the attribute from the extra fields instead
        try:
            return self.data[__name]
        except ValueError:
            try:
                return self._extra_dimensions[__name]
            except KeyError:
                raise AttributeError(f"'PlyData' object has no attribute '{__name}'")

    def write(self, path: str):
        fields = [self.data[field_name] for field_name in self.data.dtype.names]
        names = list(self.data.dtype.names)
        for field_name in self._extra_dimensions.columns:
            fields.append(self._extra_dimensions[field_name].to_numpy())
            names.append(field_name)
        ply_io.write_ply(path, fields, names)

    def add_extra_dim(self, params: laspy.ExtraBytesParams) -> None:
        self._extra_dimensions[params.name] = np.zeros(
            self._num_datapoints, dtype=params.type
        )

    def update_header(self):
        # Nothing to update (since the object does not contain a sepparate header)
        pass
