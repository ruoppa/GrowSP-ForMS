import torch
import os
import warnings

import numpy as np
import MinkowskiEngine as ME

from easydict import EasyDict
from abc import abstractmethod, ABC
from typing import List, Tuple, Union
from torch.utils.data import Dataset
from util.data_augmentation import shift_coords, rotate_coords, scale_coords
from util import get_read_fn, Mode, data_type


class cfl_collate_fn:
    def __call__(self, list_data):
        coords, features, normals, labels, inverse_map, pseudo, region, index = list(
            zip(*list_data)
        )
        (
            coords_batch,
            features_batch,
            normal_batch,
            labels_batch,
            inverse_batch,
            pseudo_batch,
        ) = [], [], [], [], [], []
        region_batch = []
        accm_num = 0
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            coords_batch.append(
                torch.cat(
                    (
                        torch.ones(num_points, 1).int() * batch_id,
                        torch.from_numpy(coords[batch_id]).int(),
                    ),
                    1,
                )
            )
            features_batch.append(torch.from_numpy(features[batch_id]))
            normal_batch.append(torch.from_numpy(normals[batch_id]))
            labels_batch.append(torch.from_numpy(labels[batch_id]))
            inverse_batch.append(torch.from_numpy(inverse_map[batch_id]))
            pseudo_batch.append(torch.from_numpy(pseudo[batch_id]))
            region_batch.append(torch.from_numpy(region[batch_id])[:, None])
            accm_num += coords[batch_id].shape[0]

        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()
        features_batch = torch.cat(features_batch, 0).float()
        normal_batch = torch.cat(normal_batch, 0).float()
        labels_batch = torch.cat(labels_batch, 0)
        inverse_batch = torch.cat(inverse_batch, 0).int()
        pseudo_batch = torch.cat(pseudo_batch, -1)
        region_batch = torch.cat(region_batch, 0)

        return (
            coords_batch,
            features_batch,
            normal_batch,
            labels_batch,
            inverse_batch,
            pseudo_batch,
            region_batch,
            index,
        )


class cfl_collate_fn_val:
    def __call__(self, list_data):
        coords, features, _, labels, inverse_map, _, region, index, overlap_ids = list(
            zip(*list_data)
        )
        coords_batch, features_batch, inverse_batch, labels_batch, overlap_id_batch = (
            [],
            [],
            [],
            [],
            [],
        )
        region_batch = []
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            coords_batch.append(
                torch.cat(
                    (
                        torch.ones(num_points, 1).int() * batch_id,
                        torch.from_numpy(coords[batch_id]).int(),
                    ),
                    1,
                )
            )
            features_batch.append(torch.from_numpy(features[batch_id]))
            inverse_batch.append(torch.from_numpy(inverse_map[batch_id]))
            labels_batch.append(torch.from_numpy(labels[batch_id]).int())
            region_batch.append(torch.from_numpy(region[batch_id])[:, None])
            overlap_id_batch.append(torch.from_numpy(overlap_ids[batch_id]).int())

        # Concatenate all lists
        coords_batch = torch.cat(coords_batch, 0).float()
        features_batch = torch.cat(features_batch, 0).float()
        inverse_batch = torch.cat(inverse_batch, 0).int()
        labels_batch = torch.cat(labels_batch, 0).int()
        region_batch = torch.cat(region_batch, 0)
        overlap_id_batch = torch.cat(overlap_id_batch, 0).int()

        return (
            coords_batch,
            features_batch,
            labels_batch,
            inverse_batch,
            region_batch,
            index,
            overlap_id_batch,
        )


class dataset_base(Dataset, ABC):
    def __init__(self, config: EasyDict):
        """Initialize the dataloader

        Args:
            config (EasyDict): dataset config as an EasyDict
        """
        self.config = config
        # Model config is the neural network backbone config. It's passed as one value in the dataset config to allow easily
        # building datasets from config files
        load_config = config.model_config.dataloader
        self._mode = Mode.train  # Mode set to train by default
        # Name, label map and ignored label from config
        self.name = config.NAME
        self.label_map = config.LABELS
        self.ignore_label = config.IGNORE_LABEL
        # Path variables from dataset config
        self.input_path, self.sp_path, self.pseudo_path = (
            config.INPUT_PATH,
            config.SP_PATH,
            config.PSEUDO_PATH,
        )
        # Data augmentation modes from model config
        self.augment_data = load_config.augment_data
        # Extra feature names and available rgb channels from model config
        self._extra_features = load_config.extra_features
        self.has_extra_features = (
            len(self._extra_features) > 0
        )  # True if the extra features are used, i.e. self._extra_features != []
        self.has_rgb = load_config.has_rgb
        self._rgb_channels = np.array(["red", "green", "blue"])[self.has_rgb]
        self.n_rgb_channels = np.sum(self.has_rgb)
        # superpoint drop threshold from model config
        self.sp_drop_threshold = load_config.sp_drop_threshold
        # Function for reading the input point clouds (different for .las and .ply)
        self._read = get_read_fn(config.FILENAME_EXTENSION)
        # voxel size from backbone config
        self.voxel_size = config.model_config.backbone.voxel_size
        # Clipping large point clouds
        self.clip_large_pc = load_config.clip_large_pc
        self.clip_bound = load_config.clip_bound
        # Get NN type
        self.type = data_type(config.model_config.backbone.type)
        # Data augmentations
        self.shift_coords = shift_coords(shift_ratio=50)
        self.rotate_coords = rotate_coords(
            rotation_bound=(
                (-np.pi / 32, np.pi / 32),
                (-np.pi / 32, np.pi / 32),
                (-np.pi, np.pi),
            )
        )
        self.scale_coords = scale_coords(scale_bound=(0.9, 1.1))

    @abstractmethod
    def _get_filenames(self) -> np.ndarray:
        # Return a list of filenames pointing to input point clouds. Filenames may differ between dataset modes
        pass

    @property
    @abstractmethod
    def filenames(self):
        pass

    @filenames.setter
    @abstractmethod
    def filenames(self, new_names: List[str]):
        pass

    @abstractmethod
    def _init_is_labeled(self) -> np.ndarray:
        """Return a boolean array where element at i is True if the pointcloud at index i of the dataloader is labeled and False otherwise
        Implementing your own dataset class should be very straightforward (all you need to do is implement this method, and _get_filenames,
        everything else can be copied from datasets.py)
        """
        pass

    @property
    @abstractmethod
    def is_labeled(self):
        pass

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, new_mode: Mode):
        """Setter function for the mode property to prevent setting the value to something undefined.

        Args:
            new_mode (Mode): mode to set the dataloader to. Must be an instance of Mode

        Raises:
            ValueError: if 'new_mode' is not one of the permitted values.
        """
        if isinstance(new_mode, Mode):
            self._mode = new_mode
            self.filenames = self._get_filenames()
            self.is_labeled = self._init_is_labeled()
        else:
            raise ValueError(
                f"Mode '{new_mode}' not recognized! 'mode' must be one of {Mode._member_names_}"
            )

    @property
    def extra_features(self):
        return self._extra_features

    @extra_features.setter
    def extra_features(self, new_features: List[str]):
        self._extra_features = new_features
        self.has_extra_features = len(new_features) > 0

    def _augment_coords(self, coords: np.ndarray) -> np.ndarray:
        # NOTE: augmentation is not performed, unless mode is set to 'train'
        if self.mode is Mode.train or self.mode is Mode.cluster:
            coords = self.rotate_coords(coords)
            coords = self.shift_coords(coords)
            coords = self.scale_coords(coords)
        return coords

    def _norm_coords(self, coords: np.ndarray) -> np.ndarray:
        coords_center = np.mean(coords, axis=0)
        coords[:, :2] -= coords_center[:2]
        coords[:, 2] -= np.min(coords[:, 2])
        return coords

    def _norm_features(self, features: np.ndarray) -> np.ndarray:
        # IQR scaling for available rgb values
        rgb = features[:, 0 : self.n_rgb_channels].copy()
        iqr = np.quantile(rgb, 0.75, axis=0) - np.quantile(rgb, 0.25, axis=0)
        rgb = (rgb - np.median(rgb, axis=0)) / iqr
        rgb = rgb - np.min(rgb, axis=0)
        nan_mask = np.isnan(rgb)
        rgb[nan_mask] = 0.0
        features[:, 0 : self.n_rgb_channels] = rgb
        # Replace any nan values with 0.5 (this assumes that the features are geometric, i.e. value between 0 and 1)
        nan_mask = np.isnan(features[:, self.n_rgb_channels :])
        features[:, self.n_rgb_channels :][nan_mask] = 0.5
        return features

    def _voxelize(self, coords: np.ndarray, features: np.ndarray, labels: np.ndarray):
        scale = 1 / self.voxel_size
        coords = np.floor(coords * scale)
        coords, features, labels, unique_map, inverse_map = ME.utils.sparse_quantize(
            np.ascontiguousarray(coords),
            features,
            labels=labels,
            ignore_label=self.config.IGNORE_LABEL,
            return_index=True,
            return_inverse=True,
        )
        return coords, features, labels, unique_map, inverse_map

    def _get_clip_inds(self, coords: np.ndarray, center=None):
        bound_min = np.min(coords, 0).astype(float)
        bound_max = np.max(coords, 0).astype(float)
        bound_size = bound_max - bound_min
        if center is None:
            center = bound_min + bound_size * 0.5
        lim = self.clip_bound
        clip_inds = None

        if bound_size.max() >= self.clip_bound:
            clip_inds = (
                (coords[:, 0] >= (-lim + center[0]))
                & (coords[:, 0] < (lim + center[0]))
                & (coords[:, 1] >= (-lim + center[1]))
                & (coords[:, 1] < (lim + center[1]))
                & (coords[:, 2] >= (-lim + center[2]))
                & (coords[:, 2] < (lim + center[2]))
            )
        return clip_inds

    def _get_pointcloud(
        self,
        index: int,
        extra_feature_names: List[str] = [],
        get_normals: bool = False,
        eval: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        """Read preprocessed point cloud from file

        Args:
            index (int): index to point cloud path
            extra_feature_names (List[str], optional): List of names of extra features to read. Defaults to [].
            get_normals (bool, optional): if True, return precomputed point cloud normals as well. Defaults to False.
            eval (bool, optional): if True, return overlap ids for each point

        Raises:
            ValueError: if one or more of the feature names in 'extra_features' cannot be found from the point cloud.
            ValueError: if one or more precomputed normals cannot be found from the point cloud

        Returns:
            Tuple[np.ndarray, ...]: tuple of np.ndarrays. By default coords, rgb and labels, but may also include extra features and normals.
        """
        pc_filename = os.path.join(
            self.input_path, self.filenames[index] + self.config.FILENAME_EXTENSION
        )
        pc = self._read(pc_filename)
        # Fetch available rgb channels
        rgb = []
        for c in self._rgb_channels:
            try:
                rgb.append(np.array(pc[c]).reshape(-1, 1))
            except ValueError as e:
                raise ValueError(
                    f"'{pc_filename}' does not contain field '{c}'!"
                ) from e
        rgb = np.hstack(rgb).astype(np.float32)
        if len(rgb) == 1:
            rgb = rgb.reshape(-1, 1)  # Reshape if there's only one channel available
        # Fetch coordinates
        coords = pc.xyz
        if self.is_labeled[index]:
            labels = pc.classification.astype(np.int8)
            if self.name == "EvoMS":
                labels[labels == 2] = 0  # Replace understory class with foliage
            labels[labels == self.ignore_label] = -1
        else:
            labels = np.zeros_like(
                pc.classification, dtype=np.int8
            )  # No labels so set label values to 0
        get_extra_features = len(extra_feature_names) > 0
        if get_extra_features:
            extra_features = []
            for feature in extra_feature_names:
                try:
                    extra_features.append(np.array(pc[feature]).reshape(-1, 1))
                except ValueError as e:
                    raise ValueError(
                        f"'{pc_filename}' does not contain field '{feature}'!"
                    ) from e
            extra_features = np.hstack(extra_features)
            if len(extra_features) == 1:
                extra_features = extra_features.reshape(
                    -1, 1
                )  # Reshape to enable concatenating with rgb
        if get_normals:
            try:
                normals = np.vstack((pc.x_normals, pc.y_normals, pc.z_normals)).T
            except ValueError as e:
                raise ValueError(
                    f"'{pc_filename}' is missing one or more normals! (NOTE: field name should be <axis>_normals, e.g. x_normals)"
                ) from e
        if eval:
            try:
                overlap_ids = pc.overlap_point_id
            except ValueError as e:
                raise ValueError(
                    f"'{pc_filename}' does not contain overlap point id field! (NOTE: field name should be overlap_point_id)"
                ) from e

        get_all = get_extra_features and get_normals and eval
        get_extra_and_eval = get_extra_features and eval
        get_extra_and_normals = get_extra_features and get_normals
        get_normals_and_eval = get_normals and eval

        if get_all:
            return coords, rgb, labels, normals, extra_features, overlap_ids
        elif get_extra_and_eval:
            return coords, rgb, labels, extra_features, overlap_ids
        elif get_extra_and_normals:
            return coords, rgb, labels, normals, extra_features
        elif get_extra_features:
            return coords, rgb, labels, extra_features
        elif get_normals_and_eval:
            return coords, rgb, labels, normals, overlap_ids
        elif get_normals:
            return coords, rgb, labels, normals
        elif eval:
            return coords, rgb, labels, overlap_ids
        else:
            return coords, rgb, labels

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int):
        """_summary_

        Args:
            index (int): index of element in data

        Returns:
            Tuple: element at index. The Tuple contains the following values in the listed order:
                1. point cloud coordinates
                2. point cloud features, i.e. coordinates concatenated with features. Features contains rgb values + any extra features defined in config
                3. (precomputed) point cloud normals
                4. point cloud labels
                5. inverse map to map the coordinates back to the original point cloud (the returned point clous has been quantized, which is similar to voxelization)
                6. pseudo_labels of each point
                7. indices of original points in the coordinates (only really necessary if mixup is applied, otherwise all points are original)
                8. initial superpoint labels (which have been filtered such that superpoints considered too small have been labeled invalid)
                9. the index
                NOTE: depending on the mode, only placeholder values may be returned for some (e.g. empty list) since said values are not
                necessary for any computations in that mode.
        """
        # Load data from point cloud
        if self.mode is Mode.train or self.mode is Mode.cluster:
            if self.has_extra_features:
                coords, rgb, labels, normals, extra_features = self._get_pointcloud(
                    index, self.extra_features, get_normals=True
                )
                features = np.hstack([rgb, extra_features])
            else:
                coords, rgb, labels, normals = self._get_pointcloud(
                    index, get_normals=True
                )
                features = rgb
        else:
            # Also fetch overlap ids in evaluation mode
            if self.has_extra_features:
                coords, rgb, labels, normals, extra_features, overlap_ids = (
                    self._get_pointcloud(
                        index, self.extra_features, get_normals=True, eval=True
                    )
                )
                features = np.hstack([rgb, extra_features])
            else:
                coords, rgb, labels, normals, overlap_ids = self._get_pointcloud(
                    index, get_normals=True, eval=True
                )
                features = rgb

        # Clip point cloud, if necessary
        if self.clip_large_pc:
            clip_inds = self._get_clip_inds(coords)
            if clip_inds is not None:
                coords, features, labels, normals = (
                    coords[clip_inds],
                    features[clip_inds],
                    labels[clip_inds],
                    normals[clip_inds],
                )

        # Normalize coordinates and features
        coords = self._norm_coords(coords)
        features = self._norm_features(features)

        # Voxelize data
        if self.mode is Mode.train or self.mode is Mode.cluster:
            coords, features, labels, unique_map, inverse_map = self._voxelize(
                coords, features, labels
            )
        else:
            # Do not voxelize labels in eval mode
            coords, features, _, unique_map, inverse_map = self._voxelize(
                coords, features, labels
            )
        normals = normals[unique_map]
        coords = coords.astype(np.float32)
        features = features.astype(np.float32)

        # NOTE: applying augmentations before voxelization will result in a different number of points every time
        if self.augment_data:
            # Apply augmentations if enabled and mode is 'train' (the function checks the mode automatically)
            coords = self._augment_coords(coords)  # coordinates used in training

        # Load initial superpoints
        sp_filename = os.path.join(
            self.sp_path, self.filenames[index] + "_superpoints.npy"
        )
        try:
            initial_sp = np.load(sp_filename)
        except FileNotFoundError:
            warnings.warn(
                f"no superpoints files found for '{self.filenames[index]}'",
                RuntimeWarning,
            )
            initial_sp = np.zeros_like(inverse_map)
        if self.clip_large_pc and clip_inds is not None:
            initial_sp[clip_inds]
        initial_sp = initial_sp[unique_map]

        # If evaluation mode, return without augmentations etc.
        if self.mode is not Mode.train and self.mode is not Mode.cluster:
            return (
                coords,
                np.concatenate((coords, features), axis=-1),
                [],
                np.ascontiguousarray(labels),
                inverse_map,
                [],
                initial_sp,
                index,
                np.ascontiguousarray(overlap_ids.astype(np.int64)),
            )

        if self.mode is Mode.cluster:
            # Filter out small superpoints
            valid_mask = (initial_sp) != -1
            initial_sp_sizes = np.bincount(
                initial_sp[valid_mask]
            )  # Count superpoint sizes
            small_initial_sp = np.where(initial_sp_sizes < self.sp_drop_threshold)[
                0
            ]  # Superpoints to be marked as invalid (-1)
            # In case all superpoints are small, keep them despite threshold (rare)
            small_mask = np.isin(initial_sp, small_initial_sp)
            if (~small_mask).sum() > 0:
                initial_sp[small_mask] = -1

            # Step 2: Map valid superpoints to a continuous range
            valid_mask = initial_sp != -1
            valid_initial_sp = initial_sp[valid_mask]
            unique_vals = np.unique(valid_initial_sp)
            valid_initial_sp = np.searchsorted(unique_vals, valid_initial_sp)
            initial_sp[valid_mask] = valid_initial_sp

            pseudo_filename = os.path.join(
                self.pseudo_path, self.filenames[index] + ".npy"
            )
            pseudo_labels = -np.ones_like(labels).astype(np.longlong)

            # If some normals are nan, replace them with 0
            invalid_normal_mask = np.isnan(normals)
            normals[invalid_normal_mask] = 0

        else:  # In this case mode should be 'train'
            normals = np.zeros_like(coords)
            pseudo_filename = os.path.join(
                self.pseudo_path, self.filenames[index] + ".npy"
            )
            pseudo_labels = np.array(np.load(pseudo_filename), dtype=np.longlong)

        return (
            coords,
            np.concatenate((coords, features), axis=-1),
            normals,
            labels,
            inverse_map,
            pseudo_labels,
            initial_sp,
            index,
        )
