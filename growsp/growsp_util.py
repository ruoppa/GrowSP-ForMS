import os
import logging
import torch

import numpy as np
import torch.nn as nn
import MinkowskiEngine as ME
import torch.nn.functional as F

from tqdm import tqdm
from easydict import EasyDict
from typing import Tuple, Union
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from util import data_type
from util.logger import print_log


def get_sp_features(
    config: EasyDict,
    data_loader: DataLoader,
    model: nn.Module,
    current_num_sp: Union[int, None],
    epoch: int,
    logger: logging.Logger = None,
) -> Tuple[list, list, list]:
    """Exratct superpoint features from each point cloud

    Args:
        config (EasyDict): config as a dict
        data_loader (DataLoader): DataLoader object for the current dataset
        model (nn.Module): neural network backbone
        current_num_sp (int): current number of superpoints

    Raises:
        NotImplementedError: if attempting to extract features for backbone type for which it has not been implemented yet

    Returns:
        Tuple[list, list, list]: three lists that contain the following information for each point cloud in the dataloader:
            1. superpoint neural features
            2. new superpoint labels
            3. context (point cloud filename and initial superpoint labels)
    """
    growsp_config = config.growsp
    print_log("Computing superpoint features...", logger)
    # List to store superpoint features
    sp_features_list = []
    # List to store new superpoint labels
    all_sp_labels = []
    # Set model to evaluation mode
    model.eval()
    # Required for the get_pseudo_labels() function, contains name of data batch, ground truth, original superpoints and bool indicating whether the data is labeled (i.e. ground truth is correct)
    context = []
    n_rgb_channels = data_loader.dataset.n_rgb_channels
    last_rgb_ind = 3 + n_rgb_channels

    with torch.no_grad():
        for data in tqdm(data_loader):  # Iterate over all batches
            coords, features, normals, labels, _, _, initial_sp, index = data

            initial_sp = initial_sp.squeeze()
            # fetch name of the current data batch (file name)
            pc_filename = data_loader.dataset.filenames[index[0]]
            # fetch bool that indicates whether labels are correct
            has_labels = data_loader.dataset.is_labeled[index[0]]
            # Copy of the initial superpoint labels
            original_sp = initial_sp.clone()

            if config.backbone.type is data_type.sparse:
                # For sparse data, create a tensorfield before passing to model
                in_field = ME.TensorField(
                    features[:, 0 : config.backbone.kwargs.in_channels],
                    coords,
                    device=0,
                )
                # calculate pointwise features
                neural_features = model(in_field)
            else:
                raise NotImplementedError(
                    f"Superpoint features not implemented for backbone of type '{config.backbone.type}'"
                )

            # Compute average rgb/xyz/norm for each superpoint for superpoint merging
            valid_mask = initial_sp != -1
            # Filter out points that are part of invalid initial superpoints
            features = features[valid_mask].cuda()
            normals = normals[valid_mask].cuda()
            neural_features = neural_features[valid_mask]
            initial_sp = initial_sp[valid_mask].long()
            # Extract raw features
            pc_xyz = features[:, 0:3] * config.backbone.voxel_size
            pc_rgb = features[:, 3:last_rgb_ind]

            # Extract superpoint features
            if (
                current_num_sp is not None
            ):  # superpoints are growing -> merge some superpoints
                unique_sp = torch.unique(initial_sp)
                # Mean superpoint features for each superpoint
                mean_sp_neural = []
                mean_sp_rest = []
                for sp in unique_sp:
                    # Compute features for each superpoint by taking the mean of features for all points within the superpoint.
                    # All features considered when superpoints are growing. And features other than neural features are given weights
                    indices = torch.nonzero(initial_sp == sp).squeeze(dim=1)
                    mean_neural_feature = torch.mean(neural_features[indices], dim=0)
                    mean_xyz = torch.mean(pc_xyz[indices], dim=0)
                    mean_rgb = torch.mean(pc_rgb[indices], dim=0)
                    mean_normals = torch.mean(normals[indices], dim=0)
                    # Concat the all features and append to list
                    mean_sp_neural.append(mean_neural_feature)
                    mean_sp_rest.append(
                        torch.cat((mean_rgb, mean_xyz, mean_normals), dim=-1)
                    )
                mean_sp_neural = torch.stack(mean_sp_neural)
                min_neural, _ = mean_sp_neural.min(dim=-2)
                mean_sp_neural = (mean_sp_neural - min_neural) / (
                    mean_sp_neural.max(dim=-2)[0] - min_neural
                )
                mean_sp_neural[torch.isnan(mean_sp_neural)] = 0
                mean_sp_rest = torch.stack(mean_sp_rest)
                min_rest, _ = mean_sp_rest.min(dim=-2)
                mean_sp_rest = (mean_sp_rest - min_rest) / (
                    mean_sp_rest.max(dim=-2)[0] - min_rest
                )
                mean_sp_rest[torch.isnan(mean_sp_rest)] = 0
                mean_sp_features = torch.cat((mean_sp_neural, mean_sp_rest), dim=-1)
                # Determine the number of clusters. If number of superpoint features is less than desired number of superpoints
                # (i.e. current_num_sp), the number of clusters is set to the number of superpoint features
                if growsp_config.use_percentage:
                    percent_num_sp = int(len(unique_sp) * (current_num_sp / 100))
                    if mean_sp_features.size(0) < percent_num_sp:
                        n_clusters = mean_sp_features.size(0)
                    else:
                        n_clusters = percent_num_sp
                else:
                    if mean_sp_features.size(0) < current_num_sp:
                        n_clusters = mean_sp_features.size(0)
                    else:
                        n_clusters = current_num_sp

                sp_idx = torch.from_numpy(
                    KMeans(n_clusters=n_clusters, n_init=5, random_state=0).fit_predict(
                        mean_sp_features.cpu().numpy()
                    )
                ).long()

                new_sp = sp_idx[
                    initial_sp
                ]  # New superpoints, where some superpoints have been merged
            else:  # Superpoints have not yet begun growing
                new_sp = initial_sp

            # Get mean superpoint features. If superpoints are growing, these are the features for the new superpoints,
            # otherwise they're the features for the initial superpoints (since new_sp = initial_sp)
            # Mean superpoint features for each (new) superpoint
            mean_sp_features = []
            # Mean rgb features for each superpoint
            mean_rgb_features = []
            new_unique_sp = torch.unique(new_sp)
            for sp in new_unique_sp:
                indices = torch.nonzero(new_sp == sp).squeeze(dim=1)
                mean_sp_features.append(torch.mean(neural_features[indices], dim=0))
                mean_rgb_features.append(torch.mean(pc_rgb[indices], dim=0))
            mean_sp_features = torch.stack(mean_sp_features, dim=0)
            mean_rgb_features = torch.stack(mean_rgb_features, dim=0)
            # list for superpoint point feature histograms
            pfh = []
            for sp in torch.unique(new_sp):
                if sp != -1:
                    sp_mask = sp == new_sp
                    # Compute pfh and append to list
                    pfh.append(
                        _compute_hist(normals[sp_mask].cpu()).unsqueeze(0).cuda()
                    )
            # transform into matrix [#sp, 10]
            pfh = torch.cat(pfh, dim=0)
            mean_sp_features = F.normalize(mean_sp_features, dim=-2)
            # concat neural features with rgb and pfh for primitive clustering, set weights for rgb and pfh
            # NOTE: If using constant weights, set w_coef = 0.6
            w_coef = np.max([1 - (epoch / 100), 0.2])
            w_rgb = growsp_config.w_rgb_cluster * w_coef
            w_pfh = growsp_config.w_pfh_cluster * w_coef
            mean_sp_features = torch.cat(
                (mean_sp_features, w_rgb * mean_rgb_features, w_pfh * pfh), dim=-1
            )
            # Save results
            sp_features_list.append(mean_sp_features.cpu())
            all_sp_labels.append(new_sp.cpu())
            context.append((pc_filename, original_sp, labels, has_labels))

            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))

    return sp_features_list, all_sp_labels, context


def get_sp_features_geometric(
    config: EasyDict,
    data_loader: DataLoader,
    model: nn.Module,
    current_num_sp: Union[int, None],
    epoch: int,
    logger: logging.Logger = None,
) -> Tuple[list, list, list]:
    # Uses geometric features instead of pfh for primitive clustering
    growsp_config = config.growsp
    print_log("Computing superpoint features...", logger)
    sp_features_list = []  # List to store superpoint features
    all_sp_labels = []  # List to store new superpoint labels
    model.eval()  # Set model to evaluation mode
    context = []  # Required for the get_pseudo_labels() function, contains name of data batch, ground truth, original superpoints and bool indicating whether the data is labeled (i.e. ground truth is correct)
    n_rgb_channels = data_loader.dataset.n_rgb_channels
    last_rgb_ind = 3 + n_rgb_channels

    with torch.no_grad():
        for data in tqdm(data_loader):  # Iterate over all batches
            coords, features, normals, labels, _, _, initial_sp, index = data
            initial_sp = initial_sp.squeeze()
            # fetch name of the current data batch (file name)
            pc_filename = data_loader.dataset.filenames[index[0]]
            # fetch bool that indicates whether labels are correct
            has_labels = data_loader.dataset.is_labeled[index[0]]
            original_sp = initial_sp.clone()  # Copy of the initial superpoint labels

            if config.backbone.type is data_type.sparse:
                # For sparse data, create a tensorfield before passing to model
                in_field = ME.TensorField(
                    features[:, 0 : config.backbone.kwargs.in_channels],
                    coords,
                    device=0,
                )
                # Calculate pointwise features
                neural_features = model(in_field)
            else:
                raise NotImplementedError(
                    f"Superpoint features not implemented for backbone of type '{config.backbone.type}'"
                )

            # Compute average rgb/xyz/norm for each superpoint for superpoint merging
            valid_mask = initial_sp != -1
            # Filter out points that are part of invalid initial superpoints
            features = features[valid_mask].cuda()
            normals = normals[valid_mask].cuda()
            neural_features = neural_features[valid_mask]
            initial_sp = initial_sp[valid_mask].long()
            # Extract raw features
            pc_xyz = features[:, 0:3] * config.backbone.voxel_size
            pc_rgb = features[:, 3:last_rgb_ind]
            # other features
            pc_other = features[:, last_rgb_ind:]

            # Extract superpoint features
            if (
                current_num_sp is not None
            ):  # superpoints are growing -> merge some superpoints
                unique_sp = torch.unique(initial_sp)
                # Mean superpoint features for each superpoint
                mean_sp_neural = []
                mean_sp_rest = []
                for sp in unique_sp:
                    # Compute features for each superpoint by taking the mean of features for all points within the superpoint.
                    # All features considered when superpoints are growing. And features other than neural features are given weights
                    indices = torch.nonzero(initial_sp == sp).squeeze(dim=1)
                    mean_neural_feature = torch.mean(neural_features[indices], dim=0)
                    mean_xyz = torch.mean(pc_xyz[indices], dim=0)
                    mean_rgb = torch.mean(pc_rgb[indices], dim=0)
                    mean_other = torch.mean(pc_other[indices], dim=0)
                    # Concat the all features and append to list
                    mean_sp_neural.append(mean_neural_feature)
                    mean_sp_rest.append(
                        torch.cat((mean_rgb, mean_xyz, mean_other), dim=-1)
                    )
                mean_sp_neural = torch.stack(mean_sp_neural)
                min_neural, _ = mean_sp_neural.min(dim=-2)
                eps = 1e-8
                mean_sp_neural = (mean_sp_neural - min_neural) / (
                    mean_sp_neural.max(dim=-2)[0] - min_neural + eps
                )
                mean_sp_rest = torch.stack(mean_sp_rest)
                min_rest, _ = mean_sp_rest.min(dim=-2)
                mean_sp_rest = (mean_sp_rest - min_rest) / (
                    mean_sp_rest.max(dim=-2)[0] - min_rest + eps
                )
                mean_sp_features = torch.cat((mean_sp_neural, mean_sp_rest), dim=-1)
                # Determine the number of clusters. If number of superpoint features is less than desired number of superpoints
                # (i.e. current_num_sp), the number of clusters is set to the number of superpoint features
                if growsp_config.use_percentage:
                    percent_num_sp = int(len(unique_sp) * (current_num_sp / 100))
                    if mean_sp_features.size(0) < percent_num_sp:
                        n_clusters = mean_sp_features.size(0)
                    else:
                        n_clusters = percent_num_sp
                else:
                    if mean_sp_features.size(0) < current_num_sp:
                        n_clusters = mean_sp_features.size(0)
                    else:
                        n_clusters = current_num_sp

                sp_idx = torch.from_numpy(
                    KMeans(n_clusters=n_clusters, n_init=5, random_state=0).fit_predict(
                        mean_sp_features.cpu().numpy()
                    )
                ).long()
                # New superpoints, where some superpoints have been merged
                new_sp = sp_idx[initial_sp]
            else:  # Superpoints have not yet began growing
                new_sp = initial_sp

            # Get mean superpoint features. If superpoints are growing, these are the features for the new superpoints,
            # otherwise they're the features for the initial superpoints (since new_sp = initial_sp)
            # Mean superpoint features for each (new) superpoint
            mean_sp_features = []
            # Mean rgb features for each superpoint
            mean_rgb_features = []
            new_unique_sp = torch.unique(new_sp)
            for sp in new_unique_sp:
                indices = torch.nonzero(new_sp == sp).squeeze(dim=1)
                mean_sp_features.append(torch.mean(neural_features[indices], dim=0))
                mean_rgb_features.append(torch.mean(pc_rgb[indices], dim=0))
            mean_sp_features = torch.stack(mean_sp_features, dim=0)
            mean_rgb_features = torch.stack(mean_rgb_features, dim=0)
            # list for superpoint geometric features
            geof = []
            for sp in torch.unique(new_sp):
                if sp != -1:
                    indices = torch.nonzero(new_sp == sp).squeeze(dim=1)
                    geof.append(torch.mean(pc_other[indices], dim=0))
            # transform into matrix [#sp, 10]
            geof = torch.stack(geof, dim=0)
            mean_sp_features = F.normalize(mean_sp_features, dim=-2)
            # concat neural features with rgb and pfh for primitive clustering, set weights for rgb and geof
            # NOTE: If using constant weights, set w_coef = 0.6
            w_coef = np.max([1 - (epoch / 100), 0.2])
            w_rgb = growsp_config.w_rgb_cluster * w_coef
            w_geof = growsp_config.w_geof_cluster * w_coef
            mean_sp_features = torch.cat(
                (mean_sp_features, w_rgb * mean_rgb_features, w_geof * geof), dim=-1
            )
            # Save results
            sp_features_list.append(mean_sp_features.cpu())
            all_sp_labels.append(new_sp.cpu())
            context.append((pc_filename, original_sp, labels, has_labels))

            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))

    return sp_features_list, all_sp_labels, context


def get_pseudo_labels(
    pseudo_path: str,
    context: list,
    primitive_labels: np.ndarray,
    all_sp_labels: list,
    logger: logging.Logger = None,
):
    """Get pseudo labels for each point cloud from an array of primitive labels. This is required, since not all
    points are considered when clustering to create primitive labels -> primitive labels can not be casted to
    the original data as is

    Args:
        pseudo_path (str): path to directory for saving the pseudo labels
        context (list): output of get_sp_features(). Contains the filename of each point cloud and the initial superpoint labels
        primitive_labels (np.ndarray): primitive labels from clustering the superpoint features
        all_sp_labels (list, optional): list of all superpoint labels for current superpoints. Defaults to None.

    Returns:
        tuple: pseudo labels for all points, ground truth labels and pseudo labels mapped to most common ground truth label within the
            pseudo label area. The latter two are only returned for labeled point clouds. Lastly, a mask that points the pseudo labels
            with corresponding ground truth available from the first return value is also returned.
    """
    print_log("Computing pseudo labels...", logger)
    if not os.path.exists(pseudo_path):  # if there is not pseudo-label folder, make it
        os.makedirs(pseudo_path)
    all_pseudo_labels = []
    # Not all actually, only the labels of point clouds that are considered labeled by the dataloader
    all_gt_labels = []
    # Same here as above
    all_pseudo_gt_labels = []
    labeled_mask = []
    cum_sum_sp = 0

    # Iterate over all superpoint labels and contexts by batch
    for sp_labels, sp_context in tqdm(
        zip(all_sp_labels, context), total=len(all_sp_labels)
    ):
        pc_filename, initial_sp, labels, has_labels = sp_context
        # Get superpoints from one data batch and add cum_sum_sp to track where current batch superpoints start in primitive_labels
        primitive_labels_batch = sp_labels + cum_sum_sp
        valid_sp_mask = sp_labels != -1
        # Add number of valid superpoints in current batch to cum_sum_sp
        cum_sum_sp += len(np.unique(sp_labels[valid_sp_mask]))

        valid_initial_sp_mask = initial_sp != -1

        # Assign pseudo labels
        pseudo_labels = -np.ones_like(
            labels.numpy(), dtype=np.int32
        )  # Initialize array to save pseudo labels
        pseudo_labels[valid_initial_sp_mask] = primitive_labels[primitive_labels_batch]
        # Save pseudo labels
        pseudo_label_file = os.path.join(pseudo_path, pc_filename + ".npy")
        np.save(pseudo_label_file, pseudo_labels)

        all_pseudo_labels.append(pseudo_labels)
        labeled_mask.append(np.full(len(pseudo_labels), has_labels))

        # If the data has labels, compute pseudo gt labels and append to list
        if has_labels:
            labels_tmp = labels[valid_initial_sp_mask]
            pseudo_gt_labels = -torch.ones_like(labels, dtype=torch.int32)
            pseudo_gt_labels_tmp = pseudo_gt_labels[valid_initial_sp_mask]
            # For each primitive, determine the corresponding ground truth label by simply finding the most common
            # ground truth label within the area covered by said primitive
            for p in np.unique(primitive_labels_batch):
                if p != -1:
                    mask = p == primitive_labels_batch
                    primitive_labels_batch_gt = torch.mode(labels_tmp[mask]).values
                    pseudo_gt_labels_tmp[mask] = primitive_labels_batch_gt
            pseudo_gt_labels[valid_initial_sp_mask] = pseudo_gt_labels_tmp
            # Append results to list
            all_gt_labels.append(labels)
            all_pseudo_gt_labels.append(pseudo_gt_labels)

    all_pseudo_labels = np.concatenate(all_pseudo_labels)
    all_gt_labels = np.concatenate(all_gt_labels)
    all_pseudo_gt_labels = np.concatenate(all_pseudo_gt_labels)
    labeled_mask = np.concatenate(labeled_mask)

    return all_pseudo_labels, all_gt_labels, all_pseudo_gt_labels, labeled_mask


def get_fixclassifier(
    in_channels: int, n_centroids: int, centroids: torch.tensor
) -> nn.Linear:
    classifier = nn.Linear(
        in_features=in_channels, out_features=n_centroids, bias=False
    )
    centroids = F.normalize(centroids, dim=1)
    classifier.weight.data = centroids
    for param in classifier.parameters():
        param.requires_grad = False
    return classifier


def _compute_hist(
    normal: torch.tensor, bins: int = 10, min: int = -1, max: int = 1
) -> torch.tensor:
    """Compute point feature histograms based on point cloud normals

    Args:
        normal (torch.tensor): tensor of point cloud normals [N x 3]
        bins (int, optional): number of features to compute. Defaults to 10.
        min (int, optional): minimum accepted normal value. Defaults to -1.
        max (int, optional): maximum accepted normal value. Defaults to 1.

    Returns:
        torch.tensor: point histogram features
    """
    normal = F.normalize(normal)
    relation = torch.mm(normal, normal.t())
    # Margin for error between min and max (the min and max values seem to cause issues when there's only one point in the
    # superpoint, which may result in 'relation' being very close to -1 or 1)
    err_margin = 1e-5
    # Upper triangular matrix
    relation = torch.triu(relation, diagonal=0)
    hist = torch.histc(relation, bins, min - err_margin, max + err_margin)
    hist /= hist.sum()
    del relation

    return hist
