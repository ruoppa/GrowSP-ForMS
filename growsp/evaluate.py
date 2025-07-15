import os
import logging
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from easydict import EasyDict
from typing import Tuple, List
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans

from util import Mode, data_type
from util import get_read_fn
from util.logger import print_log
from util.accuracy import (
    get_accuracy,
    get_formatted_acc_str,
    correct_overlapping_predictions,
)
from .growsp_util import get_fixclassifier
from vis.visualize_pointcloud import create_evoms_prediction_vis


def evaluate(
    config: EasyDict,
    dataset_config: EasyDict,
    data_loader: DataLoader,
    model: nn.Module,
    cls: nn.Linear,
    test_set: bool = False,
    unlabeled: bool = False,
    visualize: bool = False,
    vis_path: str = None,
    logger: logging.Logger = None,
) -> None:
    """Evaluate the model once and log accuracy metrics. (Accuracy metrics over labeled data). If no labeled data is available,
    a message that states so is displayed instead

    Args:
        config (EasyDict): model config dict
        dataset_config (EasyDict): dataset config dict
        test_loader (DataLoader): test dataset loader. Mode should be set to 'eval'
        model (nn.Module): neural network backbone
        cls (nn.Linear): linear classifier
        test_set (bool): If True, use test set instead of training set. Defaults to False.
        unlabeled (bool): If True, no accuracy computations are done. Set to True if the data contains no ground truth labels. Defaults to False.
        visualize (bool): if True, save visualization of each evaluated point cloud. Defaults to False
        logger (logging.Logger, optional): logger object. Defaults to None.
    """
    if visualize:
        assert vis_path is not None, (
            "'vis_path' must be set if 'visualize' is set to True"
        )
    if unlabeled:
        assert visualize, (
            "'unlabeled' set to True, but 'visualize' is False. Evaluating does nothing!"
        )
    model.eval()
    cls.eval()
    # Save original mode
    init_mode = data_loader.dataset.mode
    if test_set:
        data_loader.dataset.mode = Mode.test
    else:
        data_loader.dataset.mode = Mode.eval

    primitive_centers = cls.weight.data
    print_log("Merging primitives...", logger)
    cluster_pred = KMeans(
        n_clusters=config.growsp.n_overseg_classes, n_init=10, random_state=0
    ).fit_predict(primitive_centers.cpu().numpy())
    # Extract neural network backbone config
    backbone_config = config.backbone

    # Compute Class Centers
    centroids = torch.zeros(
        (config.growsp.n_overseg_classes, backbone_config.kwargs.out_channels)
    )
    for label in range(config.growsp.n_overseg_classes):
        indices = cluster_pred == label
        cluster_mean = primitive_centers[indices].mean(0, keepdims=True)
        centroids[label] = cluster_mean
    classifier = get_fixclassifier(
        backbone_config.kwargs.in_channels, config.growsp.n_overseg_classes, centroids
    ).cuda()
    classifier.eval()
    # Fetch predicted labels
    (
        overseg_pred_labels,
        gt_labels,
        all_inds,
        all_l,
        all_pred_scores,
        all_overlap_ids,
        all_full_pred_scores,
    ) = _get_predicted_labels(
        backbone_config,
        dataset_config,
        data_loader,
        model,
        classifier,
        unlabeled,
        config.growsp.l_ind,
    )
    # Fix predictions for overlapping points such that each pair of overlapping points is assigned the same
    # prediction based on whichever class has the highest score
    print_log("Solving labels for overlapping points...", logger)
    (
        corrected_overseg_pred_labels,
        all_masks,
        gt_labels,
        all_inds,
        all_l,
        all_full_pred_scores,
    ) = correct_overlapping_predictions(
        overseg_pred_labels,
        gt_labels,
        all_inds,
        all_l,
        all_pred_scores,
        all_overlap_ids,
        all_full_pred_scores,
    )
    # Cast oversegmented labels to ground truth labels
    pred_labels = _match_oversegmented_labels(
        config.growsp.l_min,
        config.growsp.n_overseg_classes,
        corrected_overseg_pred_labels,
        all_l,
    )
    pred_labels = torch.cat(pred_labels).numpy()

    if not unlabeled:
        gt_labels = torch.cat(gt_labels).numpy()
        rmv_overlap_mask = torch.cat(all_masks).numpy()
        # Match predicted classes to ground truth classes and compute accuracy
        o_acc, m_acc, m_iou, iou_array, _ = get_accuracy(
            dataset_config.N_CLASSES,
            gt_labels[rmv_overlap_mask],
            pred_labels[rmv_overlap_mask],
            hungarian_matching=True,
        )
        if test_set:
            title = "Test accuracy"
        else:
            title = "Training accuracy"
        acc_str = get_formatted_acc_str(
            o_acc,
            m_acc,
            m_iou,
            iou_array,
            dataset_config.N_CLASSES,
            dataset_config.LABELS,
            title,
        )
        print_log(acc_str, logger)

    if visualize:
        if dataset_config.NAME == "EvoMS":
            pc_filenames = data_loader.dataset.filenames
            filename_extension = dataset_config.FILENAME_EXTENSION
            read_fn = get_read_fn(filename_extension)
            filename_list = [
                pc_filenames[index] + filename_extension for index in all_inds
            ]
            current_start_ind = 0
            for filename in filename_list:
                pc_path = os.path.join(dataset_config.INPUT_PATH, filename)
                pc = read_fn(pc_path)
                n_points = pc.header.point_count
                current_pred_range = range(
                    current_start_ind, current_start_ind + n_points
                )
                pred = pred_labels[current_pred_range]
                show_errors = not unlabeled  # prediction errors can only be shown in the visualization if ground truth labels are available
                create_evoms_prediction_vis(pc, pred, vis_path, filename, show_errors)
                current_start_ind += n_points
        else:
            raise NotImplementedError(
                f"Prediction visualization not implemented for dataset '{dataset_config.NAME}'"
            )
    # Restore original mode
    data_loader.dataset.mode = init_mode


def _get_predicted_labels(
    backbone_config: EasyDict,
    dataset_config: EasyDict,
    data_loader: DataLoader,
    model: nn.Module,
    classifier: nn.Linear,
    unlabeled: bool,
    l_ind: int,
    use_sp: bool = False,
) -> Tuple[List[torch.tensor], List[torch.tensor]]:
    """Iterate through the dataloader and get the predicted labels for each labeled data point. Unlabeled data points are
    ignored, since computing accuracy metrics for them is not possible, unless 'unlabeled' is set to True.

    Args:
        backbone_config (EasyDict): neural network backbone config dict
        dataset_config (EasyDict): dataset config dict
        data_loader (DataLoader): test dataset loader. Mode should be set to 'eval', 'test'
        model (nn.Module): neural network backbone
        classifier (nn.Linear): linear classifier
        unlabeled (bool): if True, predict labels for unlabeled data
        l_ind (int): index of the linearity feature in the set of extra features (0 for first index etc.)
        use_sp (bool, optional): if True, use initial superpoints when predicting labels. Defaults to False.

    Returns:
        Tuple[Dict[str, List[torch.tensor]], Dict[str, List[torch.tensor]], Dict[str, List[int]], Dict[str, List[torch.tensor]], Dict[str, List[torch.tensor]]]:
            list of predicted label tensors, list of ground truth label tensors, list of dataloader indices,
            list of predicted scores and list of overlap ids for each plot stored in a dictionary, where key
            'i' contains the values for plot with id 'i'
    """
    n_rgb_channels = data_loader.dataset.n_rgb_channels
    l_ind = 3 + n_rgb_channels + l_ind
    all_preds, all_labels, all_inds, all_l, all_pred_scores, all_overlap_ids = (
        {},
        {},
        {},
        {},
        {},
        {},
    )
    all_full_pred_scores = {}
    for data in data_loader:
        with torch.no_grad():
            coords, features, labels, inverse_map, initial_sp, index, overlap_id = data
            linearity = features[:, l_ind]
            linearity = linearity[inverse_map.long()]
            has_labels = data_loader.dataset.is_labeled[index[0]]
            filename = data_loader.dataset.filenames[index[0]]
            # id of the plot the current point cloud belongs to
            plot_id = filename.split("_")[1]
            if has_labels or unlabeled:
                if backbone_config.type is data_type.sparse:
                    # If datapoint has labels, fetch predictions
                    in_field = ME.TensorField(
                        features[:, 0 : backbone_config.kwargs.in_channels],
                        coords,
                        device=0,
                    )
                    neural_features = model(in_field)
                else:
                    raise NotImplementedError(
                        f"Evaluate not implemented for backbone of type '{backbone_config.type}'"
                    )

                neural_features = F.normalize(neural_features, dim=-2)
                initial_sp = initial_sp.squeeze()

                if use_sp:
                    sp_labels = torch.unique(initial_sp)
                    sp_features = []
                    for sp in sp_labels:
                        if sp != -1:
                            valid_mask = sp == initial_sp
                            sp_features.append(
                                neural_features[valid_mask].mean(0, keepdim=True)
                            )
                    sp_features = torch.cat(sp_features, dim=0)
                    scores = F.linear(
                        F.normalize(neural_features), F.normalize(classifier.weight)
                    )
                    pred_scores, preds = torch.max(scores, dim=1)
                    pred_scores, preds = pred_scores.cpu(), preds.cpu()

                    sp_scores = F.linear(
                        F.normalize(sp_features), F.normalize(classifier.weight)
                    )
                    sp_num = 0
                    for sp in sp_labels:
                        if sp != -1:
                            valid_mask = sp == initial_sp
                            preds[valid_mask] = torch.argmax(sp_scores, dim=1).cpu()[
                                sp_num
                            ]
                            sp_num += 1
                else:
                    scores = F.linear(
                        F.normalize(neural_features), F.normalize(classifier.weight)
                    )
                    pred_scores, preds = torch.max(scores, dim=1)
                    pred_scores, preds = pred_scores.cpu(), preds.cpu()

                # Check if list for current plot id already exists in the dict. If not, create it
                list_exists = all_preds.get(plot_id) is not None
                if not list_exists:
                    all_preds[plot_id], all_labels[plot_id], all_inds[plot_id] = (
                        [],
                        [],
                        [],
                    )
                    (
                        all_l[plot_id],
                        all_pred_scores[plot_id],
                        all_overlap_ids[plot_id],
                    ) = [], [], []
                    all_full_pred_scores[plot_id] = []

                preds = preds[inverse_map.long()]
                pred_scores = pred_scores[inverse_map.long()]
                scores = scores.cpu()
                scores = scores[inverse_map.long(), :]
                all_preds[plot_id].append(preds[labels != dataset_config.IGNORE_LABEL])
                all_labels[plot_id].append(
                    labels[labels != dataset_config.IGNORE_LABEL]
                )
                all_inds[plot_id].append(index[0])
                all_l[plot_id].append(linearity[labels != dataset_config.IGNORE_LABEL])
                all_pred_scores[plot_id].append(
                    pred_scores[labels != dataset_config.IGNORE_LABEL]
                )
                all_overlap_ids[plot_id].append(
                    overlap_id[labels != dataset_config.IGNORE_LABEL]
                )
                all_full_pred_scores[plot_id].append(
                    scores[labels != dataset_config.IGNORE_LABEL, :]
                )

    return (
        all_preds,
        all_labels,
        all_inds,
        all_l,
        all_pred_scores,
        all_overlap_ids,
        all_full_pred_scores,
    )


def _match_oversegmented_labels(
    l_min: float,
    n_overseg_classes: int,
    overseg_pred_labels: List[torch.tensor],
    all_l: List[torch.tensor],
    foliage_label: int = 0,
    wood_label: int = 1,
) -> List[torch.tensor]:
    """Assign each class in the oversegmentation to either foliage or wood

    Args:
        l_min (float): linearity threshold. Oversegmented classes with average linearity higher than this are considered wood
        n_overseg_classes (int): number of classes in the oversegmentation. It is assumed that the labels range from 0 to (n_overseg_labels - 1)
        overseg_pred_labels (List[torch.tensor]): output of _get_predicted_labels() containing oversegmented class labels of each input point cloud
        all_l (List[torch.tensor]): list of linearity tensors corresponding to the oversegmented class labels (output of _get_predicted_labels())
        foliage_label (int, optional): label used for classes assigned to foliage. Defaults to 0.
        wood_label (int, optional): label used for classes assigned to wood. Defaults to 1.

    Returns:
        List[torch.tensor]:
    """
    assigned_wood = []
    # Iterate over all oversegmented class labels
    for c in range(n_overseg_classes):
        c_linearity = []
        for overseg_labels, linearity in zip(overseg_pred_labels, all_l):
            c_mask = overseg_labels == c
            c_linearity.append(linearity[c_mask])
        c_linearity = torch.cat(c_linearity)
        c_mean_linearity = torch.mean(c_linearity).item()
        c_mean_linearity = np.round(c_mean_linearity, 2)  # round to 2 decimals
        # If mean linearity is over the required threshold, assign the current oversegmented class to wood, otherwise, assign it to foliage
        if c_mean_linearity >= l_min:
            assigned_wood.append(c)
    # List for matched ground truth label tensors
    pred_labels = []
    for overseg_labels in overseg_pred_labels:
        matched_labels = torch.full_like(overseg_labels, foliage_label)
        for c in assigned_wood:
            c_mask = overseg_labels == c
            matched_labels[c_mask] = wood_label
        pred_labels.append(matched_labels)

    return pred_labels
