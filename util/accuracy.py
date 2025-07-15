import torch

import numpy as np

from typing import Tuple, List, Union, Dict
from scipy.stats import mode
from scipy.optimize import linear_sum_assignment
from easydict import EasyDict
from tabulate import tabulate


"""
TODO/NOTE: for future consideration - the accuracy computations break for OA and mAcc if the ground truth labels do not contain
at least one instance of all possible semantic classes. Fixing is not a high priority (although it should not be that much work),
since such a case does not occur in the current EvoMS dataset.
"""


def get_superpoint_accuracy(
    n_classes: int, gt_labels: List[np.ndarray], sp_labels: List[np.ndarray]
) -> Tuple[float, float, float, np.ndarray]:
    """Compute superpoint accuracy metrics. In the accuracy computations the label assigned to each superpoint is simply
    the dominant semantic class in that superpoint

    Args:
        n_classes (int): number of semantic classes in the data
        gt_labels List[(np.ndarray)]: ground truth labels. Here we assume that the labels are between [0, n_classes - 1]. Labels should be provided in a list of arrays where each array contains the labels of one input point cloud.
        sp_labels List[(np.ndarray)]: superpoint labels for each point. The format for the labels should be the same as for 'gt_labels'.

    Returns:
        Tuple[float, float, float, np.ndarray]: oAcc, mAcc, mIoU and class specific IoUs in an array
    """
    assert len(gt_labels) == len(sp_labels), (
        f"'gt_labels' and 'sp_labels' should contain the same number of label arrays [{len(gt_labels)} != {len(sp_labels)}]"
    )
    # Find semantic labels for each superpoint
    sp_gt_labels = []
    for sp_labels_i, gt_labels_i in zip(sp_labels, gt_labels):
        sp_gt_labels_i = -np.ones_like(gt_labels_i)
        for sp in np.unique(sp_labels_i):  # Process by superpoint
            if sp != -1:
                sp_mask = sp == sp_labels_i
                sp_gt_labels_i[sp_mask] = mode(gt_labels_i[sp_mask], keepdims=True)[0][
                    0
                ]
        sp_gt_labels.append(sp_gt_labels_i)
    sp_gt_labels = np.concatenate(sp_gt_labels)
    gt_labels = np.concatenate(gt_labels)

    return get_accuracy(n_classes, gt_labels, sp_gt_labels)


def sp_labels_to_gt(
    gt_labels: Union[np.ndarray, None], sp_labels: np.ndarray
) -> Union[np.ndarray, None]:
    if gt_labels is None:
        return None  # labels cannot be computed -> return None
    sp_gt_labels = -np.ones_like(gt_labels)
    for sp in np.unique(sp_labels):
        if sp != -1:
            sp_mask = sp == sp_labels
            sp_gt_labels[sp_mask] = mode(gt_labels[sp_mask], keepdims=True)[0][0]

    return sp_gt_labels


def get_accuracy(
    n_classes: int,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    hungarian_matching: bool = False,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """Compute accuracy metrics

    Args:
        n_classes (int): number of semantic classes in the data
        gt_labels (np.ndarray): ground truth labels. Here we assume that the labels are between [0, n_classes - 1]
        pred_labels (np.ndarray): predicted labels for each point
        hungarian_matching (bool, optional): if True, perform hungarian matching for the ground truth and predicted labels prior to computing the accuracy metrics. This should be done if the labels between the two arrays do not match (e.g. class x is always labeled with 1 in gt and 3 in pred). Defaults to False.

    Returns:
        Tuple[float, float, float, np.ndarray, np.ndarray]: oAcc, mAcc, mIoU and class specific IoUs and accuracies in two arrays
    """
    mask = (gt_labels >= 0) & (gt_labels < n_classes) & (pred_labels >= 0)
    # The code within the bincount() function constructs an unique index for each possible combination of ground truth label and
    # predicted label. For example, if the value of the ground truth array is x and the value of the predicted label
    # is y, all positions in the resulting array where x and y are in the same position will have the same unique value and any other
    # combination will have a different value. Effectively, the line below constructs a confusion matrix
    conf_matrix = np.bincount(
        n_classes * gt_labels[mask].astype(np.int64)
        + pred_labels[mask].astype(np.int64),
        minlength=n_classes**2,
    ).reshape(n_classes, n_classes)
    if hungarian_matching:
        conf_matrix = _match_pred_to_gt(n_classes, conf_matrix)
    tp = np.diag(
        conf_matrix
    )  # Elements where the ground truth and superpoint label match
    o_acc = (tp.sum() / conf_matrix.sum()) * 100
    fp = np.sum(conf_matrix, 1) - tp  # sum of rows
    fn = np.sum(conf_matrix, 0) - tp  # sum of columns
    m_acc = np.nanmean(tp / (tp + fn)) * 100
    acc_array = (tp / (tp + fn + 1e-8)) * 100
    iou_array = (tp / (tp + fp + fn + 1e-8)) * 100
    m_iou = np.nanmean(iou_array)

    return o_acc, m_acc, m_iou, iou_array, acc_array


def _match_pred_to_gt(n_classes: int, conf_matrix: np.ndarray) -> np.ndarray:
    """Match the predicted labels to ground truth labels in a case where classes are labeled differently betwen the two.
    For example, if class x has label 0 in ground truth and label 2 in predictions, the function rearranges the confusion
    matrix such that the two classes are matched

    Args:
        n_classes (int): number of semantic classes in the data
        conf_matrix (np.ndarray): confusion matrix computed with the mismatched ground truth and prediction labels

    Returns:
        np.ndarray: confusion matrix in the correct order
    """
    _, col_ind = linear_sum_assignment(conf_matrix.max() - conf_matrix)
    conf_matrix_matched = np.zeros_like(conf_matrix)
    for idx in range(n_classes):
        conf_matrix_matched[:, idx] = conf_matrix[:, col_ind[idx]]

    return conf_matrix_matched


def get_formatted_acc_str(
    o_acc: float,
    m_acc: float,
    m_iou: float,
    iou_array: np.ndarray,
    n_classes: int,
    label_map: EasyDict,
    title: str = None,
) -> str:
    """Generate a formatted string of the accuracy metrics for printing/logging purposes

    Args:
        o_acc (float): overall accuracy
        m_acc (float): mean accuracy
        m_iou (float): mean intersection over union
        iou_array (np.ndarray): intersection over union of each class
        n_classes (int): number of semantic classes
        label_map (EasyDict): dict where the key value pairs are the intger label (as a string) and the name of the class
        title (str, optional): title for the accuracy metrics. Defaults to None.

    Returns:
        str: accuracy metrics as a (nicely) formatted string
    """
    headers = ["OA", "mAcc", "mIoU"]
    table = [[o_acc, m_acc, m_iou]]
    for label in range(n_classes):
        headers.append(label_map[str(label)])
        table[0].append(iou_array[label])
    formatted_acc_str = tabulate(table, headers=headers)
    # Add title to table (if defined)
    if title is not None:
        first_line = formatted_acc_str.partition("\n")[0]
        line_len = len(first_line)
        title_decorator = ((line_len - len(title)) // 2 - 1) * "-"
        title_line = title_decorator + " " + title + " " + title_decorator
        if len(title_line) < line_len:
            title_line += "-"  # Cannot create a completely centered title
        title_line += "\n"
        formatted_acc_str = "\n" + title_line + formatted_acc_str

    return formatted_acc_str


def get_formatted_acc_str_w_boundary_metrics(
    o_acc: float,
    m_acc: float,
    m_iou: float,
    iou_array: np.ndarray,
    br: float,
    bp: float,
    n_classes: int,
    label_map: EasyDict,
    title: str = None,
) -> str:
    """Generate a formatted string of the accuracy metrics for printing/logging purposes that includes the superpoint boundary accuracy metrics

    Args:
        o_acc (float): overall accuracy
        m_acc (float): mean accuracy
        m_iou (float): mean intersection over union
        br (float): superpoint boundary recall
        bp (float): superpoint boundary precision
        iou_array (np.ndarray): intersection over union of each class
        n_classes (int): number of semantic classes
        label_map (EasyDict): dict where the key value pairs are the intger label (as a string) and the name of the class
        title (str, optional): title for the accuracy metrics. Defaults to None.

    Returns:
        str: accuracy metrics as a (nicely) formatted string
    """
    headers = ["OA", "mAcc", "mIoU"]
    table = [[o_acc, m_acc, m_iou]]
    for label in range(n_classes):
        headers.append(label_map[str(label)])
        table[0].append(iou_array[label])
    headers.extend(["BR", "BP"])
    table[0].extend([br, bp])
    formatted_acc_str = tabulate(table, headers=headers)
    # Add title to table (if defined)
    if title is not None:
        first_line = formatted_acc_str.partition("\n")[0]
        line_len = len(first_line)
        title_decorator = ((line_len - len(title)) // 2 - 1) * "-"
        title_line = title_decorator + " " + title + " " + title_decorator
        if len(title_line) < line_len:
            title_line += "-"  # Cannot create a completely centered title
        title_line += "\n"
        formatted_acc_str = "\n" + title_line + formatted_acc_str

    return formatted_acc_str


def get_formatted_class_acc_str(
    acc_array: np.ndarray, n_classes: int, label_map: EasyDict, title: str = None
) -> str:
    headers = []
    table = [[]]
    for label in range(n_classes):
        headers.append(label_map[str(label)])
        table[0].append(acc_array[label])
    formatted_class_acc_str = tabulate(table, headers=headers)
    # Add title to table (if defined)
    if title is not None:
        first_line = formatted_class_acc_str.partition("\n")[0]
        line_len = len(first_line)
        title_decorator = ((line_len - len(title)) // 2 - 1) * "-"
        title_line = title_decorator + " " + title + " " + title_decorator
        if len(title_line) < line_len:
            title_line += "-"  # Cannot create a completely centered title
        title_line += "\n"
        formatted_class_acc_str = "\n" + title_line + formatted_class_acc_str

    return formatted_class_acc_str


def correct_overlapping_predictions(
    all_preds: Dict[str, List[torch.tensor]],
    all_labels: Dict[str, List[torch.tensor]],
    all_inds: Dict[str, List[int]],
    all_l: Dict[str, List[torch.tensor]],
    all_pred_scores: Dict[str, List[torch.tensor]],
    all_overlap_ids: Dict[str, List[torch.tensor]],
    all_full_pred_scores,
) -> Tuple[
    List[torch.tensor],
    List[torch.tensor],
    List[torch.tensor],
    List[int],
    List[torch.tensor],
]:
    """This function corrects the predictions within overlapping regions such that each overlapping point
    is assigned the same class based on the highest probability class across all predictions for the given point
    (each overlapping point has at most two predicted classes)

    Args:
        all_preds (Dict[str, List[torch.tensor]]): output of _get_predicted_labels
        all_labels (Dict[str, List[torch.tensor]]): output of _get_predicted_labels
        all_inds (Dict[str, List[int]]): output of _get_predicted_labels
        all_l (Dict[str, List[torch.tensor]]): output of _get_predicted_labels
        all_pred_scores (Dict[str, List[torch.tensor]]): output of _get_predicted_labels
        all_overlap_ids (Dict[str, List[torch.tensor]]): output of _get_predicted_labels

    Returns:
        Tuple[List[torch.tensor], List[torch.tensor], List[torch.tensor], List[int], List[torch.tensor]]:
            corrected predictions for each plot, list of masks that removes overlapping points (for accuracy
            computations such that overlapping points are not considered twice). Lists of labels, indices and
            linearities for each plot extracted from the input dictionaries
    """
    corrected_all_preds = []
    all_masks = []
    all_labels_list, all_inds_list, all_l_list, all_full_scores_list = [], [], [], []
    for plot_id in all_preds.keys():
        # Extract gt labels, indices and linearity from dict to list (they were put there to ensure that the
        # order w.r.t. the predicted labels is preserved
        all_labels_list.append(torch.cat(all_labels[plot_id]))
        all_inds_list.extend(all_inds[plot_id])
        all_l_list.append(torch.cat(all_l[plot_id]))
        all_full_scores_list.append(torch.cat(all_full_pred_scores[plot_id]))
        # Extract predictions, scores and overlap ids
        pred_labels = torch.cat(all_preds[plot_id]).numpy()
        pred_scores, overlap_ids = (
            torch.cat(all_pred_scores[plot_id]).numpy(),
            torch.cat(all_overlap_ids[plot_id]).numpy(),
        )
        overlap_mask = (
            overlap_ids != 0
        )  # Mask to points that are overlapping (overlap id 0 = non-overlapping point)
        if np.sum(overlap_mask) > 0:
            # Extract unique elements directly
            unq_elem, inverse_indices = np.unique(
                overlap_ids[overlap_mask], return_inverse=True
            )
            # Create an empty list of lists to store matches
            overlap_id_matches_tmp = [[] for _ in range(len(unq_elem))]
            # Populate overlap_id_matches using inverse indices
            for idx, val in enumerate(inverse_indices):
                overlap_id_matches_tmp[val].append(idx)
            overlap_id_matches = []
            non_overlap_ind = []
            for pair in overlap_id_matches_tmp:
                if len(pair) < 2:
                    # Due to data being separated to train and test sets, overlap does not necessarily always
                    # occur for overlap points
                    non_overlap_ind.append(pair[0])
                else:
                    overlap_id_matches.append(pair)
            overlap_id_matches = np.array(overlap_id_matches)
            # Scores for overlapping points sorted by matching ids
            overlap_scores = pred_scores[overlap_mask][overlap_id_matches]
            # Index of the higher score for each overlap point
            max_score_ind = np.argmax(overlap_scores, axis=1)
            # Indices of the overlap points that have a higher score
            correct_label_ind = overlap_id_matches[
                np.arange(0, overlap_id_matches.shape[0]), max_score_ind
            ]
            # Indices of the overlap points that have a lower score
            wrong_label_ind = overlap_id_matches[
                np.arange(0, overlap_id_matches.shape[0]), 1 - max_score_ind
            ]
            # The two lists form a mapping such that points at wrong_label_ind should be mapped to the same
            # prediction as points at correct_label_ind
            corrected_pred_labels = pred_labels.copy()
            corrected_pred_labels_tmp = corrected_pred_labels[overlap_mask]
            corrected_pred_labels_tmp[wrong_label_ind] = corrected_pred_labels_tmp[
                correct_label_ind
            ]
            corrected_pred_labels[overlap_mask] = corrected_pred_labels_tmp
            corrected_pred_labels = torch.tensor(corrected_pred_labels)
            corrected_all_preds.append(corrected_pred_labels)
            # Create a mask that removes the overlapping points with lower scores
            keep_mask = np.ones_like(overlap_mask)
            overlap_point_ind = np.nonzero(overlap_mask)[0]
            keep_mask_tmp = keep_mask[overlap_point_ind]
            keep_mask_tmp[wrong_label_ind] = False
            keep_mask[overlap_point_ind] = keep_mask_tmp
        else:
            # If no overlaps do nothing
            corrected_pred_labels = torch.tensor(pred_labels)
            corrected_all_preds.append(corrected_pred_labels)
            keep_mask = np.ones_like(overlap_mask)

        all_masks.append(torch.tensor(keep_mask))

    return (
        corrected_all_preds,
        all_masks,
        all_labels_list,
        all_inds_list,
        all_l_list,
        all_full_scores_list,
    )


def get_superpoint_boundary_accuracy(
    gt_labels: List[np.ndarray],
    sp_labels: List[np.ndarray],
    source: List[np.ndarray],
    target: List[np.ndarray],
) -> Tuple[float, float]:
    """Compute superpoint boundary recall and boundary precision, as described in "Point Cloud Oversegmentation
    with Graph-Structured Deep Metric Learning", (Landrieu et al., 2019)

    Args:
        gt_labels (List[np.ndarray]): list of ground truth labels for each point cloud
        sp_labels (List[np.ndarray]): list of superpoint labels for each point cloud
        source (List[np.ndarray]): list of lists containing source indices for all edges in the graph used for generating the superpoints
        target (List[np.ndarray]): list of lists containing target indices for all edges in the graph used for generating the superpoints

    Returns:
        Tuple[float, float]: boundary recall and boundary precision
    """
    br_numerator = 0
    br_denominator = 0
    bp_numerator = 0
    bp_denominator = 0

    for gt, sp, src, tgt in zip(gt_labels, sp_labels, source, target):
        # Boolean masks for edges crossing superpoint boundaries and GT boundaries
        pred_transition = sp[src] != sp[tgt]
        is_transition = gt[src] != gt[tgt]
        br_num, br_denom = _get_br_parts(
            is_transition, _relax_edge_binary(pred_transition, src, tgt, gt.shape[0])
        )
        bp_num, bp_denom = _get_bp_parts(
            _relax_edge_binary(is_transition, src, tgt, gt.shape[0]), pred_transition
        )
        br_numerator += br_num
        br_denominator += br_denom
        bp_numerator += bp_num
        bp_denominator += bp_denom

    # Avoid division by zero
    br = (br_numerator / br_denominator) * 100 if br_denominator > 0 else 0.0
    bp = (bp_numerator / bp_denominator) * 100 if br_denominator > 0 else 0.0

    return br, bp


def _get_br_parts(
    is_transition: np.ndarray, pred_transition: np.ndarray
) -> Tuple[float, float]:
    br_num = ((is_transition == pred_transition) * is_transition).sum()
    br_denom = is_transition.sum()
    return br_num, br_denom


def _get_bp_parts(
    is_transition: np.ndarray, pred_transition: np.ndarray
) -> Tuple[float, float]:
    bp_num = ((is_transition == pred_transition) * pred_transition).sum()
    bp_denom = pred_transition.sum()
    return bp_num, bp_denom


def _relax_edge_binary(
    edge_binary: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    n_vertices: int,
    tolerance: int = 1,
) -> np.ndarray:
    """Relax a binary mask of transitions such that it also contains all edges within 'tolerance'. Code copied
    from: https://github.com/loicland/superpoint_graph/blob/ssp%2Bspg/supervized_partition/losses.py

    Args:
        edge_binary (np.ndarray): binary mask that is true for edges that are transitions
        source (np.ndarray): indices of source nodes
        target (np.ndarray): indices of target nodes
        n_vertices (int): number of vertices in the graph
        tolerance (int, optional): Tolerance to relax to. Defaults to 1.

    Returns:
        np.ndarray: relaxed binary mask
    """
    relaxed_binary = edge_binary.copy()
    transition_vertex = np.full((n_vertices,), 0, dtype="uint8")
    for i_tolerance in range(tolerance):
        transition_vertex[source[relaxed_binary.nonzero()]] = True
        transition_vertex[target[relaxed_binary.nonzero()]] = True
        relaxed_binary[transition_vertex[source]] = True
        relaxed_binary[transition_vertex[target] > 0] = True
    return relaxed_binary
