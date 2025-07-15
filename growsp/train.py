import argparse
import time
import logging
import math
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
import MinkowskiEngine as ME
import torch.nn.functional as F

from easydict import EasyDict
from typing import Union
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from util import data_type
from .evaluate import evaluate
from util.logger import print_log
from models.model_util import save_checkpoint
from util.accuracy import get_accuracy, get_formatted_acc_str
from .growsp_util import (
    get_sp_features,
    get_sp_features_geometric,
    get_pseudo_labels,
    get_fixclassifier,
)


def train(
    config: EasyDict,
    args: argparse.Namespace,
    train_loader: DataLoader,
    cluster_loader: DataLoader,
    test_loader: DataLoader,
    model: nn.Module,
    optimizer_1: optim.Optimizer,
    optimizer_2: optim.Optimizer,
    scheduler_1: optim.lr_scheduler._LRScheduler,
    scheduler_2: optim.lr_scheduler._LRScheduler,
    loss,
    state: str = "pretrain",
    ckpt_epoch: int = 0,
    classifier: nn.Linear = None,
    logger: logging.Logger = None,
) -> None:
    """Train GrowSP

    Args:
        config (EasyDict): config dict
        args (argparse.Namespace): args passed to main
        train_loader (DataLoader): data loader for backbone training (mode should be 'train')
        cluster_loader (DataLoader): data loader for clustering into semantic primitives (mode should be 'cluster')
        test_loader (DataLoader): data loader for evaluating model performance (mode should be 'evaluate')
        model (nn.Module): neural network backbone
        optimizer_1 (optim.Optimizer): optimizer for the pretrain stage where superpoints are not growing
        optimizer_2 (optim.Optimizer): optimizer for the growth stage where superpoints are growing
        scheduler_1 (optim.lr_scheduler._LRScheduler): learning rate scheduler for the pretrain stage
        scheduler_2 (optim.lr_scheduler._LRScheduler): learning rate scheduler for the growing stage
        loss (_type_): loss function for the neural network backbone
        state (str, optional): curren state. Shoule be one ['pretrain', 'grow']. Defaults to "pretrain".
        ckpt_epoch (int, optional): checkpoint to start from (if model was loaded from checkpoint). Defaults to 0.
        classifier (nn.Linear): linear classifier. Only required if training resumes from a checkpoint. Defaults to None
        logger (logging.Logger, optional): current logger. Defaults to None.
    """
    assert state == "pretrain" or state == "grow", (
        f"'state' must be one of ['pretrain', 'grow'], got '{state}' instead!"
    )
    if ckpt_epoch > 0:
        assert classifier is not None, "'classifier' is required if 'ckpt_epoch' > 0!"

    start_grow_epoch = (
        0  # Epoch at which GrowSP stopped pretraining and entered 'grow' state
    )
    growsp_config = config.growsp

    if state == "pretrain":  # Pretrain skipped if state is 'grow'
        is_growing = False
        # Train and cluster. In the first stage the superpoints do not grow
        first_epoch = np.max([1, ckpt_epoch])  # Begin at checkpoint epoch, if given
        ckpt_epoch = 0  # If state is pretrain, set checkpoint epoch to zero such that training starts from the first epoch in the 'grow' state
        for epoch in range(first_epoch, growsp_config.pretrain_epochs + 1):
            # Cluster every 'cluster_interval' epochs
            if (epoch - 1) % config.growsp.cluster_interval == 0:
                classifier = _cluster(
                    config,
                    args,
                    cluster_loader,
                    model,
                    epoch,
                    start_grow_epoch,
                    is_growing,
                    logger,
                )
            _train_one_epoch(
                config,
                train_loader,
                model,
                optimizer_1,
                loss,
                epoch,
                scheduler_1,
                classifier,
                logger,
            )

            if epoch % config.growsp.cluster_interval == 0:
                # Save checkpoint
                ckpt_filename = f"{args.experiment_name}_{epoch}_ckpt.pth"
                save_checkpoint(
                    model,
                    classifier,
                    optimizer_1,
                    optimizer_2,
                    scheduler_1,
                    scheduler_2,
                    epoch,
                    state,
                    ckpt_filename,
                    args,
                    config.growsp.cluster_interval,
                    logger,
                )
                # Compute current accuracy metrics
                with torch.no_grad():
                    evaluate(
                        config,
                        args.dataset_config,
                        test_loader,
                        model,
                        classifier,
                        logger=logger,
                    )

                iterations = (epoch + config.growsp.cluster_interval) * len(
                    train_loader
                )
                # Stop pretraining if maximum number of iterations defined in config is reached
                if iterations > growsp_config.pretrain_max_iter:
                    start_grow_epoch = epoch
                    break
            start_grow_epoch = epoch
    if state == "grow" or start_grow_epoch == 0:
        # GrowSP was loaded from checkpoint -> we infer the start_grow_epoch value from the number of datapoints in the trainloader
        for epoch in range(1, growsp_config.pretrain_epochs + 1):
            if epoch % config.growsp.cluster_interval == 0:
                iterations = (epoch + config.growsp.cluster_interval) * len(
                    train_loader
                )
                if iterations > growsp_config.pretrain_max_iter:
                    start_grow_epoch = epoch
                    break
            start_grow_epoch = epoch

    state = "grow"
    is_growing = True
    first_epoch = np.max([1, ckpt_epoch])
    # If this is the first epoch of growing superpoints, print message and create a checkpoint
    if first_epoch == 1:
        state_change_msg = "### Superpoints begin growing ###"
        msg_decorator = "#" * len(state_change_msg)
        state_change_msg = (
            "\n" + msg_decorator + "\n" + state_change_msg + "\n" + msg_decorator
        )
        print_log(state_change_msg, logger)
        ckpt_filename = (
            f"{args.experiment_name}_{first_epoch + start_grow_epoch}_ckpt.pth"
        )
        save_checkpoint(
            model,
            classifier,
            optimizer_1,
            optimizer_2,
            scheduler_1,
            scheduler_2,
            first_epoch - 1,
            state,
            ckpt_filename,
            args,
            config.growsp.cluster_interval,
            logger,
        )
    for epoch in range(first_epoch, growsp_config.grow_epochs + 1):
        orig_epoch = epoch  # Needed for checkpoint saving
        epoch += start_grow_epoch

        # Cluster every 'cluster_interval' epochs
        if (epoch - 1) % config.growsp.cluster_interval == 0:
            classifier = _cluster(
                config,
                args,
                cluster_loader,
                model,
                epoch,
                start_grow_epoch,
                is_growing,
                logger,
            )
        _train_one_epoch(
            config,
            train_loader,
            model,
            optimizer_2,
            loss,
            epoch,
            scheduler_2,
            classifier,
            logger,
        )

        if epoch % config.growsp.cluster_interval == 0:
            # Save checkpoint
            ckpt_filename = f"{args.experiment_name}_{epoch}_ckpt.pth"
            save_checkpoint(
                model,
                classifier,
                optimizer_1,
                optimizer_2,
                scheduler_1,
                scheduler_2,
                orig_epoch,
                state,
                ckpt_filename,
                args,
                config.growsp.cluster_interval,
                logger,
            )
            # Compute current accuracy metrics
            with torch.no_grad():
                evaluate(
                    config,
                    args.dataset_config,
                    test_loader,
                    model,
                    classifier,
                    logger=logger,
                )


def _cluster(
    config: EasyDict,
    args: argparse.Namespace,
    cluster_loader: DataLoader,
    model: nn.Module,
    epoch: int,
    start_grow_epoch: Union[int, None],
    is_growing: bool = False,
    logger: logging.Logger = None,
) -> nn.Linear:
    """Fetch superpoint features and perform clustering to form semantic primitives. Save pseudo labels for all
    datapoints.

    Args:
        config (EasyDict): config dict
        args (argparse.Namespace): args passed to main
        cluster_loader (DataLoader): cluster dataloader (mode should be 'cluster')
        model (nn.Module): neural network backbone
        epoch (int): current epoch
        start_grow_epoch (Union[int, None]): epoch at which the superpoints started growing (None if they're not growing yet)
        is_growing (bool, optional): True if the superpoints are growing. Defaults to False.
        logger (logging.Logger, optional): current logger. Defaults to None.

    Returns:
        nn.Linear: linear classifier that maps neural features to semantic primitives
    """
    time_start = time.time()
    current_num_sp = None
    growsp_config = config.growsp
    if is_growing:
        current_num_sp = int(
            growsp_config.n_sp_start
            - (
                (epoch - start_grow_epoch - 1)
                / (growsp_config.grow_epochs - config.growsp.cluster_interval)
            )
            * (growsp_config.n_sp_start - growsp_config.n_sp_end)
        )
        current_num_sp = np.max([current_num_sp, growsp_config.n_sp_end])
        grow_msg = (
            f"@ epoch {epoch} | number of superpoints reduces to {current_num_sp}"
        )
        print_log(grow_msg, logger)

    # Extract superpoint features
    sp_features_list, all_sp_labels, context = get_sp_features_geometric(
        config, cluster_loader, model, current_num_sp, epoch, logger
    )
    # NOTE: if you want to use PFHs instead of geometric features, comment out the function call above and replace it with the one below
    # sp_features_list, all_sp_labels, context = get_sp_features(
    #    config, cluster_loader, model, current_num_sp, epoch, logger
    # )
    # concat features from all batches into one tensor
    sp_features = torch.cat(sp_features_list, dim=0)
    # Find primitive labels with kmeans using extracted superpoint features
    print_log("Computing semantic primitives...", logger)
    primitive_labels = KMeans(
        n_clusters=growsp_config.n_primitives, n_init=5, random_state=0
    ).fit_predict(sp_features.numpy())
    sp_features = sp_features[
        :, 0 : config.backbone.kwargs.out_channels
    ]  # drop geometric feature as it's not needed in primitve centroid computation

    # Compute primitive centers
    primitive_centers = torch.zeros(
        (growsp_config.n_primitives, config.backbone.kwargs.out_channels)
    )
    for primitive in range(growsp_config.n_primitives):
        indices = primitive_labels == primitive
        cluster_mean = sp_features[indices].mean(0, keepdims=True)
        primitive_centers[primitive] = cluster_mean
    classifier = get_fixclassifier(
        config.backbone.kwargs.out_channels,
        growsp_config.n_primitives,
        primitive_centers,
    )

    # Compute and save pseudo labels
    all_pseudo_labels, all_gt_labels, all_pseudo_gt_labels, labeled_mask = (
        get_pseudo_labels(
            args.dataset_config.PSEUDO_PATH,
            context,
            primitive_labels,
            all_sp_labels,
            logger,
        )
    )
    labeled_ratio = (all_pseudo_labels != -1).sum() / all_pseudo_labels.shape[0]
    cluster_msg = f"Labeled points ratio: {labeled_ratio:.3f} | clustering time: {(time.time() - time_start):.3f} s"
    print_log(cluster_msg, logger)

    can_compute_acc = all_gt_labels.size > 0
    # Accuracy metrics can only be computed if ground truth labels are available for some data points
    if can_compute_acc:
        valid_mask = all_pseudo_gt_labels != -1
        # Compute superpoint and primitive accuracy metrics and log them
        o_acc, m_acc, m_iou, iou_array, _ = get_accuracy(
            args.dataset_config.N_CLASSES,
            all_gt_labels[valid_mask],
            all_pseudo_gt_labels[valid_mask],
        )
        sp_acc_str = get_formatted_acc_str(
            o_acc,
            m_acc,
            m_iou,
            iou_array,
            args.dataset_config.N_CLASSES,
            args.dataset_config.LABELS,
            "Superpoint accuracy",
        )
        print_log(sp_acc_str, logger)

        # Cast primitive labels to ground truth by finding the most common ground truth label within each primitive
        pseudo_labels_to_gt = -np.ones_like(all_gt_labels, dtype=np.int8)
        for i in range(growsp_config.n_primitives):
            mask = all_pseudo_labels[labeled_mask] == i
            try:
                pseudo_labels_to_gt[mask] = torch.mode(
                    torch.from_numpy(all_gt_labels[mask])
                ).values
            except IndexError:
                pass  # It can happen that some primitive contains no points -> torch.mode() raises an expception
        valid_mask = (pseudo_labels_to_gt != -1) & (all_gt_labels != -1)
        o_acc, m_acc, m_iou, iou_array, _ = get_accuracy(
            args.dataset_config.N_CLASSES,
            all_gt_labels[valid_mask],
            pseudo_labels_to_gt[valid_mask],
        )
        primitive_acc_str = get_formatted_acc_str(
            o_acc,
            m_acc,
            m_iou,
            iou_array,
            args.dataset_config.N_CLASSES,
            args.dataset_config.LABELS,
            "Primitive accuracy",
        )
        print_log(primitive_acc_str, logger)
    else:
        print_log("No labeled data available, cannot compute accuracy metrics!", logger)

    return classifier.cuda()


def _train_one_epoch(
    config: EasyDict,
    train_loader: DataLoader,
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss,
    epoch: int,
    scheduler: optim.lr_scheduler._LRScheduler,
    classifier: nn.Linear,
    logger: logging.Logger = None,
) -> None:
    """Train the GrowSP neural network backbone for one epoch

    Args:
        config (EasyDict): config as a dict
        train_loader (DataLoader): training data loader (mode should be 'train')
        model (nn.Module): neural network backbone
        optimizer (optim.Optimizer): current optimizer
        loss: current loss function
        epoch (int): current epoch
        scheduler (optim.lr_scheduler._LRScheduler): current scheduler
        classifier (nn.Linear): linear classifier returned by the latest call to cluster()
        logger (logging.Logger, optional): current logger. Defaults to None.
    """
    model.train()
    # Loss is accumulated here for the purposes of dispalying mean loss every x batches
    loss_display = 0
    time_current = time.time()

    for batch_ind, data in enumerate(train_loader):
        coords, features, _, _, _, pseudo_labels, _, _ = data
        # Pass data through model
        if config.backbone.type is data_type.sparse:
            in_field = ME.TensorField(
                features[:, 0 : config.backbone.kwargs.in_channels], coords, device=0
            )
            neural_features = model(in_field)
            del coords, features
        else:
            raise NotImplementedError(
                f"Train not implemented for backbone of type '{config.backbone.type}'"
            )

        neural_features = F.normalize(neural_features, dim=-2)
        # Compute loss
        pseudo_labels = pseudo_labels.long().cuda()
        logits = F.linear(F.normalize(neural_features), F.normalize(classifier.weight))
        batch_loss = loss(logits * 3, pseudo_labels).mean()
        # Optimize
        loss_display += batch_loss.item()
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        scheduler.step()

        del batch_loss, in_field, neural_features, pseudo_labels, logits
        torch.cuda.empty_cache()
        torch.cuda.synchronize(torch.device("cuda"))
        # Display training metrics after log_interval
        if (batch_ind + 1) % config.growsp.log_interval == 0:
            time_used = time.time() - time_current
            loss_display /= config.growsp.log_interval
            n_datapoints = len(train_loader)
            # Number of digits in n_datapoints (see: https://stackoverflow.com/questions/2189800/how-to-find-length-of-digits-in-an-integer)
            n_digits = int(math.log10(n_datapoints)) + 1
            progress_pct = (batch_ind + 1) / n_datapoints
            progress_msg = (
                f"@ epoch {epoch:4} | batch: [{(batch_ind + 1):{n_digits}}/{n_datapoints} ({progress_pct:4.0%})] | mean loss: {loss_display:.8f} "
                f"| lr: {scheduler.get_last_lr()[0]:.8f} | Time: {time_used:.3f} s ({config.growsp.log_interval} iterations)"
            )
            print_log(progress_msg, logger)
            time_current = time.time()
            loss_display = 0  # Reset accumulated display loss
