import os
import warnings

from util import parser, misc, Mode, data_type
from torch.nn import Linear
from util.logger import get_logger, print_log
from util.config_yaml import (
    cfg_from_yaml_file,
    log_args_to_file,
    copy_experiment_config,
)
from growsp import train, evaluate
from datasets import build_dataloader_from_config
from preprocess import (
    preprocess_data,
    create_superpoints,
    populate_missing_rgb,
)
from models import (
    build_model_from_config,
    build_loss_from_config,
    build_optimizer_from_config,
    build_scheduler_from_config,
    parse_load_ckpt,
)
from models.model_util import load_model, load_optimizer, load_scheduler


def main():
    warnings.showwarning = misc.showwarning  # Disable showing source line for warnings (it's annoying and the warning messages are descriptive enough)
    args = parser.get_args()
    # Load dataset config
    dataset_config = cfg_from_yaml_file(args.dataset_config_path)
    assert dataset_config.NAME == args.dataset_name, (
        f"Name in dataset config does not match argument --dataset_name"
        f"[{dataset_config.NAME} != {args.dataset_name}]"
    )
    # Add to args
    args.dataset_config = dataset_config
    # Set seed for (almost) deterministic results (see comments on set_seed())
    misc.set_seed(args.seed)
    # Check the mode
    if args.mode == "data_prepare":
        preprocess_data(args)
    elif args.mode == "initial_sp":
        create_superpoints(args)
    elif args.mode == "populate_rgb":
        populate_missing_rgb(args)
    else:
        # Initialize logger
        logger = get_logger(
            args.log_name, os.path.join(args.log_path, args.mode.value + ".log")
        )
        if args.log_args:
            # log args
            log_args_to_file(args, logger=logger)
        # Load model config
        model_config = cfg_from_yaml_file(args.config_path, logger)
        model_config.backbone.type = data_type(model_config.backbone.type)
        dataset_config.model_config = model_config
        # Dump config files to experiment path
        if not args.disable_config_dump:
            copy_experiment_config(args, args.dataset_config_path, logger)
            copy_experiment_config(args, args.config_path, logger)
        # Check if some values in the config were overloaded
        if args.l_min is not None:
            dataset_config.model_config.growsp.l_min = args.l_min
        if args.n_overseg_classes is not None:
            dataset_config.model_config.growsp.n_overseg_classes = (
                args.n_overseg_classes
            )
        # Build models
        # Build neural network backbone form config
        model = build_model_from_config(model_config.backbone)
        model = model.cuda()
        classifier = Linear(
            in_features=model_config.backbone.kwargs.out_channels,
            out_features=model_config.growsp.n_primitives,
            bias=False,
        )
        for param in classifier.parameters():
            param.requires_grad = False
        classifier = classifier.cuda()
        # Build loss from config
        loss = build_loss_from_config(model_config.loss).cuda()
        loss = loss.cuda()
        # Build optimizers and schedulers from config
        optimizer_1 = build_optimizer_from_config(
            model_config.pretrain_optimizer, base_model=model
        )
        optimizer_2 = build_optimizer_from_config(
            model_config.grow_optimizer, base_model=model
        )
        scheduler_1 = build_scheduler_from_config(
            model_config.pretrain_scheduler, optimizer=optimizer_1
        )
        scheduler_2 = build_scheduler_from_config(
            model_config.grow_scheduler, optimizer=optimizer_2
        )

        # Load from checkpoint if necessary
        epoch, state = 0, "pretrain"
        if args.load_ckpt is not None:
            # Load values from the given path
            ckpt_path = parse_load_ckpt(args, logger)
            eval = (
                args.mode is Mode.eval
                or args.mode is Mode.test
                or args.mode is Mode.eval_unlabeled
            )
            epoch, state = load_model(
                model, classifier, ckpt_path, model_config, eval, logger
            )
            # Load optimizers and schedulers
            load_optimizer(optimizer_1, "pretrain", ckpt_path, logger)
            load_scheduler(scheduler_1, "pretrain", ckpt_path, logger)
            load_optimizer(optimizer_2, "grow", ckpt_path, logger)
            is_growth_start = (state == "grow") and (epoch == 0)
            load_scheduler(scheduler_2, "grow", ckpt_path, logger, is_growth_start)
            if eval:
                action = "Evaluating"
            else:
                # Training for epoch has already been done, so we resume from the next epoch
                epoch += 1
                action = "Resuming training"
            # Load succesful
            print_log(
                f"Succesfully loaded model. {action} @ state '{state}'...",
                logger=logger,
            )
        # Train or evaluate
        if args.mode is Mode.train:
            # Build datasets from config
            train_loader = build_dataloader_from_config(
                dataset_config, mode=Mode.train, seed=args.seed
            )
            cluster_loader = build_dataloader_from_config(
                dataset_config, mode=Mode.cluster
            )
            eval_loader = build_dataloader_from_config(dataset_config, mode=Mode.eval)
            # Train growsp
            train(
                model_config,
                args,
                train_loader,
                cluster_loader,
                eval_loader,
                model,
                optimizer_1,
                optimizer_2,
                scheduler_1,
                scheduler_2,
                loss,
                state,
                epoch,
                classifier,
                logger,
            )
        elif args.mode is Mode.eval or args.mode is Mode.test:
            # Build dataset from config
            data_loader = build_dataloader_from_config(dataset_config, mode=Mode.eval)
            evaluate(
                model_config,
                dataset_config,
                data_loader,
                model,
                classifier,
                args.mode is Mode.test,
                False,
                args.visualize,
                args.vis_path,
                logger,
            )  # Evaluate model
        elif args.mode is Mode.eval_unlabeled:
            # Set list of ids to the plot id provided as argument
            dataset_config.IDS = [args.plot_id]
            data_loader = build_dataloader_from_config(dataset_config, mode=Mode.eval)
            # Evaluate model
            evaluate(
                model_config,
                dataset_config,
                data_loader,
                model,
                classifier,
                False,
                True,
                args.visualize,
                args.vis_path,
                logger,
            )
        else:
            raise ValueError(f"mode '{args.mode}' not recognized!")


if __name__ == "__main__":
    main()
