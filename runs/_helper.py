import argparse
import os

import numpy as np

import torch
from torch import optim
from torch.utils.data import ConcatDataset

from future_od.datasets import nu_images, nu_scenes
from future_od.trainer import Trainer
from future_od.utils.wandb import WandBConfig


def get_trainer(
    args,
    config,
    detr_args,
    lr_sched,
    model,
    optimizer,
    train_loader,
    val_loaders,
):
    lookup_dataset = train_loader.dataset
    if isinstance(lookup_dataset, ConcatDataset):
        lookup_dataset = lookup_dataset.datasets[0]
    if isinstance(lookup_dataset, nu_scenes.NuScenesDataset):
        category_dict = nu_scenes.CATEGORY_DICT
    elif isinstance(lookup_dataset, nu_images.NuImagesDataset):
        category_dict = nu_images.CATEGORY_DICT
    else:
        raise ValueError(f"Unknown dataset: {lookup_dataset}")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        lr_sched=lr_sched,
        train_loader=train_loader,
        val_loaders=val_loaders,
        checkpoint_path=config["checkpoint_path"],
        visualization_path=os.path.join(config["visualization_path"], args.experiment_idf),
        save_name=args.experiment_idf,
        device=args.device,
        checkpoint_epochs=not args.no_checkpoints,
        print_interval=25,
        visualization_epochs=set([int(i) for i in np.linspace(1, args.epochs, 10)]),
        visualization_iterations=[0],
        distributed=args.distributed,
        is_master=(args.world_rank == 0),
        wandb_config=WandBConfig(
            enabled=(not args.disable_wandb),
            name=args.experiment_idf + getattr(args, "wandb_suffix", ""),
            notes="",
            num_images=32,
            hyperparams={
                "slurm-id": os.environ.get("SLURM_JOB_ID"),
                "epochs": args.epochs,
            },
            resume_id=args.wandb_resume_id,
        ),
        max_norm=detr_args.max_norm,
        category_dict=category_dict,
    )
    if not args.restart:
        trainer.load_checkpoint(args.checkpoint, getattr(args, "load_only_net", False))
    return trainer


def get_lr_func(epochs):
    warmup = int(0.1 * epochs)
    drop_1 = int(0.6 * epochs)
    drop_2 = int(0.9 * epochs)
    return (
        lambda e: (e + 1) / (1 + warmup)
        if e < warmup
        else 1
        if e <= drop_1
        else 0.5
        if e <= drop_2
        else 0.1
    )


def setup_optimizer(detr_args, model, lr_func):
    model_without_ddp = (
        model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    )
    param_dicts = [
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": detr_args.lr_backbone,
        },
    ]
    optimizer = optim.AdamW(param_dicts, lr=detr_args.lr, weight_decay=detr_args.weight_decay)
    lr_sched = optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    return lr_sched, optimizer


def add_pytorch_args(parser):
    parser.add_argument(
        "-d",
        "--device",
        dest="device",
        help="Device to run on, the cpu or gpu.",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help=(
            "Use multi-processing distributed training to launch "
            "N processes per node, which has N GPUs. This is the "
            "fastest way to use PyTorch for either single node or "
            "multi node data parallel training"
        ),
    )
    parser.add_argument("--local_rank", default=0, type=int, help="number of distributed processes")
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""",
    )


def build_base_parser():
    parser = argparse.ArgumentParser(
        description="Experiment runfile, you run experiments from this file"
    )
    parser.add_argument("--restart", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--disable_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_resume_id", default=None)
    parser.add_argument("--no_checkpoints", action="store_true", default=False)
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint to be loaded")
    parser.add_argument("--short_train", action="store_true", default=False)
    parser.add_argument("--night", action="store_true", default=False)
    parser.add_argument("--load-only-net", action="store_true", default=False)
    add_pytorch_args(parser)
    return parser
