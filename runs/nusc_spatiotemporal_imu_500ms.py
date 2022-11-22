import argparse
import os

import torch

from future_od.datasets import nu_scenes
from future_od.models.st_detr import SpatioTemporalDETRArgs
from future_od.utils.distributed import init_distributed_and_device_

from config import config
from runs._helper import build_base_parser, get_lr_func, get_trainer, setup_optimizer
from runs._loader import get_nusc_loaders
from runs._model import build_model


def train(model, args, detr_args):
    lr_sched, optimizer = setup_optimizer(detr_args, model, get_lr_func(args.epochs))
    print("starting dataset loading...")
    train_loader, val_loaders = get_nusc_loaders(
        (448, 800),
        offsets=[-1.0, -0.5, 0],
        config=config,
        args=args,
        train_batch_size=32,
    )
    trainer = get_trainer(
        args, config, detr_args, lr_sched, model, optimizer, train_loader, val_loaders
    )

    print(f"Starting first training stage")
    trainer.train(int(args.epochs * 0.60))

    print(f"Starting second training stage")
    trainer._train_loader, trainer._val_loaders = get_nusc_loaders(
        (896, 1600),
        offsets=[-1.0, -0.5, 0],
        config=config,
        args=args,
        train_batch_size=16,
    )
    trainer.train(args.epochs)


def main():
    print(
        "Started script: {}, with pytorch {}".format(os.path.basename(__file__), torch.__version__)
    )

    parser = build_base_parser()
    parser.add_argument("--epochs", default=160, type=int, help="Number of training epochs")
    args = parser.parse_args()
    args.experiment_idf = os.path.splitext(os.path.basename(__file__))[0]
    detr_args = SpatioTemporalDETRArgs(
        num_classes=len(nu_scenes.CATEGORY_DICT),
        num_queries=128,
        lr_backbone=1e-4,
    )

    init_distributed_and_device_(args)
    model = build_model(args, detr_args)
    print("built model")
    train(model, args, detr_args)


if __name__ == "__main__":
    main()
