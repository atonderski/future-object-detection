import argparse
import os

import torch

from future_od.datasets import nu_scenes
from future_od.models.st_detr import SpatioTemporalDETRArgs
from future_od.utils.distributed import init_distributed_and_device_

from config import config
from runs._helper import add_pytorch_args, get_trainer
from runs._loader import get_nusc_loaders
from runs._model import build_model
from runs.eval.helpers import add_hardcoded_eval_args


def train(model, args, detr_args):
    print("starting dataset loading...")
    train_loader, val_loaders = get_nusc_loaders(
        (896, 1600),
        offsets=[-0.5, -0.25, 0],
        config=config,
        args=args,
        train_batch_size=8,
        filter_offsets=[-0.5, -0.25, 0],
    )
    print("Running eval")
    trainer = get_trainer(args, config, detr_args, None, model, None, train_loader, val_loaders)
    trainer.eval()


def main():
    print(
        "Started script: {}, with pytorch {}".format(os.path.basename(__file__), torch.__version__)
    )

    parser = argparse.ArgumentParser(
        description="Experiment runfile, you run experiments from this file"
    )
    parser.add_argument("--disable_wandb", action="store_true", default=False)
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint to be loaded")
    parser.add_argument("--night", action="store_true", default=False)
    add_pytorch_args(parser)
    args = parser.parse_args()
    add_hardcoded_eval_args(args, "w6_nusc_250ms_attendprev_decoder")
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
