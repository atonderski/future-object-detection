from typing import Dict, Iterable, Tuple, Union

import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

import future_od.datasets.transforms as T
from future_od.datasets import nu_images, nu_scenes


def get_nuim_loaders(
    img_size: Tuple[int, int],
    offsets: Union[Iterable[int], Dict[str, Iterable[int]]],
    args,
    config,
    train_batch_size: int,
    random_aug=T.RandomSizedCrop(0.5, 1.0),
    val_annotated_frame_override=None,
) -> Tuple[DataLoader, Dict[str, DataLoader]]:
    """Construct data loaders."""
    if isinstance(offsets, dict):
        assert "train" in offsets and "val" in offsets
        train_offsets, val_offsets = offsets["train"], offsets["val"]
    else:
        train_offsets, val_offsets = offsets, offsets
    training_data = nu_images.NuImagesDataset(
        root_path=config["nuimages_path"],
        split="mini" if args.debug or args.short_train else "train",
        night=args.night,
        front_camera_only=True,
        joint_transform=T.JointCompose(
            [
                random_aug,
                T.JointResize(size=img_size),
            ]
        ),
        frames=[nu_images.ANNOTATED_FRAME + offset for offset in train_offsets],
    )
    print("Loaded training set with", len(training_data), "samples")
    validation_data = nu_images.NuImagesDataset(
        root_path=config["nuimages_path"],
        split="mini" if args.debug else "val",
        night=args.night,
        front_camera_only=True,
        max_frame_random_offset=0,
        joint_transform=T.JointCompose([T.JointCenterCrop(size=img_size)]),
        frames=[nu_images.ANNOTATED_FRAME + offset for offset in val_offsets],
        annotated_frame_idx_override=val_annotated_frame_override,
    )
    print("Loaded validation set with", len(validation_data), "samples")
    return _build_loaders(args, train_batch_size, training_data, validation_data)


def get_nusc_loaders(
    img_size: Tuple[int, int],
    offsets: Union[Iterable[Union[str, float]], Dict[str, Iterable[Union[str, float]]]],
    args,
    config,
    train_batch_size: int,
    random_aug=T.RandomSizedCrop(0.5, 1.0),
    val_annotated_frame_override=None,
    filter_offsets=None,
) -> Tuple[DataLoader, Dict[str, DataLoader]]:
    """Construct data loaders."""
    if isinstance(offsets, dict):
        assert "train" in offsets and "val" in offsets
        train_offsets, val_offsets = offsets["train"], offsets["val"]
    else:
        train_offsets, val_offsets = offsets, offsets
    training_data = nu_scenes.NuScenesDataset(
        root_path=config["nuscenes_path"],
        split="mini_train" if args.debug or args.short_train else "train",
        night=args.night,
        front_camera_only=True,
        joint_transform=T.JointCompose(
            [
                random_aug,
                T.JointResize(size=img_size),
            ]
        ),
        frame_offsets=train_offsets,
        filter_offsets=filter_offsets,
    )
    print("Loaded training set with", len(training_data), "samples")
    validation_data = nu_scenes.NuScenesDataset(
        root_path=config["nuscenes_path"],
        split="mini_val" if args.debug else "val",
        night=args.night,
        front_camera_only=True,
        joint_transform=T.JointCompose([T.JointCenterCrop(size=img_size)]),
        frame_offsets=val_offsets,
        annotated_frame_idx_override=val_annotated_frame_override,
        filter_offsets=filter_offsets,
    )
    print("Loaded validation set with", len(validation_data), "samples")
    return _build_loaders(args, train_batch_size, training_data, validation_data)


def _build_loaders(args, train_batch_size, training_data, validation_data):
    if args.distributed:
        sampler_train = DistributedSampler(training_data)
        sampler_val = DistributedSampler(validation_data, seed=9069788369656784)
    else:
        sampler_train = RandomSampler(training_data)
        generator = torch.Generator().manual_seed(9069788369656784)
        sampler_val = RandomSampler(validation_data, generator=generator)
    num_workers = getattr(args, "num_workers", 16)
    training_loader = DataLoader(
        training_data,
        sampler=sampler_train,
        batch_size=min(2, train_batch_size)
        if (args.debug or args.short_train)
        else train_batch_size // args.world_size,
        num_workers=num_workers,
        drop_last=True,
    )
    validation_loader = {
        "val0": DataLoader(
            validation_data,
            sampler=sampler_val,
            batch_size=2 if args.debug else 12,
            num_workers=num_workers,
        ),
    }
    return training_loader, validation_loader
