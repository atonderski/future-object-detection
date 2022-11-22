import random
from abc import ABC, abstractmethod
from typing import List, Tuple

from numpy.random import triangular

import torch
from torch import Tensor
from torchvision.transforms import functional as tvtf


class ImageRemap:
    def __call__(self, tensor):
        tensor = tensor.float() / 255
        return tensor


class JointTransform(ABC):
    @abstractmethod
    def __call__(
        self, images: Tensor, boxes: Tensor, classes: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Perform a joint transform."""


class JointNoOpTransform(JointTransform):
    def __call__(self, images, boxes, classes):
        return images, boxes, classes


class JointCompose:
    def __init__(self, transforms: List[JointTransform]):
        self.transforms = transforms

    def __call__(self, images, boxes, classes):
        for transform in self.transforms:
            images, boxes, classes = transform(images, boxes, classes)
        return images, boxes, classes


class JointResize(JointTransform):
    def __init__(
        self,
        size: Tuple[int, int],
        interpolation: tvtf.InterpolationMode = tvtf.InterpolationMode.BILINEAR,
    ):
        self._size = list(size)
        self._interpolation = interpolation

    def __call__(self, images, boxes, classes):
        old_h, old_w = images.size()[-2:]
        new_h, new_w = self._size

        images = tvtf.resize(images, size=self._size, interpolation=self._interpolation)

        h_scale = new_h / old_h
        w_scale = new_w / old_w
        scaling = torch.tensor([w_scale, h_scale, w_scale, h_scale])
        boxes = scaling * boxes

        return images, boxes, classes


class BaseCrop(JointTransform, ABC):
    @abstractmethod
    def _get_crop_param(self, image_h: int, image_w: int) -> Tuple[int, int, int, int]:
        """Compute the top left coordinates and size of the crop."""
        pass

    def _crop_boxes(self, boxes, width, height):
        boxes[:, 0].clamp_(0, width)
        boxes[:, 1].clamp_(0, height)
        boxes[:, 2].clamp_(0, width)
        boxes[:, 3].clamp_(0, height)
        return boxes

    def _remove_inactive(self, crop_width, crop_height, boxes, *args):
        # Is this really right?
        # mask = ~(
        #        (boxes[:, 0] >= crop_width)
        #        & (boxes[:, 1] >= crop_height)
        #        & (boxes[:, 2] <= 0)
        #        & (boxes[:, 3] <= 0)
        # )
        mask = (
            (boxes[:, 0] <= crop_width)
            & (boxes[:, 1] <= crop_height)
            & (boxes[:, 2] >= 0)
            & (boxes[:, 3] >= 0)
        )
        return boxes[mask], *(arg[mask] for arg in args)

    def __call__(self, images, boxes, classes):
        _, _, image_h, image_w = images.size()
        i, j, crop_h, crop_w = self._get_crop_param(image_h, image_w)
        images = tvtf.crop(images, i, j, crop_h, crop_w)
        boxes = boxes - torch.tensor([j, i, j, i])

        boxes, classes = self._remove_inactive(crop_w, crop_h, boxes, classes)
        boxes = self._crop_boxes(boxes, crop_w, crop_h)
        return images, boxes, classes


class JointCenterCrop(BaseCrop):
    def __init__(self, size):
        self.th = size[0]
        self.tw = size[1]

    def _get_crop_param(self, image_h, image_w):
        i = (image_h - self.th) // 2
        j = (image_w - self.tw) // 2
        return i, j, self.th, self.tw


class JointRandomCrop(JointCenterCrop):
    def _get_crop_param(self, image_h, image_w):
        i = torch.randint(0, image_h - self.th + 1, size=(1,)).item()
        j = torch.randint(0, image_w - self.tw + 1, size=(1,)).item()
        return i, j, self.th, self.tw


class RandomSizedCrop(BaseCrop):
    def __init__(self, min_scale, max_scale):
        self._min_scale = min_scale
        self._max_scale = max_scale
        assert max_scale <= 1.0, "Cannot crop more than the whole image!"

    def _get_crop_param(self, image_h, image_w):
        scale = random.uniform(self._min_scale, self._max_scale)
        crop_h = int(image_h * scale)
        crop_w = int(image_w * scale)
        i = torch.randint(0, image_h - crop_h + 1, size=(1,)).item()
        j = torch.randint(0, image_w - crop_w + 1, size=(1,)).item()
        return i, j, crop_h, crop_w


class CenterBiasedRandomSizedCrop(RandomSizedCrop):
    def _get_crop_param(self, image_h, image_w):
        scale = random.uniform(self._min_scale, self._max_scale)
        crop_h = int(image_h * scale)
        crop_w = int(image_w * scale)

        max_i = image_h - crop_h + 1
        max_j = image_w - crop_w + 1

        i = int(triangular(0, max_i / 2, max_i, size=(1,)).item())
        j = int(triangular(0, max_j / 2, max_j, size=(1,)).item())
        print(i, j)
        return i, j, crop_h, crop_w


class JointHorizontalFlip(JointTransform):
    def __init__(self, p: float = 0.5):
        self._p = p

    def __call__(self, images, boxes, classes):
        if random.random() < self._p:
            images = tvtf.hflip(images)
            w = images.size()[-1]
            boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor(
                [w, 0, w, 0]
            )
        return images, boxes, classes


class RandomSelect:
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, *args, **kwargs):
        if random.random() < self.p:
            return self.transforms1(*args, **kwargs)
        return self.transforms2(*args, **kwargs)


class SizeFilter(JointTransform):
    """Filter objects based on size (relative to image size)."""

    def __init__(self, min_size):
        self.min_size = min_size

    def __call__(self, images, boxes, classes):
        _, _, image_h, image_w = images.size()
        tot_size = image_h * image_w
        box_sizes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        mask = (box_sizes / tot_size) > self.min_size
        return images, boxes[mask], classes[mask]
