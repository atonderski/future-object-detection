import numpy as np

import torch


def specify_rank0_first_batch(data, first_ids, world_size):
    remaining_ids = list(set(range(len(data))) - set(first_ids))
    ids = []
    i = 0
    for first_idx in first_ids:
        ids = ids + [first_idx]
        ids = ids + remaining_ids[i * (world_size - 1) : (i + 1) * (world_size - 1)]
        i = i + 1
    ids = ids + remaining_ids[i * (world_size - 1) :]
    data = torch.utils.data.Subset(data, indices=ids)
    return data


def construct_box_targets(boxes, classes, max_num_objects, ignore_categories=None):
    # Separate ignore objects from non-ignore objects
    if ignore_categories:
        ignore_mask = sum(classes == ignore_class for ignore_class in ignore_categories) > 0
    else:
        ignore_mask = torch.zeros_like(classes, dtype=torch.bool)
    obj_classes = classes[~ignore_mask]
    obj_boxes = boxes[~ignore_mask]
    ignore_boxes = boxes[ignore_mask]

    # Construct dense targets
    boxes = torch.zeros((max_num_objects, 4))
    ignores = torch.zeros((max_num_objects, 4))
    classes = torch.zeros((max_num_objects,), dtype=torch.int64)
    active = torch.zeros((max_num_objects,), dtype=torch.int64)
    boxes[: len(obj_boxes)] = obj_boxes[:max_num_objects]
    ignores[: len(ignore_boxes)] = ignore_boxes[:max_num_objects]
    classes[: len(obj_classes)] = obj_classes[:max_num_objects]
    active[: len(obj_classes)] = 1
    return boxes, classes, ignores, active


def concat_quaternion(q1, q2):
    """Returns the composed rotation of q1 and q2, applied in sequence (first q1, then q2). The
    input are two rotation quaternions, representing two different rotations.
    Args:
        q1 (Tensor): Of size (*, 4). Unit quaternion
        q2 (Tensor): Of size (*, 4). Unit quaternion
    Returns:
        Tensor: Of size (*, 4). Unit quaternion
    """
    orig_size = q1.size()
    assert q2.size() == orig_size
    assert orig_size[-1] == 4
    q1 = q1.view(-1, 4)
    q2 = q2.view(-1, 4)

    a1 = q1[:, 0:1]
    a2 = q2[:, 0:1]
    v1 = q1[:, 1:4]
    v2 = q2[:, 1:4]
    scalar = a1 * a2 - torch.einsum("mn,mn->m", v1, v2)[:, None]
    vector = a1 * v2 + a2 * v1 + torch.cross(v1, v2, dim=1)
    result = torch.cat([scalar, vector], dim=1)
    return result.view(*orig_size)


def inverse_quaternion(q):
    """Returns the inverse of a unit quaternion q
    Args:
        q (Tensor): Of size (*, 4). Unit quaternion
    Returns
        Tensor: Of size (*, 4). Unit quaternion
    """
    return torch.cat([q[..., 0:1], -q[..., 1:4]], -1)
