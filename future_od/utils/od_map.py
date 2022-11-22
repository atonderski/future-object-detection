import numpy as np

import torch
import torch.nn.functional as F


"""In this module, we use capital letters to denote dimensionalities.
Dims:
    B is batch size
    C is the number of object classes, including an any-object category
    S is the number of size categories, including an any-size category
    T is the number of iou-thresholds
    M' is the number of detection slots in the predictions
    M is a reduced number of detection slots used to compute performance metrics
    N' is the number of annotation slots
    N is a reduced number of evaluation slots
"""

# SIZE_CATEGORY_DELIMITERS = [(H/16) * (W/16), (H * (3/16)) * (W * (3/16)))]
SIZE_CATEGORY_DELIMITERS = [
    (1 / 24) * (1 / 64),
    (1 / 4) * (1 / 12),
]  # Will be multiplied by H and W


def _cut_annotation_tensor(anno_boxes, anno_classes, anno_active):
    """For efficiency, we cut the dense annotation tensor, removing annotation slots that are
    not used in any of the batch-samples.
    Args:
        anno_boxes   (Tensor)     (*, N', 4):
        anno_classes (LongTensor) (*, N')   :
        anno_active  (LongTensor) (*, N')   :
    Returns:
        anno_boxes   (Tensor)     (*, N, 4):
        anno_classes (LongTensor) (*, N)   :
        anno_active  (LongTensor) (*, N)   :
    """
    anno_mask = anno_active.any(dim=0) == 1  # (N',)
    anno_mask[0] = True  # Code crashes if we remove all elements. TODO: Make this a bit nicer
    anno_boxes = anno_boxes[:, anno_mask, :]  # (B, N, 4)
    anno_classes = anno_classes[:, anno_mask]  # (B, N)
    anno_active = anno_active[:, anno_mask]  # (B, N)
    return anno_boxes, anno_classes, anno_active


def get_batch_many_to_many_box_iou(boxes_one, boxes_two):
    assert boxes_one.dim() == 3
    assert boxes_two.dim() == 3
    assert boxes_one.size(0) == boxes_two.size(0), (boxes_one.size(), boxes_two.size())
    assert boxes_one.size(2) == 4
    assert boxes_two.size(2) == 4
    B, M, _ = boxes_one.size()
    _, N, _ = boxes_two.size()
    boxes_one = boxes_one.view(B, M, 1, 4)
    boxes_two = boxes_two.view(B, 1, N, 4)
    area1 = F.relu(boxes_one[:, :, :, 2] - boxes_one[:, :, :, 0]) * F.relu(
        boxes_one[:, :, :, 3] - boxes_one[:, :, :, 1]
    )
    area2 = F.relu(boxes_two[:, :, :, 2] - boxes_two[:, :, :, 0]) * F.relu(
        boxes_two[:, :, :, 3] - boxes_two[:, :, :, 1]
    )
    intersection = F.relu(
        torch.min(boxes_one[:, :, :, 2], boxes_two[:, :, :, 2])
        - torch.max(boxes_one[:, :, :, 0], boxes_two[:, :, :, 0])
    ) * F.relu(
        torch.min(boxes_one[:, :, :, 3], boxes_two[:, :, :, 3])
        - torch.max(boxes_one[:, :, :, 1], boxes_two[:, :, :, 1])
    )
    iou = (intersection + 1e-7) / (area1 + area2 - intersection + 1e-7)
    return iou


def _get_iou(pred_boxes, anno_boxes):
    """Returns the IoU between each predicted box and each annotation box. Note that in our case,
    each predicted box is C detections. Further, some annotation boxes are inactive and just
    padding elements.
    Args:
        pred_boxes (Tensor)     (B, M', 4):
        anno_boxes (Tensor)     (B, N, 4):
    Returns:
        iou (Tensor) (B, M', N)
    """
    M, N = pred_boxes.size(1), anno_boxes.size(1)
    return get_batch_many_to_many_box_iou(
        pred_boxes.view(-1, M, 4), anno_boxes.view(-1, N, 4)
    )  # (B, M, N)


def _get_perclass_topk_predictions(pred_class_scores, K):
    """Return the topK predictions for each class.
    Sort confidences and select the top 100 for each class
        Args:
            pred_class_scores    (Tensor) (B, M', C):
        Returns:
            confs     (Tensor)     (B, M, 3C): The topm confidences per class
            ordered_m (LongTensor) (B, M, 3C): Indices the orig. det. tensor, matching the topm confs
    """
    B, M, C = pred_class_scores.size()
    confs = pred_class_scores.view(-1, M, C).detach()
    confs, ordered_m = confs.sort(dim=1, descending=True)  # (B, M, C)
    confs = confs[:, :K, :]
    ordered_m = ordered_m[:, :K, :]
    return confs, ordered_m


def _get_anno_available_mask(anno_active, anno_classes, C):
    """We compare predictions and annotations per-class. An annotation is only assigned to a
    prediction if their class matches. To facilitate comparisons, we scatter the annotation class
    tensor over a new class-dimension, similar to the predictions. The last class slot corresponds
    to a generate detection, a detection corresponding to any class. Further, inactive detections
    are masked out.
    Args:
        anno_active   (LongTensor) (B, N)
        anno_classes  (LongTensor) (B, N)
    Returns:
        anno_available_mask (BoolTensor) (B, C, N): The shape is designed to be used on the IoU
    """
    B, N = anno_classes.size()
    device = anno_active.device
    active_mask = anno_active[:, None, :] == 1  # (B, 1, N)
    class_mask = torch.cat(
        [
            anno_classes[:, None, :]
            == torch.arange(C - 1, dtype=torch.long, device=device)[None, :, None],
            torch.ones((B, 1, N), dtype=torch.bool, device=device),
        ],
        dim=1,
    )  # (B, C, N)
    anno_available_mask = active_mask * class_mask  # (B, C, N)
    return anno_available_mask


def _get_perclass_topk_iou(iou, ordered_m, anno_available_mask):
    """
    Args:
        iou        (Tensor)     (B, M', N):
        ordered_m  (LongTensor) (B, M, C):
    Returns:
        iou (Tensor) (B, M, C, N)
    """
    B, Mp, N = iou.size()
    _, M, C = ordered_m.size()
    iou = iou.view(B, Mp, 1, N).expand(-1, -1, C, -1)  # (B, M', C, N)
    ordered_m = ordered_m.view(B, M, C, 1).expand(-1, -1, -1, N)  # (B, M, C, N)
    iou = iou.gather(1, ordered_m)
    iou = iou.where(
        anno_available_mask[:, None, :, :].expand(-1, M, -1, -1), torch.zeros_like(iou)
    )  # (B, M, C, N)
    return iou


def _get_box_size_categories(boxes, imsize):
    """Takes a set of boxes and returns the size category of each box. The box area is compared to
    the height and width of the image, returning 0, 1, or 2. This approximately corresponds to the
    sizes in COCO, where the sizes are delineated at 32^2 and 96^2 pixels with an approximate
    image size of 512x512. Thus, the categories are divided by (H/16)*(W/16) and (3H/16)*(3W/16).
    Args:
        boxes (Tensor) (B, M or N, 4)
    Returns:
        sizecats (Tensor) (B, M or N, S)
    """
    B, M, _ = boxes.size()
    H, W = imsize
    box_areas = (boxes[:, :, 2] - boxes[:, :, 0]) * (boxes[:, :, 3] - boxes[:, :, 1])
    sizes = [size_delim * H * W for size_delim in SIZE_CATEGORY_DELIMITERS]
    mask_small = box_areas <= sizes[0]
    mask_medium = (sizes[0] < box_areas) * (box_areas <= sizes[1])
    mask_large = sizes[1] < box_areas
    mask_all = torch.ones_like(mask_small)
    size_categories = torch.stack([mask_all, mask_small, mask_medium, mask_large], dim=2)
    return size_categories


def _get_pred_size_categories(pred_boxes, ordered_m, imsize):
    """
    Args:
        pred_boxes (Tensor)     (B, M', 4)
        ordered_m  (LongTensor) (B, M, C)
        imsize     (tuple)                : Containing (height, width)
    Returns:
        size_categories (LongTensor) (B, M, C, S)
    """
    B, Mp, _ = pred_boxes.size()
    _, M, C = ordered_m.size()
    S = 4
    size_categories = _get_box_size_categories(pred_boxes, imsize)  # (B, M', S)
    size_categories = size_categories.view(B, Mp, 1, S).expand(-1, -1, C, -1)  # (B, M', C, S)
    ordered_m = ordered_m.view(B, M, C, 1).expand(-1, -1, -1, S)
    size_categories = size_categories.gather(1, ordered_m)  # (B, M, C, S)
    return size_categories


def _get_num_annos(anno_boxes, anno_available_mask, imsize):
    """
    Args:
        anno_boxes          (Tensor)     (B, N, 4)
        anno_available_mask (BoolTensor) (B, C, N)
        imsize              (tuple)               : Containing (height, width)
    Returns:
        num_annos (LongTensor) (C, S)
    """
    B, C, N = anno_available_mask.size()
    S = 4
    size_onehot = _get_box_size_categories(anno_boxes, imsize)  # (B, N, S)
    num_annos_per_size_category = (
        anno_available_mask.view(B, C, N, 1).expand(-1, -1, -1, S)
        * size_onehot.view(B, 1, N, S).expand(-1, C, -1, -1)
    ).sum(
        dim=(0, 2)
    )  # (C, S)
    return num_annos_per_size_category


def prepare_od_map_stuffs(
    pred_boxes,
    pred_class_scores,
    anno_boxes,
    anno_classes,
    anno_active,
    imsize,
):
    """Returns the intermediary results needed for AP calculation. AP is calculated for a range of
    IoU-thresholds and is later averaged to give mAP. This procedure is done per class and per
    size category.
    Args:
        pred_boxes        (Tensor)    : (*, M, 4)
        pred_class_scores (Tensor)    : (*, M, C)
        anno_boxes        (Tensor)    : (*, N, 4)
        anno_classes      (LongTensor): (*, N)
        anno_active       (LongTensor): (*, N)
        imsize            (tuple)     : containing (height, width)
        sun_elevation     (Tensor)    : (B,)
    Returns:
        confs       (Tensor)     (T, C, M * prod(*)) with values in [0, 1]
        is_positive (BoolTensor) (T, C, M * prod(*)) where M is the topm predictions per class
        active      (LongTensor) (C, S, M * prod(*))
        num_annos   (LongTensor) (C, S)
    """
    device = pred_boxes.device
    *sizes, M, C = pred_class_scores.size()
    N = anno_classes.size(-1)
    S = 4

    with torch.no_grad():
        anno_classes = anno_classes.view(-1, N).detach()
        anno_active = anno_active.view(-1, N).detach()
        B = anno_classes.size(0)
        thresholds = torch.arange(0.50, 1.00, 0.05, device=device)
        T = len(thresholds)

        anno_boxes, anno_classes, anno_active = _cut_annotation_tensor(
            anno_boxes, anno_classes, anno_active
        )
        N = anno_boxes.size(1)
        M = 50
        iou = _get_iou(pred_boxes, anno_boxes)  # (B, M', N)
        confs, ordered_m = _get_perclass_topk_predictions(
            pred_class_scores, M
        )  # (B, M, C), (B, M, C)
        anno_available_mask = _get_anno_available_mask(anno_active, anno_classes, C)  # (B, C, N)
        iou = _get_perclass_topk_iou(iou, ordered_m, anno_available_mask)  # (B, M, C, N)
        iou = iou.view(B, 1, M, C, N).repeat(1, T, 1, 1, 1)  # (B, T, M, C, N)

        is_positive = torch.zeros(
            (B, T, M, C), dtype=torch.bool, device=device
        )  # (B, nthresh, M, C)
        for m in range(M):
            best_score, best_n = iou[:, :, m].max(dim=3)  # (B, num_thresh, C)
            is_positive[:, :, m, :] = best_score >= thresholds[None, :, None]
            # iou[b, i, m, c, best_n[b, i, c]] = 0 if is_positive[b, i, m, c] else iou[b, i, m, c, n]
            # i.e., for claimed annotations, set all matching iou with them to 0 s.t. they are not used again
            new_iou = iou.scatter(
                4,
                best_n[:, :, None, :, None].expand(-1, -1, M, -1, -1),  # (B, T, M, C, 1)
                torch.zeros_like(iou),
            )
            iou = new_iou.where(is_positive[:, :, m, None, :, None].expand(-1, -1, M, -1, N), iou)

        confs = confs.reshape(B * M, C).permute([1, 0]).reshape(1, C, B * M).repeat(T, 1, 1)
        is_positive = is_positive.permute([1, 3, 0, 2]).reshape(T, C, B * M)
        # num_annos   = anno_available_mask.sum(dim=(0, 2)).view(1, C).repeat(T, 1)

        size_categories = _get_pred_size_categories(pred_boxes, ordered_m, imsize)  # (B, M, C, S)
        size_categories = size_categories.reshape(B * M, C, S).permute([1, 2, 0])
        num_annos = _get_num_annos(anno_boxes, anno_available_mask, imsize)

    return confs, is_positive, size_categories, num_annos


def _get_ap(confs, is_positive, size_categories, num_annos):
    """
    Args:
        confs           (Tensor)      (C, num_objects)
        is_positive     (BoolTensor)  (C, num_objects)
        size_categories (BoolTensor)  (C, S, num_objects)
        num_annos       (LongTensor)  (C, S, num_iter)
    Returns
        ap (Tensor)  (C, S)
    """
    C, S, M = size_categories.size()
    ids = confs.argsort(dim=1, descending=True)
    ids = ids.view(C, 1, M).expand(-1, S, -1)
    is_positive = is_positive.view(C, 1, M) * size_categories

    # Get the parts we need, sorted by confidence
    is_positive = is_positive.gather(2, ids)
    size_categories = size_categories.gather(2, ids)
    num_annos = num_annos.sum(dim=2)

    # Aggregate the AP
    precision = is_positive.cumsum(dim=2) / (size_categories.cumsum(dim=2) + 1e-5)
    ap = (precision * is_positive).sum(dim=2) / num_annos

    return ap


def aggregate_mean_average_precision(confs, is_positive, size_categories, num_annos, device):
    """Takes stacked intermediaries from different iterations and predicts various forms of AP

    T is the number of thresholds {50, 55, ..., 95}
    C is the number of classes, including generic
    S is the number of sizes

    Args:
        confs           (Tensor)      (T, C, num_objects)
        is_positive     (BoolTensor)  (T, C, num_objects)
        size_categories (LongTensor)  (C, S, num_objects)
        num_annos       (LongTensor)  (C, S, num_iter)
        device (torch.Device)
    Returns:
        Tensor: (T, C, S)
    """
    T, C, N = confs.size()
    S = 4

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    with torch.no_grad():
        ap = torch.zeros((T, C, S), device=device)
        for threshold_idx in range(T):
            ap[threshold_idx] = _get_ap(
                confs[threshold_idx].to(device),
                is_positive[threshold_idx].to(device),
                size_categories.to(device),
                num_annos.to(device),
            )

    t1.record()
    torch.cuda.synchronize()
    print(f"AP aggregation took {t0.elapsed_time(t1):.3f} ms.")
    print(f"Number of annos for each class and size category is:")
    print(num_annos.sum(dim=2))

    ap = ap.cpu()
    ap = {
        "all": ap[:, 0:-1, :],
        "classavg": np.nanmean(ap[:, 0:-1, :], axis=1),
        "threshavg": np.nanmean(ap[:, 0:-1, :], axis=0),
        "classavg threshavg": np.nanmean(ap[:, 0:-1, :], axis=(0, 1)),
        "generic": ap[:, -1, :],
        "generic threshavg": np.nanmean(ap[:, -1, :], axis=0),
    }
    return ap
