import os

import wandb

import torch
import torchvision as tv


COLOURS = torch.stack(
    [
        torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])[None, None, :].expand(5, 5, -1),
        torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])[None, :, None].expand(5, -1, 5),
        torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])[:, None, None].expand(-1, 5, 5),
    ],
    dim=3,
).view(-1, 3)


def revert_imagenet_normalization(sample):
    """
    sample (Tensor): of size (nsamples,nchannels,height,width)
    """
    # Imagenet mean and std
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean_tensor = torch.Tensor(mean).view(3, 1, 1).to(sample.device)
    std_tensor = torch.Tensor(std).view(3, 1, 1).to(sample.device)
    non_normalized_sample = sample * std_tensor + mean_tensor
    return non_normalized_sample


def draw_boxes(image, boxes, colours, thickness=3):
    _, H, W = image.size()
    N, _ = boxes.size()
    for n in range(N):
        left = int(boxes[n, 0].clamp(thickness, W - thickness))
        bottom = int(boxes[n, 3].clamp(thickness, H - thickness))
        right = int(boxes[n, 2].clamp(thickness, W - thickness))
        top = int(boxes[n, 1].clamp(thickness, H - thickness))
        image[:, top - thickness : bottom, left - thickness : left] = colours[n].view(3, 1, 1)
        image[:, bottom : bottom + thickness, left - thickness : right] = colours[n].view(3, 1, 1)
        image[:, top : bottom + thickness, right : right + thickness] = colours[n].view(3, 1, 1)
        image[:, top - thickness : top, left : right + thickness] = colours[n].view(3, 1, 1)
    return image


def visualize(image, classes, boxes, fpath, BACKGROUND_CLASS):
    """
    Args:
        video (Tensor): Of size (3, H, W)
        classes (Tensor or LongTensor): Of size (M, C) if Tensor or (M,) if LongTensor. 0 if background
        boxes (Tensor): Of size (M, 4), encoded as (x1, y1, x2, y2)
        fpath (str): Path where visualization is saved
    """
    print("Storing visualization at", fpath)
    vis = revert_imagenet_normalization(image)

    if boxes is not None:
        if isinstance(classes, (torch.FloatTensor, torch.cuda.FloatTensor)):  # We get logits
            scores, classes = classes.max(dim=1)
            classes[scores < 0.5] = BACKGROUND_CLASS
        boxes = boxes[classes != BACKGROUND_CLASS]
        colours = COLOURS[classes[classes != BACKGROUND_CLASS]]
        vis = draw_boxes(vis, boxes, colours)

    if not os.path.exists(os.path.dirname(fpath)):
        os.makedirs(os.path.dirname(fpath))

    vis = vis * 255
    tv.io.write_png(vis.byte(), fpath)
    return vis


def visualize_wandb(
    image,
    background_class,
    category_dict,
    pred_scores=None,
    pred_boxes=None,
    anno_classes=None,
    anno_boxes=None,
    ignore_boxes=None,
    model_mood=None,
):
    image = revert_imagenet_normalization(image)
    boxes = {}
    if pred_boxes is not None:
        pred_scores, pred_classes = pred_scores.max(dim=1)
        boxes["predictions"] = {
            "box_data": [
                {
                    "position": {
                        "minX": int(box[0].round()),
                        "maxX": int(box[2].round()),
                        "minY": int(box[1].round()),
                        "maxY": int(box[3].round()),
                    },
                    "domain": "pixel",
                    "class_id": int(cls),
                    "scores": {
                        "score": float(score),
                    },
                    "box_caption": f"{idx} {category_dict[int(cls)]}",
                }
                for idx, score, cls, box in zip(
                    range(pred_scores.size(0)),
                    pred_scores.numpy(),
                    pred_classes.numpy(),
                    pred_boxes.numpy(),
                )
                if score > 0.1
            ],
            "class_labels": category_dict,
        }
    if anno_classes is not None:
        boxes["ground_truth"] = {
            "box_data": [
                {
                    "position": {
                        "minX": int(box[0].round()),
                        "maxX": int(box[2].round()),
                        "minY": int(box[1].round()),
                        "maxY": int(box[3].round()),
                    },
                    "domain": "pixel",
                    "class_id": int(cls),
                }
                for cls, box in zip(anno_classes.numpy(), anno_boxes.numpy())
                if cls != background_class
            ],
            "class_labels": category_dict,
        }
    if ignore_boxes is not None:
        boxes["ignore_regions"] = {
            "box_data": [
                {
                    "position": {
                        "minX": int(box[0].round()),
                        "maxX": int(box[2].round()),
                        "minY": int(box[1].round()),
                        "maxY": int(box[3].round()),
                    },
                    "domain": "pixel",
                    "class_id": 0,
                }
                for box in ignore_boxes.numpy()
            ],
            "class_labels": {0: "Ignore"},
        }
    return wandb.Image(image.numpy().transpose(1, 2, 0), boxes=boxes, caption=model_mood)
