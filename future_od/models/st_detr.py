from dataclasses import dataclass

import torch
import torch.nn as nn

from future_od.models.set_criterion import SetCriterion
from future_od.utils.od_map import prepare_od_map_stuffs

from ConditionalDETR.models.matcher import build_matcher


@dataclass
class SpatioTemporalDETRArgs:
    # General settings
    num_classes: int
    masks: bool = False

    # Optimization
    lr_backbone: float = 1e-5
    lr: float = 1e-4
    weight_decay: float = 1e-4
    max_norm: float = 0.1

    # Backbone
    backbone: str = "resnet50"
    dilation: bool = False
    position_embedding: str = "sine"
    pretrained_backbone: bool = True

    # Transformer settings
    enc_layers: int = 6
    dec_layers: int = 6
    dim_feedforward: int = 2048
    hidden_dim: int = 256
    dropout: float = 0.1
    enc_nheads: int = 8
    nheads: int = 8
    num_queries: int = 300
    pre_norm: bool = False

    # Matcher settings
    set_cost_class: float = 2.0
    set_cost_bbox: float = 5.0
    set_cost_giou: float = 2.0

    # Loss settings
    aux_loss: bool = True
    cls_loss_coef: float = 2.0
    bbox_loss_coef: float = 5.0
    giou_loss_coef: float = 2.0
    focal_alpha: float = 0.25

    # Data settings
    no_imu_speed: bool = False
    encode_offset: bool = False


class SpatioTemporalDETR(nn.Module):
    """This is a general spatio-temporal extension of DETR, specifically based on ConditionalDETR."""

    def __init__(self, args: SpatioTemporalDETRArgs, model, loss_matching_mode="per level"):
        super().__init__()

        self._model = model
        matcher = build_matcher(args)

        weight_dict = {
            "loss_ce": args.cls_loss_coef,
            "loss_bbox": args.bbox_loss_coef,
            "loss_giou": args.giou_loss_coef,
        }
        # TODO this is a hack
        if args.aux_loss:
            aux_weight_dict = {}
            for i in range(args.dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes", "cardinality"]
        self._criterion = SetCriterion(
            args.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            focal_alpha=args.focal_alpha,
            losses=losses,
            matching_mode=loss_matching_mode,
        )
        self._imu_keys = ["translation", "acceleration", "rotation", "rotation_rate"]
        if not args.no_imu_speed:
            self._imu_keys.append("speed")

        self._encode_offset = args.encode_offset

    @staticmethod
    def get_stat_idfs():
        return ["labels", "box_l1", "box_giou", "cardinality", "class_error"]

    def forward(self, data=None, visualize=False, epoch=None, distributed=False):
        """
        Args:
            data (dict):
            visualize (bool): Used for debugging if there is something internal to be visualized.
            epoch (int): Cant remember the point of this one
        Returns
            dict:
                'class_scores': Tensor of shape (B, M, C), containing categorical logits (scores)
                'boxes': Tensor of shape (B, M, 4), containing non-normalized bboxes on xyxy-form
            Tensor: Scalar representing total loss
            dict
        """
        images = data["video"]
        B, L, _, H, W = images.size()

        kwargs = {}
        if data.get("translation") is not None:
            kwargs["imu"] = torch.cat([data[imu_key] for imu_key in self._imu_keys], dim=2)
        if self._encode_offset:
            kwargs["temporal_offsets"] = data["temporal_offsets"]

        outputs, model_moods = self._model(images, **kwargs)
        # TODO: Properly handle annotated frame idx if they are different for different batch elements
        if isinstance(outputs, list):  # TODO This is always true for CausalCore
            pred_logits = torch.stack([output["pred_logits"] for output in outputs], dim=1)
            pred_boxes = torch.stack([output["pred_boxes"] for output in outputs], dim=1)
            num_decoder_layers = len(outputs[0]["aux_outputs"])
            annotated_frame_output = {
                "pred_logits": pred_logits[range(B), data["annotated_frame_idx"]],
                "pred_boxes": pred_boxes[range(B), data["annotated_frame_idx"]],
                "aux_outputs": [
                    {
                        "pred_logits": torch.stack(
                            [
                                outputs[data["annotated_frame_idx"][b]]["aux_outputs"][
                                    decoder_layer_idx
                                ]["pred_logits"][b]
                                for b in range(B)
                            ]
                        ),
                        "pred_boxes": torch.stack(
                            [
                                outputs[data["annotated_frame_idx"][b]]["aux_outputs"][
                                    decoder_layer_idx
                                ]["pred_boxes"][b]
                                for b in range(B)
                            ]
                        ),
                    }
                    for decoder_layer_idx in range(num_decoder_layers)
                ],
            }
        elif "pred_logits" in outputs and outputs["pred_logits"].dim() == 3:
            pred_logits = outputs["pred_logits"][:, None]
            pred_boxes = outputs["pred_boxes"][:, None]
            annotated_frame_output = outputs
        else:
            raise ValueError("cannot interpret output on the format: %s" % outputs)
            # outputs = outputs[data["annotated_frame_idx"][0]]
        # assert "pred_boxes" in outputs
        # assert "pred_logits" in outputs

        loss, stats = self.loss(data, annotated_frame_output, distributed)

        od_map_stuffs, outputs = self.post_proc(pred_logits, pred_boxes, data, images)
        outputs["moods"] = model_moods

        state = None
        return outputs, state, loss, stats, od_map_stuffs

    def loss(self, data, outputs, distributed):
        H, W = data["video"].size()[-2:]
        targets = to_detr_targets(
            H=H,
            W=W,
            anno_active=data["active"],
            anno_boxes=data["boxes"],
            anno_classes=data["classes"],
        )
        loss_dict = self._criterion(outputs, targets, distributed)
        weight_dict = self._criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        stats = {
            "labels": loss_dict["loss_ce"] * weight_dict["loss_ce"],
            "box_l1": loss_dict["loss_bbox"] * weight_dict["loss_bbox"],
            "box_giou": loss_dict["loss_giou"] * weight_dict["loss_giou"],
            "cardinality": loss_dict["cardinality_error"],
            "class_error": loss_dict["class_error"],
        }
        return loss, stats

    def post_proc(self, class_scores, boxes, data, images):
        """
        Args:
            class_scores (Tensor): Of size (B, L, M, C)
            boxes (Tensor)       : Of size (B, L, M, 4)
        """
        # boxes, class_scores = outputs["pred_boxes"], outputs["pred_logits"]
        B, L, _, H, W = images.size()
        class_scores = torch.sigmoid(class_scores)

        # Append generic-object
        class_scores = torch.cat([class_scores, class_scores.max(dim=3, keepdim=True)[0]], dim=3)

        boxes = boxes * torch.tensor([W, H, W, H], device=images.device).view(1, 1, 1, 4)
        boxes = torch.cat(
            [
                boxes[:, :, :, 0:2] - 0.5 * boxes[:, :, :, 2:4],
                boxes[:, :, :, 0:2] + 0.5 * boxes[:, :, :, 2:4],
            ],
            dim=-1,
        )

        # Extract the detections to compare with the annotations
        if L == boxes.shape[1]:
            annotated_frame_class_scores = class_scores[range(B), data["annotated_frame_idx"]]
            annotated_frame_boxes = boxes[range(B), data["annotated_frame_idx"]]
        else:
            assert boxes.shape[1] == 1, "If different #outs than #ins, #outs must be 1"
            # assume that the detection frame (index 0) corresponds to the annotated frame
            annotated_frame_class_scores = class_scores[:, 0]
            annotated_frame_boxes = boxes[:, 0]

        od_map_stuffs = prepare_od_map_stuffs(
            annotated_frame_boxes,
            annotated_frame_class_scores,
            data["boxes"],
            data["classes"],
            data["active"],
            (H, W),
        )
        output = {
            "class_scores": class_scores[:, :, None, ...],
            "boxes": boxes[:, :, None, ...],
        }
        return od_map_stuffs, output


def to_detr_targets(H, W, anno_active, anno_boxes, anno_classes):
    """
    Returns:
        list of {
            'labels': Tensor of size (Nb,),
            'boxes': Tensor of size (Nb, 4)
        }. Only active objects are included. Thus, the tensors vary in size.
    """
    # Convert our representations into one desired by the DETR set criterion. For predictions,
    # we also select predictions for which we have annotations.
    # @todo - this does not take future frame predictions in dimension T into account.
    # @todo - we do not yet use auxillary losses
    anno_boxes = torch.cat(
        [
            0.5 * (anno_boxes[:, :, 0:2] + anno_boxes[:, :, 2:4]),
            anno_boxes[:, :, 2:4] - anno_boxes[:, :, 0:2],
        ],
        dim=2,
    )
    anno_boxes = anno_boxes * torch.tensor(
        [1 / W, 1 / H, 1 / W, 1 / H], device=anno_boxes.device
    ).view(1, 1, 4)
    targets = [
        {"labels": classes[active == 1], "boxes": boxes[active == 1]}
        for classes, boxes, active in zip(anno_classes, anno_boxes, anno_active)
    ]
    return targets
