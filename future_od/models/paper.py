"""Paper model.

Separate encoder - each frame processed independently (batched)
    - CNN backbone
    - optionally single frame transformer encoder
Joint encoder - frames that belong to the same sequence are processed jointly
    - for example joint transformer encoder
Decoder - cross-attends to the image features and produces a single set of (future) predictions
    - for example single decoder that attends to all frames jointly (with temporal encoding)
    - or separate attention layers to each previous frame (without temporal encoding)
    - or a recurrent variant with slotstates

"""
import math
from typing import Dict, List, Optional, Tuple

from einops import rearrange
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torchvision
from torch import Tensor
from torchvision.models._utils import IntermediateLayerGetter

from future_od.models.transformer import MLP, TransformerEncoder

from ConditionalDETR.models.backbone import FrozenBatchNorm2d
from ConditionalDETR.util.misc import inverse_sigmoid, is_main_process


class PositionalEncoder(nn.Module):
    def __init__(
        self,
        no_temporal: bool = False,
        temperature=10000,
        extra_temporal_offset: float = 0.0,
    ):
        super().__init__()
        self._no_temporal = no_temporal
        self._temperature = temperature
        self._extra_temporal_offset = extra_temporal_offset
        self.scale = 2 * math.pi
        self._pos_encodings = {}  # shape to encoding

    def get_spatial_encoding(self, b, c, h, w, device):
        mask = torch.ones((b, 1, h, w), dtype=torch.float32, device=device)
        return self._build_spatial(mask, c)[:, 0]

    def get_spatio_temporal_encoding(self, b, l, c, h, w, device, temporal_offsets):
        mask = torch.ones((b, l, h, w), dtype=torch.float32, device=device)
        encoding = self._build_spatial(mask, c)
        if not self._no_temporal:
            encoding += self._build_temporal(mask, c, temporal_offsets)
        return encoding

    def _build_spatial(self, mask, c):
        y_embed = mask.cumsum(2, dtype=torch.float32)
        x_embed = mask.cumsum(3, dtype=torch.float32)
        eps = 1e-6
        y_embed = self._encode(y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale, c // 2)
        x_embed = self._encode(x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale, c // 2)
        pos_enc = torch.cat((y_embed, x_embed), dim=4)
        return rearrange(pos_enc, "b l h w c -> b l c h w")

    def _build_temporal(self, mask, c, temporal_offsets=None):
        if temporal_offsets is not None:
            t_embed = mask * temporal_offsets[..., None, None] + self._extra_temporal_offset
        else:
            t_embed = mask.cumsum(1, dtype=torch.float32)
        eps = 1e-6
        t_embed = self._encode(t_embed / (t_embed[:, -1:, :, :] + eps) * self.scale, c)
        return rearrange(t_embed, "b l h w c -> b l c h w")

    def _encode(self, embedding, num_features):
        dim_t = torch.arange(num_features, dtype=torch.float32, device=embedding.device)
        dim_t = self._temperature ** (2 * (dim_t // 2) / num_features)
        pos = embedding[..., None] / dim_t
        pos = torch.stack((pos[..., 0::2].sin(), pos[..., 1::2].cos()), dim=5).flatten(4)
        return pos


class CDetrBackbone(nn.Module):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        dilation: bool,
        hidden_dim: int,
        pretrained=True,
    ):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process() and pretrained,
            norm_layer=FrozenBatchNorm2d,
        )
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__()

        for name, parameter in backbone.named_parameters():
            if (
                not train_backbone
                or "layer2" not in name
                and "layer3" not in name
                and "layer4" not in name
            ):
                parameter.requires_grad_(False)
        self.body = IntermediateLayerGetter(backbone, return_layers={"layer4": "0"})
        self.num_channels = num_channels
        self.input_proj = nn.Conv2d(self.num_channels, hidden_dim, kernel_size=(1, 1))

    def forward(self, images):
        features = self.body(images)
        return self.input_proj(features["0"])


class SeparateEncoder(nn.Module):
    def __init__(
        self,
        backbone: CDetrBackbone,
        transformer: TransformerEncoder = None,
        imu_layers: nn.Module = None,
        concat_imu: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.imu_layers = imu_layers
        self.transformer = transformer
        self.concat_imu = concat_imu

    def forward(
        self, images: Tensor, pos_encoder: PositionalEncoder, imu: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            images: of size (B, L, 3, H, W)
            pos_encoder:
            imu:
        Returns:
            Tensor: of size (B L C H W) - single frame features
            Tensor: of size (B L C) - encoded egomotion information
        """
        b, l = images.size()[:2]
        images = rearrange(images, "b l c h w -> (b l) c h w")
        features = self.backbone(images)
        _, _, h, w = features.size()
        if imu is not None and self.imu_layers is not None:
            egodeep = self.imu_layers(imu)
        else:
            egodeep = None
        if self.concat_imu:
            egodeep = rearrange(egodeep, "b l c -> (b l) c 1 1").expand(-1, -1, h, w)
            features = features + egodeep
            egodeep = None
        if self.transformer:
            pos_enc = pos_encoder.get_spatial_encoding(*features.shape, features.device)
            features = rearrange(features, "bl c h w -> (h w) bl c")
            pos_enc = rearrange(pos_enc, "bl c h w -> (h w) bl c")
            if egodeep is not None:
                egodeep = rearrange(egodeep, "b l c -> 1 (b l) c")

            features = self.transformer(features, None, None, image_pos=pos_enc, egodeep=egodeep)
            features = rearrange(features, "(h w) (b l) c -> b l c h w", b=b, l=l, h=h, w=w)
            if egodeep is not None:
                egodeep = rearrange(egodeep, "1 (b l) c -> b l c", b=b, l=l)
        else:
            features = rearrange(features, "(b l) c h w -> b l c h w", b=b, l=l)
        return features, egodeep


class _JointEncoderBase(nn.Module):
    def forward(
        self, features: Tensor, pos_enc: Tensor, egodeep: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


class JointEncoder(_JointEncoderBase):
    def __init__(self, transformer: TransformerEncoder):
        super().__init__()
        self.transformer = transformer

    def forward(
        self, features: Tensor, pos_enc: Tensor, egodeep: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            features: of size (B, L, C, H, W)
            pos_enc: of size (B, L, C, H, W)
            egodeep: of size (B, L, Cimu)
        Returns:
            Tensor: features of size (B, L, C, H, W)
        """
        B, L, C, H, W = features.size()
        features = rearrange(features, "b l c h w -> (h w l) b c")
        pos_enc_ = rearrange(pos_enc, "b l c h w -> (h w l) b c")
        if egodeep is not None:
            egodeep = rearrange(egodeep, "b l c -> l b c")
        features = self.transformer(features, None, None, image_pos=pos_enc_, egodeep=egodeep)
        features = rearrange(features, "(h w l) b c -> b l c h w", l=L, h=H, w=W)
        return features, pos_enc


class JointEncoderSequential(JointEncoder):
    def forward(
        self, features: Tensor, pos_enc: Tensor, egodeep: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            features: of size (B, L, C, H, W)
            pos_enc: of size (B, L, C, H, W)
            egodeep: of size (B, L, Cimu)
        Returns:
            Tensor: features of size (B, L, C, H, W)
        """
        B, L, C, H, W = features.size()
        features = rearrange(features, "b l c h w -> l (h w) b c")
        pos_enc_ = rearrange(pos_enc, "b l c h w -> l (h w) b c")
        if egodeep is not None:
            egodeep = rearrange(egodeep, "b l c -> l 1 b c")
        else:
            egodeep = [None for l in range(L)]
        out_lst = []
        out = None
        features_mem = []
        for l in range(L):
            out = self.transformer(features[l], out, features_mem, pos_enc_[l], egodeep=egodeep[l])
            features_mem = [features[l]] + features_mem
            out_lst.append(out)
        out_all_frames = torch.stack(out_lst, dim=0)
        out_all_frames = rearrange(out_all_frames, "l (h w) b c -> b l c h w", l=L, h=H, w=W)
        return out_all_frames, pos_enc


class JointEncoderF2F(_JointEncoderBase):
    """This is a joint decoder is a reproduction of the mechanism of F2F.

    https://arxiv.org/abs/1803.11496
    """

    def __init__(self, hidden_dim, num_frames):
        super().__init__()
        p, n = hidden_dim, num_frames
        self.f2f_model = torch.nn.Sequential(
            torch.nn.Conv2d(n * p, 2 * p, kernel_size=(1, 1), padding="same"),
            torch.nn.ReLU(),
            torch.nn.Conv2d(2 * p, 2 * p, kernel_size=(3, 3), dilation=(2, 2), padding="same"),
            torch.nn.ReLU(),
            torch.nn.Conv2d(2 * p, 2 * p, kernel_size=(3, 3), dilation=(2, 2), padding="same"),
            torch.nn.ReLU(),
            torch.nn.Conv2d(2 * p, p, kernel_size=(3, 3), dilation=(4, 4), padding="same"),
            torch.nn.ReLU(),
            torch.nn.Conv2d(p, p, kernel_size=(3, 3), dilation=(8, 8), padding="same"),
            torch.nn.ReLU(),
            torch.nn.Conv2d(p, p, kernel_size=(3, 3), dilation=(2, 2), padding="same"),
            torch.nn.ReLU(),
            torch.nn.Conv2d(p, p, kernel_size=(7, 7), dilation=(1, 1), padding="same"),
        )

    def forward(
        self, features: Tensor, pos_enc: Tensor, egodeep: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            features: of size (B, L, C, H, W)
            pos_enc: of size (B, L, C, H, W) - not used!
            egodeep: of size (B, L, Cimu) - not used!
        Returns:
            Tensor: features of size (B, L, C, H, W)
        """
        del egodeep  # not used
        features = rearrange(features, "b l c h w -> b (l c) h w")
        features = self.f2f_model(features)
        features = rearrange(features, "b c h w -> b 1 c h w")
        return features, pos_enc[:, -1:]


class CDetrDetectorSpatioTemporal(nn.Module):
    """This version is recurrent. As in the single-frame version, queries are initialized with a
    learnt representation. These queries then gather information from input feature maps and the
    resulting final queries are used to predict objects. In this version, however, the queries are
    initialized with the final queries from the previous frame. Only in the first frame is the
    learnt representation used.
    """

    def __init__(
        self,
        decoder,
        num_classes: int,
        hidden_dim: int,
        first_layer_special_when,
        num_queries=300,
        aux_loss=True,
        image_memory_mode="attend one at a time",
    ):
        super().__init__()
        self.decoder = decoder

        self.aux_loss = aux_loss
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # init bbox_mebed
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        assert first_layer_special_when in ("first frame", "always", "never")
        self._first_layer_special_when = first_layer_special_when

        assert image_memory_mode in ("attend one at a time", "attend all at once")
        self.image_memory_mode = image_memory_mode

        self.num_images = len(self.decoder.layers[0].image_attend)
        assert all([self.num_images == len(layer.image_attend) for layer in self.decoder.layers])
        self.use_slotstates = self.decoder.layers[0].slotstates_attend is not None
        assert all(
            [
                self.use_slotstates == (layer.slotstates_attend is not None)
                for layer in self.decoder.layers
            ]
        )

    def forward(self, features, pos_enc, egodeep):
        B, L, C, H, W = features.size()
        assert L > 0
        if self.image_memory_mode == "attend all at once":
            features = rearrange(features, "b l c h w -> (l h w) b c")
            pos_enc = rearrange(pos_enc, "b l c h w -> (l h w) b c")
            if egodeep is not None:
                egodeep = rearrange(egodeep, "b l c -> l b c")
            out, _ = self.detect(features, pos_enc, egodeep, True)
        else:
            features = rearrange(features, "b l c h w -> l (h w) b c")
            pos_enc = rearrange(pos_enc, "b l c h w -> l (h w) b c")
            if egodeep is not None:
                egodeep = rearrange(egodeep, "b l c -> l 1 b c")
            else:
                egodeep = [None for l in range(L)]
            state = None
            for l in range(L):
                out, state = self.detect(features[l], pos_enc[l], egodeep[l], l == 0, state)
        return out

    def detect(self, frame_features, pos_embed, egodeep, first_frame=True, state=None):
        """Here, (H, W) is the image size, B the batch size, D the feature dimensionality, M the
        number of queries, and C the number of object categories.
        Args:
            frame_features (Tensor): Of size (HW, B, Din)
            pos_embed (Tensor)     : Of size (HW, B, Din)
            egodeep (Tensor)           : Of size (B, L, Cimu)
        Returns:
            dict {
                "pred_logits": Tensor of size (B, M, C) containing non-sigmoided scores
                "pred_boxes" : Tensor of size (B, M, 4) containing boxes TODO: ON WHAT FORM???
                "aux_outputs": List of dict, each containing pred_logits and pred_boxes for an
                    intermediate part of the decoder.
            }
            Tensor: Of size (B, M, D) containing the state, namely the final queries
        """
        num_tokens, batch_size, channels = frame_features.shape
        assert pos_embed.shape[0] == num_tokens
        query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)  # (M, B, D)
        query_content = torch.zeros_like(query_pos)

        if state is None:
            image_content_lst = [frame_features]
            slotstates_content = None
        else:
            image_content_lst = [frame_features] + state["image_content_lst"]
            slotstates_content = state["slotstates_content"]
        if self.image_memory_mode == "attend one at a time":
            image_pos_lst = [pos_embed for _ in image_content_lst]
        elif self.image_memory_mode == "attend all at once":
            image_pos_lst = [pos_embed]
        else:
            raise NotImplementedError()

        hs, reference = self.decoder(
            query_content=query_content,
            query_pos=query_pos,
            image_content_lst=image_content_lst,
            image_pos_lst=image_pos_lst,
            slotstates_content=slotstates_content,
            first_layer_special=(
                (first_frame and self._first_layer_special_when == "first frame")
                or self._first_layer_special_when == "always"
            ),
            egodeep=egodeep,
        )
        # TODO: remove use_slotstates
        state = {
            "slotstates_content": hs[-1].transpose(0, 1) if self.use_slotstates else None,
            "image_content_lst": image_content_lst[: self.num_images - 1],
        }
        # hs is (num_decoder_layers, B, M, D). The last sample are the final queries.
        # reference is (B, 128, 2) with some settings, not sure where 128 and 2 comes from

        reference_before_sigmoid = inverse_sigmoid(reference)
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            tmp = self.bbox_embed(hs[lvl])
            tmp[..., :2] += reference_before_sigmoid
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
        outputs_coord = torch.stack(outputs_coords)

        outputs_class = self.class_embed(hs)
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)
        return out, state

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


class FuturePredCore(nn.Module):
    """This core is focused on performing future prediction."""

    def __init__(
        self,
        separate_encoder: SeparateEncoder,
        joint_encoder: Optional[_JointEncoderBase],
        detector: CDetrDetectorSpatioTemporal,
        pos_encoder: PositionalEncoder,
    ):
        super().__init__()
        self.separate_encoder = separate_encoder
        self.joint_encoder = joint_encoder
        self.detector = detector
        self.pos_encoder = pos_encoder

    def forward(
        self, images: Tensor, imu: Tensor = None, temporal_offsets: Tensor = None
    ) -> Tuple[Dict[str, Tensor], List]:
        """Perform future prediction.

        Args:
            images (tensor): of size (B, L, 3, H, W)
            imu (Tensor or None): of size (B, L, 13)
            temporal_offsets (Tensor or None): of size (B, L)
        Returns:
            dict: see CDetrDetector for details on this dict
            list: model moods (TODO: what does this have to be?)
        """
        B, L, _, _, _ = images.size()

        # Remove the last frame, assuming that this is the "future" image that we want to predict
        images = images[:, :-1]
        if imu is not None:
            imu = imu[:, :-1]
        if temporal_offsets is not None:
            temporal_offsets = temporal_offsets[:, :-1]

        # Run per-frame feature extraction
        features, egodeep = self.separate_encoder(images, self.pos_encoder, imu)

        # Create positional encoding
        pos_enc = self.pos_encoder.get_spatio_temporal_encoding(
            *features.size(), features.device, temporal_offsets
        )

        # Run joint feature extraction
        if self.joint_encoder:
            features, pos_enc = self.joint_encoder(features, pos_enc, egodeep)

        # Run detector decoder
        future_pred = self.detector(features, pos_enc, egodeep)
        model_moods = [["model happy" for _ in range(L)] for _ in range(B)]
        return future_pred, model_moods


class SingleFrameCore(nn.Module):
    """This core is focused on single frame predictions."""

    def __init__(
        self,
        encoder: SeparateEncoder,
        detector: CDetrDetectorSpatioTemporal,
        pos_encoder: PositionalEncoder,
    ):
        super().__init__()
        self.encoder = encoder
        self.detector = detector
        self.pos_encoder = pos_encoder

    def forward(
        self, images: Tensor, imu: Tensor = None, temporal_offsets: Tensor = None
    ) -> Tuple[Dict[str, Tensor], List]:
        """Perform future prediction.

        Args:
            images (Tensor): of size (B, L, 3, H, W)
            imu (Tensor or None): of size (B, L, 13)
            temporal_offsets (Tensor or None): of size (B, L)
        Returns:
            dict: see CDetrDetector for details on this dict
            list: model moods (TODO: what does this have to be?)
        """
        B, L = images.size()[:2]

        # Run per-frame feature extraction
        features, egodeep = self.encoder(images, self.pos_encoder, imu)

        # Create positional encoding
        pos_enc = self.pos_encoder.get_spatio_temporal_encoding(
            *features.size(), features.device, temporal_offsets
        )

        # Run detector decoder
        future_pred = self.detector(features, pos_enc, egodeep)
        model_moods = [["model happy" for _ in range(L)] for _ in range(B)]
        return future_pred, model_moods


class TrackerFuturePredictor(nn.Module):
    """Future predictor based on simple assignment between frames and inter/extrapolation."""

    def __init__(self, dim_extrapolation: str = None):
        super().__init__()
        self._dim_extrapolation = dim_extrapolation

    def _get_center_distances(self, boxes1, boxes2):
        many_to_many_distances = torch.cdist(boxes1[:, :, 0:2], boxes2[:, :, 0:2], p=2)
        return many_to_many_distances

    def _get_class_disparities(self, logits1, logits2):
        disparities = torch.cdist(logits1.sigmoid(), logits2.sigmoid(), p=float("inf"))
        return disparities

    def _get_assignment(self, batch_cost_tensor):
        B, M, N = batch_cost_tensor.size()
        device = batch_cost_tensor.device
        ids = [linear_sum_assignment(cost) for cost in batch_cost_tensor.cpu().numpy()]
        row_ids = torch.stack([torch.tensor(row_ids) for row_ids, col_ids in ids])
        col_ids = torch.stack([torch.tensor(col_ids) for col_ids, col_ids in ids])

        row_to_col_mapping = torch.full((B, M), -1, dtype=torch.int64).scatter(
            dim=1,
            index=row_ids,  # Scatter to these rows
            src=col_ids,  # Scatter these values
        )
        return row_to_col_mapping.to(device)

    def _extrapolate(self, pred2, pred1, pred2_to_pred1_mapping, factor):
        boxes2 = pred2["pred_boxes"]
        pred2_has_pred1 = pred2_to_pred1_mapping != -1
        pred2_to_pred1_mapping[~pred2_has_pred1] = 0  # Gather cannot deal with -1

        corresponding_boxes1 = pred1["pred_boxes"].gather(
            dim=1,
            index=pred2_to_pred1_mapping[:, :, None].expand(-1, -1, 4),
        )
        # Unmatched boxes are kept as is
        corresponding_boxes1[~pred2_has_pred1] = pred2["pred_boxes"][~pred2_has_pred1]
        extrapolated_box_dim = self._extrapolate_box_dim(boxes2, corresponding_boxes1, factor)
        extrapolated_box_pos = (
            boxes2[..., 0:2] + (boxes2[..., 0:2] - corresponding_boxes1[..., 0:2]) * factor
        )
        extrapolated_boxes = torch.cat([extrapolated_box_pos, extrapolated_box_dim], dim=2)

        _, _, C = pred1["pred_logits"].size()
        corresponding_logits1 = pred1["pred_logits"].gather(
            dim=1,
            index=pred2_to_pred1_mapping[..., None].expand(-1, -1, C),
        )
        corresponding_logits1[~pred2_has_pred1] = 0.0  # Unmatched logits will not add info
        interpolated_logits = 0.5 * (pred2["pred_logits"] + corresponding_logits1)
        pred3 = {
            "pred_boxes": extrapolated_boxes,
            "pred_logits": interpolated_logits,
        }
        return pred3

    def _extrapolate_box_dim(self, boxes2, corresponding_boxes1, factor):
        if self._dim_extrapolation is None:
            return boxes2[..., 2:4]
        elif self._dim_extrapolation == "linear":
            box_dims = (
                boxes2[..., 2:4] + (boxes2[..., 2:4] - corresponding_boxes1[..., 2:4]) * factor
            )
            return torch.clamp(box_dims, min=0)
        elif self._dim_extrapolation == "percentual":
            return boxes2[..., 2:4] * (boxes2[..., 2:4] / corresponding_boxes1[..., 2:4]) ** factor
        elif self._dim_extrapolation == "average":
            return (boxes2[..., 2:4] + corresponding_boxes1[..., 2:4]) / 2
        else:
            raise ValueError("Unknown dim extrapolation: " % self._dim_extrapolation)

    def forward(self, pred1, pred2, temporal_offsets=None):
        """Takes two TransformerDecoder outputs and extrapolates based on a simple tracker.

        The two transformer decoder outputs corresponds to two sets of detection hypotheses. The
        two sets correspond to two neighbouring frames. First, assignment is made between the
        detections, using their spatial similarity as cost and then minimizing the total cost.
        Next, the future of the detections are obtained via extrapolation.

        Args:
            pred1: Previous frame prediction as dict {
                "pred_logits": Tensor of size (B, N, C). Yields class probabilities when sigmoided.
                "pred_boxes": Tensor of size (B, N, 4). Box on form (cx, cy, w, h) in range [0, 1]
                "aux_outputs": Not used
            }
            pred2: Current frame prediction as dict {
                "pred_logits": Tensor of size (B, M, C). Yields class probabilities when sigmoided.
                "pred_boxes": Tensor of size (B, M, 4). Box on form (cx, cy, w, h) in range [0, 1]
                "aux_outputs": Not used
            }
            temporal_offsets: float tensor of size (B, L)

        Returns:
            Future frame prediction as dict {
                "pred_logits": Tensor of size (B, M, C). Yields class probabilities when sigmoided.
                "pred_boxes": Tensor of size (B, M, 4). Box on form (cx, cy, w, h) in range [0, 1]
            }
        """
        with torch.no_grad():
            center_distances = self._get_center_distances(pred2["pred_boxes"], pred1["pred_boxes"])
            class_disparities = self._get_class_disparities(
                pred2["pred_logits"], pred1["pred_logits"]
            )
            if temporal_offsets is None:
                extrapolation_factor = 1.0
            else:
                first_offset = temporal_offsets[:, 1] - temporal_offsets[:, 0]
                second_offset = temporal_offsets[:, 2] - temporal_offsets[:, 1]
                extrapolation_factor = (second_offset / first_offset)[:, None, None]
            cost = 0.5 * center_distances + 0.5 * class_disparities  # (B, M, N) tensor
            pred2_to_pred1_mapping = self._get_assignment(cost)  # (B, M) tensor
            pred3 = self._extrapolate(pred2, pred1, pred2_to_pred1_mapping, extrapolation_factor)
        return pred3


class TrackerBaselineCore(nn.Module):
    """This core is focused on single frame predictions."""

    def __init__(
        self,
        encoder: SeparateEncoder,
        detector: CDetrDetectorSpatioTemporal,
        pos_encoder: PositionalEncoder,
        tracker_future_predictor: TrackerFuturePredictor,
    ):
        super().__init__()
        self.encoder = encoder
        self.detector = detector
        self.pos_encoder = pos_encoder
        self.tracker_future_predictor = tracker_future_predictor

    def forward(
        self, images: Tensor, imu: Tensor = None, temporal_offsets: Tensor = None
    ) -> Tuple[Dict[str, Tensor], List]:
        """Perform future prediction.

        Args:
            images (Tensor): of size (B, L, 3, H, W)
            imu (Tensor or None): of size (B, L, 13)
            temporal_offsets (Tensor or None): of size (B, L)
        Returns:
            dict: see CDetrDetector for details on this dict
            list: model moods (TODO: what does this have to be?)
        """
        B, L, _, _, _ = images.size()

        # Run per-frame feature extraction
        features, egodeep = self.encoder(images, self.pos_encoder, imu)

        # Create positional encoding
        pos_enc = self.pos_encoder.get_spatio_temporal_encoding(
            *features.size(), features.device, temporal_offsets
        )

        # Run detector decoder
        if L == 1:
            # During training, L=1 and detections for current frame are provided
            pred = self.detector(features, pos_enc, egodeep)
        elif L == 3:
            # During evaluation, L=3 and detections for first two frames are provided. Detections
            # for the third frame are provided by the tracker.
            preds = [
                self.detector(
                    features[:, l : l + 1],
                    pos_enc[:, l : l + 1],
                    egodeep[:, l : l + 1],
                )
                for l in range(L - 1)
            ]
            pred = self.tracker_future_predictor(preds[0], preds[1], temporal_offsets)

        model_moods = [["model happy" for _ in range(L)] for _ in range(B)]
        return pred, model_moods
