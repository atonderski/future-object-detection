import torch.nn as nn

import future_od.models.transformer as transformer
from future_od.models.paper import (
    CDetrBackbone,
    CDetrDetectorSpatioTemporal,
    FuturePredCore,
    PositionalEncoder,
    SeparateEncoder,
)
from future_od.models.st_detr import SpatioTemporalDETR, SpatioTemporalDETRArgs


def build_model(args, detr_args: SpatioTemporalDETRArgs):
    model = SpatioTemporalDETR(
        args=detr_args,
        model=FuturePredCore(
            separate_encoder=SeparateEncoder(
                backbone=CDetrBackbone(
                    name="resnet50",
                    train_backbone=detr_args.lr_backbone > 0,
                    dilation=False,
                    hidden_dim=detr_args.hidden_dim,
                    pretrained=detr_args.pretrained_backbone,
                ),
                imu_layers=nn.Sequential(
                    nn.Linear(14, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, detr_args.hidden_dim),
                ),
                transformer=transformer.TransformerEncoder(
                    layers=nn.ModuleList(
                        transformer.TransformerEncoderLayer(
                            D=detr_args.hidden_dim,
                            Nhead=detr_args.enc_nheads,
                            Dff=detr_args.dim_feedforward,
                            use_egodeep=True,
                        )
                        for _ in range(detr_args.enc_layers)
                    )
                ),
            ),
            joint_encoder=None,
            detector=CDetrDetectorSpatioTemporal(
                decoder=transformer.TransformerDecoder(
                    layers=nn.ModuleList(
                        [
                            transformer.TransformerDecoderLayer(
                                D=detr_args.hidden_dim,
                                Nhead=detr_args.nheads,
                                Dff=detr_args.dim_feedforward,
                                dropout=0.1,
                                num_images=2,
                                use_slotstates=False,
                            )
                            for _ in range(detr_args.dec_layers)
                        ]
                    ),
                    norm=nn.LayerNorm(detr_args.hidden_dim),
                    return_intermediate=True,
                    D=detr_args.hidden_dim,
                ),
                num_classes=detr_args.num_classes,
                hidden_dim=detr_args.hidden_dim,
                first_layer_special_when="always",
                num_queries=detr_args.num_queries,
                aux_loss=True,
                image_memory_mode="attend one at a time",
            ),
            pos_encoder=PositionalEncoder(
                no_temporal=True,
            ),
        ),
    )
    model.to(args.device)
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.device],
            output_device=args.device,
            find_unused_parameters=False,
        )
    return model
