import copy
import math
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ConditionalDETR.models.attention import MultiheadAttention


def _reset_parameters(parameters):
    for p in parameters:
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def gen_sineembed_for_position(pos_tensor, D=256):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(D // 2, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / (D // 2))
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2)
    return pos


class Attention(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.query_content = nn.Linear(D, D)
        self.query_pos = nn.Linear(D, D)
        self.key_content = nn.Linear(D, D)
        self.key_pos = nn.Linear(D, D)
        self.value = nn.Linear(D, D)


class SlotToSlotAttention(Attention):
    def __init__(self, D, Nhead, dropout):
        super().__init__(D)
        self.fun = MultiheadAttention(D, Nhead, dropout=dropout, vdim=D)

    def forward(
        self,
        query_content,
        query_pos,
        key_content,
        key_pos,
        attn_mask,
        key_padding_mask,
    ):
        out = self.fun(
            query=self.query_content(query_content) + self.query_pos(query_pos),
            key=self.key_content(key_content) + self.key_pos(key_pos),
            value=self.value(key_content),
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]
        return out


class EgodeepAttention(nn.Module):
    def __init__(self, D, Nhead, droprate, Dff=None):
        super().__init__()
        self.query_content = nn.Linear(D, D)
        self.query_pos = nn.Linear(D, D)
        self.key = nn.Linear(D, D)
        self.value = nn.Linear(D, D)
        self.fun = MultiheadAttention(D, Nhead, dropout=droprate, vdim=D)
        if Dff is not None:
            self.use_mlp = True
            self.attn_dropout = nn.Dropout(droprate)
            self.norm1 = nn.LayerNorm(D)
            self.mlp = nn.Sequential(
                nn.Linear(D, Dff),
                nn.ReLU(inplace=True),
                nn.Dropout(droprate),
                nn.Linear(Dff, D),
                nn.Dropout(droprate),
            )
            self.norm2 = nn.LayerNorm(D)
        else:
            self.use_mlp = False

    def forward(self, query_content, query_pos, key):
        out = self.fun(
            query=self.query_content(query_content) + self.query_pos(query_pos),
            key=self.key(key),
            value=self.value(key),
            attn_mask=None,
            key_padding_mask=None,
        )[0]
        if self.use_mlp:
            out = self.norm1(out + self.attn_dropout(out))
            out = self.norm2(out + self.mlp(out))
        return out


class SlotToImageAttention(Attention):
    def __init__(self, D, Nhead, dropout):
        super().__init__(D)
        self.query_sine = nn.Linear(D, D)
        self.fun = MultiheadAttention(D * 2, Nhead, dropout=dropout, vdim=D)
        self.D = D
        self.Nhead = Nhead
        # This will be toggled externally for visualization
        self.store_attention = False

    def forward(
        self,
        query_content,
        query_pos,
        query_sine,
        key_content,
        key_pos,
        key_sine,
        attn_mask,
        key_padding_mask,
    ):
        M, B, _ = query_content.size()
        N, _, _ = key_content.size()
        v = self.value(key_content)
        if query_pos is not None:
            q_content = self.query_content(query_content) + self.query_pos(query_pos)
        else:
            q_content = self.query_content(query_content)
        q_sine = self.query_sine(query_sine)
        q = torch.cat(
            [
                q_content.view(M, B, self.Nhead, self.D // self.Nhead),
                q_sine.view(M, B, self.Nhead, self.D // self.Nhead),
            ],
            dim=3,
        ).view(M, B, self.D * 2)

        k_sine = self.key_pos(key_sine)
        if key_pos is not None:  # key_pos is used to determine whether k_sine should be added
            k_content = self.key_content(key_content) + k_sine
        else:
            k_content = self.key_content(key_content)
        k = torch.cat(
            [
                k_content.view(N, B, self.Nhead, self.D // self.Nhead),
                k_sine.view(N, B, self.Nhead, self.D // self.Nhead),
            ],
            dim=3,
        ).view(N, B, self.D * 2)

        out = self.fun(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        if self.store_attention:
            self.stored_attention = out[1]
        return out[0]


class TransformerDecoderLayer(nn.Module):
    """
    TODO Strange that the encoded input must have the same feat. dim. as the query encodings.
    """

    def __init__(
        self,
        D,
        Nhead,
        Dff=2048,
        dropout=0.1,
        num_images=1,
        use_slotstates=False,
        use_egodeep=False,
    ):
        super().__init__()
        self.self_attend = SlotToSlotAttention(D, Nhead, dropout)
        self.dropout_sa = nn.Dropout(dropout)
        self.norm_sa = nn.LayerNorm(D)

        self.image_attend = nn.ModuleList(
            [SlotToImageAttention(D, Nhead, dropout) for i in range(num_images)]
        )
        self.dropout_ia = nn.ModuleList([nn.Dropout(dropout) for i in range(num_images)])
        self.norm_ia = nn.ModuleList([nn.LayerNorm(D) for i in range(num_images)])

        if use_slotstates:
            self.slotstates_attend = SlotToSlotAttention(D, Nhead, dropout)
            self.dropout_ssa = nn.Dropout(dropout)
            self.norm_ssa = nn.LayerNorm(D)
        else:
            self.slotstates_attend = None

        if use_egodeep:
            self.egodeep_attend = EgodeepAttention(D, Nhead, droprate=dropout, Dff=None)
            self.dropout_eda = nn.Dropout(dropout)
            self.norm_eda = nn.LayerNorm(D)
        else:
            self.egodeep_attend = None

        self.Nhead = Nhead
        self.D = D

        # Implementation of Feedforward model
        self.feedforward = nn.Sequential(
            nn.Linear(D, Dff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(Dff, D),
        )
        self.dropout_out = nn.Dropout(dropout)
        self.norm_out = nn.LayerNorm(D)

        self.activation = nn.ReLU(inplace=True)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        query_content,
        query_pos,
        query_sine,
        image_content_lst,
        image_pos_lst,
        slotstates_content=None,
        slotstates_pos=None,
        slotstates_sine=None,
        is_first=False,
        egodeep: Tensor = None,
    ):
        """
        Args:
            query_content (Tensor) of size (M, B, D)  :
            input_content (Tensor) of size (HW, B, D):
            input_positions (Tensor) of size (HW, B, D)           :
            query_positions (Tensor) of size (M, B, D):
            query_sine_embedd (Tensor) of size (M, B, D):
        """
        query_content_new = self.self_attend(
            query_content=query_content,
            query_pos=query_pos,
            key_content=query_content,
            key_pos=query_pos,
            attn_mask=None,
            key_padding_mask=None,
        )
        query_content = query_content + self.dropout_sa(query_content_new)
        query_content = self.norm_sa(query_content)

        for i, (image_content, image_pos) in enumerate(zip(image_content_lst, image_pos_lst)):
            query_content_new = self.image_attend[i](
                query_content=query_content,
                query_pos=query_pos if is_first else None,
                query_sine=query_sine,
                key_content=image_content,
                key_pos=image_pos if is_first else None,
                key_sine=image_pos,
                attn_mask=None,
                key_padding_mask=None,
            )
            query_content = query_content + self.dropout_ia[i](query_content_new)
            query_content = self.norm_ia[i](query_content)

        if self.slotstates_attend is not None and slotstates_content is not None:
            query_content_new = self.slotstates_attend(
                query_content=query_content,
                query_pos=query_pos,  # if is_first else None,
                key_content=slotstates_content,
                key_pos=slotstates_pos,  # if is_first else None,
                attn_mask=None,
                key_padding_mask=None,
            )
            query_content = query_content + self.dropout_ssa(query_content_new)
            query_content = self.norm_ssa(query_content)

        if self.egodeep_attend is not None and egodeep is not None:
            query_content_new = self.egodeep_attend(
                query_content=query_content,
                query_pos=query_pos,
                key=egodeep,
            )
            query_content = query_content + self.dropout_eda(query_content_new)
            query_content = self.norm_eda(query_content)

        query_content_new = self.feedforward(query_content)
        query_content = query_content + self.dropout_out(query_content_new)
        query_content = self.norm_out(query_content)
        return query_content


class TransformerDecoder(nn.Module):
    """ """

    def __init__(self, layers, norm=None, return_intermediate=False, D=256):
        super().__init__()
        self.layers = layers
        for module in self.layers[1:]:
            for layer in module.image_attend:
                layer.query_pos.weight = None
                layer.query_pos.bias = None
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.D = D
        self.query_scale = MLP(D, D, D, 2)
        self.ref_point_head = MLP(D, D, 2, 2)
        _reset_parameters(self.parameters())

    def forward(
        self,
        query_content,
        query_pos,
        image_content_lst,
        image_pos_lst,
        slotstates_content,
        # slotstates_pos,
        # slotstates_sine,
        first_layer_special=True,
        egodeep: Tensor = None,
    ):
        """B is batch-size, M the number of points, and D the query feature dimensionality.
        Args:
            query_content      (Tensor) of size (M, B, D): Zeros in static case. Recurrent in videos!
            input_content   (Tensor) of size (HW, B, Din): Image features fed through ResNet and Encoder
            input_positions (Tensor) of size (HW, B, Din): Positional encoding. Constant!
            query_positions    (Tensor) of size (M, B, D): Learnable
        Returns:
            Tensor of size (num_layers, B, M, D): Output query vectors that has gathered info from input_content
            Tensor of size (B, M, 2)            : The reference points
        """
        intermediate = []
        reference_points_before_sigmoid = self.ref_point_head(query_pos)  # [M, B, 2]
        reference_points = reference_points_before_sigmoid.sigmoid().transpose(0, 1)

        # get sine embedding for the query vector
        obj_center = reference_points[..., :2].transpose(0, 1)  # [M, B, 2]
        unscaled_query_sine = gen_sineembed_for_position(obj_center, D=self.D)

        for layer_id, layer in enumerate(self.layers):
            if layer_id == 0 and first_layer_special:
                query_sine = unscaled_query_sine
            else:
                query_sine = self.query_scale(query_content) * unscaled_query_sine
            if slotstates_content is not None:
                slotstates_pos = query_pos
                slotstates_sine = self.query_scale(slotstates_content) * unscaled_query_sine
            else:
                slotstates_pos = None
                slotstates_sine = None

            query_content = layer(
                query_content,
                query_pos,
                query_sine,
                image_content_lst,
                image_pos_lst,
                slotstates_content,
                slotstates_pos,
                slotstates_sine,
                is_first=(layer_id == 0) and first_layer_special,
                egodeep=egodeep,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(query_content))
        # TODO: can probably remove all norms here :)
        if self.norm is not None:
            query_content = self.norm(query_content)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(query_content)

        if self.return_intermediate:
            return [torch.stack(intermediate).transpose(1, 2), reference_points]

        return query_content.unsqueeze(0)


class EncoderAttention(nn.Module):
    def __init__(self, Dsrc, num_heads, Dff, droprate=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(Dsrc, num_heads, droprate)
        self.attn_dropout = nn.Dropout(droprate)
        self.norm1 = nn.LayerNorm(Dsrc)
        self.mlp = nn.Sequential(
            nn.Linear(Dsrc, Dff),
            nn.ReLU(inplace=True),
            nn.Dropout(droprate),
            nn.Linear(Dff, Dsrc),
            nn.Dropout(droprate),
        )
        self.norm2 = nn.LayerNorm(Dsrc)

    def forward(self, src, query_base, key_base, val_base):
        src = self.norm1(src + self.attn_dropout(self.attn(query_base, key_base, val_base)[0]))
        src = self.norm2(src + self.mlp(src))
        return src


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        D,
        Nhead,
        Dff=2048,
        droprate=0.1,
        num_previmages=0,
        use_prevout=False,
        use_egodeep=False,
    ):
        super().__init__()
        self.self_attn = EncoderAttention(D, Nhead, Dff, droprate=droprate)
        if use_prevout:
            self.prevout_attn = EncoderAttention(D, Nhead, Dff, droprate=droprate)
        else:
            self.prevout_attn = None
        self.previmage_attn = nn.ModuleList(
            [EncoderAttention(D, Nhead, Dff, droprate=droprate) for _ in range(num_previmages)]
        )
        if use_egodeep:
            self.egodeep_attend = EgodeepAttention(D, Nhead, droprate=droprate, Dff=Dff)
            self.dropout_eda = nn.Dropout(droprate)
            self.norm_eda = nn.LayerNorm(D)
        else:
            self.egodeep_attend = None

    def forward(
        self,
        image_features,
        prevout: Tensor = None,
        image_feature_memory: List[Tensor] = None,
        image_pos: Tensor = None,
        egodeep: Tensor = None,
    ):
        image_features = self.self_attn(
            src=image_features,
            query_base=image_features + image_pos,
            key_base=image_features + image_pos,
            val_base=image_features,
        )
        if prevout is not None and self.prevout_attn is not None:
            image_features = self.prevout_attn(
                src=image_features,
                query_base=image_features + image_pos,
                key_base=prevout + image_pos,
                val_base=prevout,
            )
        if image_feature_memory is not None:
            for previmage_features, attn in zip(image_feature_memory, self.previmage_attn):
                image_features = attn(
                    src=image_features,
                    query_base=image_features + image_pos,
                    key_base=previmage_features + image_pos,
                    val_base=previmage_features,
                )

        if egodeep is not None and self.egodeep_attend is not None:
            image_features_new = self.egodeep_attend(
                query_content=image_features,
                query_pos=image_pos,
                key=egodeep,
            )
            image_features = image_features + self.dropout_eda(image_features_new)
            image_features = self.norm_eda(image_features)
        return image_features


class TransformerEncoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        _reset_parameters(self.parameters())

    def forward(
        self,
        image_features: Tensor,
        prevout: Tensor = None,
        image_feature_memory: Tensor = None,
        image_pos: Tensor = None,
        egodeep: Tensor = None,
    ):
        for layer in self.layers:
            image_features = layer(
                image_features=image_features,
                prevout=prevout,
                image_feature_memory=image_feature_memory,
                image_pos=image_pos,
                egodeep=egodeep,
            )
        return image_features
