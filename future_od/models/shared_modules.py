import torch
import torch.nn as nn
import torch.nn.functional as F


class SequentialWithState(nn.ModuleList):
    stateful = True

    def forward(self, x, state):
        if state is None:
            state = [None for layer in self]
        else:
            state = state.copy()
        for idx, layer in enumerate(self):
            if layer.stateful:
                x, state[idx] = layer(x, state[idx])
            else:
                x = layer(x)
        return x, state


class NoneModule(nn.Module):
    def forward(self, *args, **kwargs):
        return None


class ValueFromDict(nn.Module):
    def __init__(self, key):
        super().__init__()
        self.key = key

    def forward(self, x):
        return x[self.key]


class Attention(nn.Module):
    def __init__(self, Dquery, Dcontext, num_heads, Dhead):
        super().__init__()
        D = num_heads * Dhead
        self.scale = Dhead**-0.5
        self.num_heads = num_heads

        self.to_q = nn.Linear(Dquery, D, bias=False)
        self.to_kv = nn.Linear(Dcontext, D * 2, bias=False)
        self.to_out = nn.Linear(D, Dquery)

    def compute(self, left, right, mask=None):
        """
        Args:
            left (tensor): of size (B, M, D1)
            right (tensor): of size (B, N, D2)
            mask (tensor): of size (B, M, N)
        """
        B, M, _ = left.size()
        _, N, _ = right.size()
        q = self.to_q(left)
        k, v = self.to_kv(right).chunk(2, dim=-1)
        q = q.view(B, M, self.num_heads, q.size(2) // self.num_heads).permute(0, 2, 1, 3)
        k = k.view(B, N, self.num_heads, k.size(2) // self.num_heads).permute(0, 2, 1, 3)
        v = v.view(B, N, self.num_heads, v.size(2) // self.num_heads).permute(0, 2, 1, 3)

        sim = self.scale * torch.einsum("bkmd,bknd->bkmn", q, k)

        if mask is not None:
            mask = mask.view(B, 1, M, N).expand(-1, self.num_heads, -1, -1)
            sim = sim.where(mask, torch.full_like(sim, -1e7))

        attn = sim.softmax(dim=3)
        out = torch.einsum("bkmn,bknd->bkmd", attn, v)
        out = out.view(B, out.size(1), M, out.size(3)).permute(0, 2, 1, 3).reshape(B, M, -1)

        out = self.to_out(out)
        return out

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class SelfAttention(Attention):
    def __init__(self, Dquery, num_heads, Dhead, droprate=0.0, norm=False):
        super().__init__(Dquery, Dquery, num_heads, Dhead)
        if norm:
            self.norm = nn.LayerNorm(Dquery)
        else:
            self.norm = nn.Identity()
        self.droprate = droprate

    def forward(self, left, right=None, mask=None):
        assert mask is None
        left = self.norm(left)
        out = self.compute(left, left, None)
        if self.droprate > 0.0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return out


class CrossAttention(Attention):
    def __init__(self, Dquery, Dcontext, num_heads, Dhead, droprate=0.0, norm=False):
        super().__init__(Dquery, Dcontext, num_heads, Dhead)
        if norm:
            self.norm_left = nn.LayerNorm(Dquery)
            self.norm_right = nn.LayerNorm(Dcontext)
        else:
            self.norm_left = nn.Identity()
            self.norm_right = nn.Identity()
        self.droprate = droprate

    def forward(self, left, right, mask=None):
        assert mask is None
        left = self.norm_left(left)
        right = self.norm_right(right)
        out = self.compute(left, right, None)
        if self.droprate > 0.0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return out


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class PerceptronFF(nn.Module):
    def __init__(self, Din, D, droprate=0.0, norm=False):
        super().__init__()
        if norm:
            self.layers = nn.Sequential(
                nn.LayerNorm(Din),
                nn.Linear(Din, D * 2),
                GEGLU(),
                nn.Dropout(droprate),
                nn.Linear(D, Din),
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(Din, D * 2), GEGLU(), nn.Dropout(droprate), nn.Linear(D, Din)
            )

    def forward(self, left, right=None, mask=None):
        return self.layers(left)


class Linear(nn.Module):
    def __init__(self, Din, Dout):
        super().__init__()
        self.layer = nn.Linear(Din, Dout)

    def forward(self, left, right=None, mask=None):
        return self.layer(left)


class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args):
        return args[0] + self.module(*args)


class MISequential(nn.Sequential):
    """Same as Sequential but it can take and pass along multiple inputs."""

    def forward(self, *inputs):
        for module in self:
            if isinstance(inputs, tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
