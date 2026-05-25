import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class SoftSignGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, out, gate):
        denom_out = out.abs().add(1.0)
        denom_gate = gate.abs().add(1.0)
        out = out / denom_out
        gate = gate / denom_gate
        ctx.save_for_backward(
            out / denom_gate / denom_gate,
            gate / denom_out / denom_out)
        return out * gate

    @staticmethod
    def backward(ctx, grad_output):
        out_d_gate, gate_d_out = ctx.saved_tensors
        grad_out_part = grad_output * gate_d_out
        grad_gate_part = grad_output * out_d_gate
        return grad_out_part, grad_gate_part


class SoftSignGLU(nn.Module):
    # SoftSign-Applies the gated linear unit function.
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # out, gate = x.chunk(2, dim=self.dim)
        # Using torch.split instead of chunk for ONNX export compatibility.
        out, gate = torch.split(x, x.size(self.dim) // 2, dim=self.dim)
        return SoftSignGLUFunction.apply(out, gate)


class LYNXNet2Block(nn.Module):
    def __init__(self, dim, kernel_size=11, use_dwconv=True):
        super().__init__()
        self.net = nn.Sequential(
            Transpose((1, 2)),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim if use_dwconv else 1),
            Transpose((1, 2)),
            nn.Linear(dim, dim * 2),
            SoftSignGLU(),
            nn.Linear(dim, dim * 2),
            SoftSignGLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x, norm_x=None):
        if norm_x is None:
            norm_x = F.rms_norm(x, (x.size(-1), ))
        x = x + self.net(norm_x)
        return x


class FastWaveD(torch.nn.Module):
    def __init__(self, init_channel=16, strides=[4, 4, 4], kernel_size=31):
        super(FastWaveD, self).__init__()
        self.strides = strides
        self.hop_length = np.prod(self.strides)
        self.pre = nn.Linear(strides[0], init_channel * strides[0])
        self.residual_layers = nn.ModuleList(
            [
                LYNXNet2Block(
                    dim=init_channel * np.prod(strides[: i + 1]),
                    kernel_size=kernel_size,
                    use_dwconv=True,
                )
                for i in range(len(strides))
            ]
        )
        self.post = nn.Linear(init_channel * np.prod(strides), 1)

    def forward(self, x, infer_last_layer=True):
        fmap = []

        b, _, t = x.shape
        x = x[:, :, : (t // self.hop_length) * self.hop_length].view(b, -1, self.strides[0])

        x = self.pre(x)
        x = F.gelu(x)
        for i, layer in enumerate(self.residual_layers):
            if i > 0 and self.strides[i] > 1:
                x = x.view(b, -1, x.size(2) * self.strides[i])
            norm_x = F.rms_norm(x, (x.size(-1), ))
            if i > 0:
                fmap.append(norm_x.view(b, -1))
            if not infer_last_layer and i == len(self.residual_layers) - 1:
                return [norm_x], [fmap]
            x = layer(x, norm_x)
        x = F.rms_norm(x, (x.size(-1), ))
        x = self.post(x)
        x = x.view(b, -1)

        return [x], [fmap]

class ResBlock(nn.Module):
    def __init__(self, dim, use_dwconv=False):
        super().__init__()
        self.net = nn.Sequential(
            Permute((0, 3, 1, 2)),
            nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1, groups=dim if use_dwconv else 1),
            Permute((0, 2, 3, 1)),
            SoftSignGLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x, norm_x=None):
        if norm_x is None:
            norm_x = F.rms_norm(x, (x.size(-1), ))
        x = x + self.net(norm_x)
        return x
