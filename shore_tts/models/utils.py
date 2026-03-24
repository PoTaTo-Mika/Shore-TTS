from __future__ import annotations

import torch
from torch import einsum, tensor, Tensor, cat, stack, arange, is_tensor
import torch.nn as nn
from torch.amp import autocast
from einops import rearrange, repeat, reduce, pack, unpack
from torch.nn.utils.rnn import pad_sequence

# Basic null-check helper used across the copied F5-TTS code.
def exists(val):
    return val is not None

# Return `v` when present, otherwise fall back to `d`.
def default(v, d):
    return v if exists(v) else d

# Convert per-sample valid lengths into a boolean mask of shape `(B, N)`.
def lens_to_mask(t: int["b"], length: int | None = None) -> bool["b n"]:
    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]

# Empirically Pruned Step Sampling schedule used by F5-TTS for low-step ODE sampling.
def get_epss_timesteps(n, device, dtype):
    dt = 1 / 32
    predefined_timesteps = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    t = predefined_timesteps.get(n, [])
    if not t:
        return torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
    return dt * torch.tensor(t, device=device, dtype=dtype)

# RoPE helper: swap each even / odd pair into its quarter-turn counterpart.
def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

# Apply rotary position embedding to the rotatable prefix of the attention tensor.
def apply_rotary_pos_emb(t, freqs, scale = 1):
    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype

    freqs = freqs[:, -seq_len:, :]
    scale = scale[:, -seq_len:, :] if is_tensor(scale) else scale

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

        if is_tensor(scale):
            scale = rearrange(scale, 'b n d -> b 1 n d')

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    out = cat((t, t_unrotated), dim = -1)

    return out.type(orig_dtype)

def is_package_available(package_name: str) -> bool:
    try:
        import importlib

        package_exists = importlib.util.find_spec(package_name) is not None
        return package_exists
    except Exception:
        return False

class RotaryEmbedding(nn.Module):
    # Local RoPE / XPos implementation so the model does not depend on x_transformers.
    def __init__(
        self,
        dim,
        use_xpos = False,
        scale_base = 512,
        interpolation_factor = 1.,
        base = 10000,
        base_rescale_factor = 1.
    ):
        super().__init__()

        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1. / (base ** (arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        assert interpolation_factor >= 1.
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer('scale', None)
            return

        scale = (arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer('scale', scale)

    def forward_from_seq_len(self, seq_len):
        device = self.inv_freq.device

        t = arange(seq_len, device = device)
        return self.forward(t)

    @autocast('cuda', enabled = False)
    def forward(self, t, offset = 0):
        max_pos = t.max() + 1

        if t.ndim == 1:
            t = rearrange(t, 'n -> 1 n')

        freqs = torch.einsum('b i , j -> b i j', t.type_as(self.inv_freq), self.inv_freq) / self.interpolation_factor
        freqs = stack((freqs, freqs), dim = -1)
        freqs = rearrange(freqs, '... d r -> ... (d r)')

        if not exists(self.scale):
            return freqs, 1.

        power = (t - (max_pos // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, '... n -> ... n 1')
        scale = stack((scale, scale), dim = -1)
        scale = rearrange(scale, '... d r -> ... (d r)')

        return freqs, scale

# Vocabulary tokenizer: map each token through a provided lookup table.
def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1,
) -> int["b nt"]:
    list_idx_tensors = [
        torch.tensor([vocab_char_map.get(c, 0) for c in t], dtype=torch.long) for t in text
    ]  # pinyin or char style
    text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text

# Build a boolean span mask from explicit per-sample start / end indices.
def mask_from_start_end_indices(seq_len: int["b"], start: int["b"], end: int["b"]):
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask

# Sample a random contiguous span for each sequence based on a fractional target length.
def mask_from_frac_lengths(seq_len: int["b"], frac_lengths: float["b"]):
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)
