from __future__ import annotations

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

# 得益于yx佬实现的torch方案的mdct算子，我们可以一路把梯度回传到mdct处理器当中

class SpectralConvergenceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_mag, y_mag):
        return torch.norm(y_mag - x_mag, p="fro") / torch.clamp(torch.norm(x_mag, p="fro"), min=1e-7)

class LogSTFTMagnitudeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_mag, y_mag):
        return F.l1_loss(torch.log(y_mag + 1e-7), torch.log(x_mag + 1e-7))

class SingleSTFTLoss(nn.Module):
    def __init__(self, fft_size, shift_size, win_length, window="hann"):
        super().__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, f"{window}_window")(win_length))
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        x_stft = torch.stft(x, self.fft_size, self.shift_size, self.win_length, self.window, return_complex=True)
        y_stft = torch.stft(y, self.fft_size, self.shift_size, self.win_length, self.window, return_complex=True)
        
        x_mag = torch.abs(x_stft)
        y_mag = torch.abs(y_stft)

        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)
        return sc_loss, mag_loss

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],     # 不同的频域分辨率
        hop_sizes=[120, 240, 50],        # 不同的时域分辨率
        win_lengths=[600, 1200, 240],    # 不同的窗口长度
        factor_sc=1.0,                   # 谱收敛 Loss 的权重
        factor_mag=1.0                   # 对数幅度 Loss 的权重
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        
        self.stft_losses = nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses.append(SingleSTFTLoss(fs, ss, wl))
            
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self, x, y):
        """
        Args:
            x (Tensor): 真实波形 Ground Truth (B, T) 或 (B, 1, T)
            y (Tensor): 预测波形 Predicted (B, T) 或 (B, 1, T)
        Returns:
            loss (Tensor): 总损失标量
        """
        if x.dim() == 3:
            x = x.squeeze(1) # 转换为 (B, T)
        if y.dim() == 3:
            y = y.squeeze(1)

        sc_loss_total = 0.0
        mag_loss_total = 0.0

        for f in self.stft_losses:
            sc_loss, mag_loss = f(x, y)
            sc_loss_total += sc_loss
            mag_loss_total += mag_loss

        sc_loss_total /= len(self.stft_losses)
        mag_loss_total /= len(self.stft_losses)

        total_loss = (self.factor_sc * sc_loss_total) + (self.factor_mag * mag_loss_total)
        return total_loss


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = mask.unsqueeze(-1).expand_as(pred)
    loss = F.l1_loss(pred, target, reduction="none")
    denom = expanded_mask.sum().clamp_min(1)
    return loss.masked_select(expanded_mask).sum() / denom


class MelSpectrogramLoss(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 100,
    ):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            power=1.0,
        )

    def forward(self, pred_wave: torch.Tensor, target_wave: torch.Tensor) -> torch.Tensor:
        pred_mel = self.mel(pred_wave).clamp_min(1e-5).log()
        target_mel = self.mel(target_wave).clamp_min(1e-5).log()
        return F.l1_loss(pred_mel, target_mel)
