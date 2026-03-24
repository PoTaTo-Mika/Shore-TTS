import torch
import os
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BN_MDCT_Spectrogram(nn.Module):
    def __init__(
        self,
        hop_length=100,
        n_bands=20,
    ):
        super().__init__()
        if hop_length <= 0:
            raise ValueError("hop_length 必须大于 0")
        if n_bands <= 0:
            raise ValueError("n_bands 必须大于 0")

        self.hop_length = hop_length
        self.n_fft = 2 * hop_length
        self.n_bins = hop_length
        self.n_bands = n_bands

        window = torch.sin(torch.pi * (torch.arange(self.n_fft, dtype=torch.float64) + 0.5) / self.n_fft)
        k = torch.arange(self.n_bins, dtype=torch.float64).unsqueeze(0)
        n = torch.arange(self.n_fft, dtype=torch.float64).unsqueeze(1)
        theta = torch.pi * (n + 0.5 + self.n_bins / 2) * (k + 0.5) / self.n_bins
        basis = torch.cos(theta)
        basis_sin = torch.sin(theta)
        
        enc_kernel = basis * window.unsqueeze(1)
        enc_kernel_sin = basis_sin * window.unsqueeze(1)
        dec_kernel = 2.0 * enc_kernel.T / self.hop_length

        self.register_buffer('enc_kernel', enc_kernel.T.unsqueeze(1).float(), persistent=False)
        self.register_buffer('enc_kernel_sin', enc_kernel_sin.T.unsqueeze(1).float(), persistent=False)
        self.register_buffer('dec_kernel', dec_kernel.unsqueeze(1).float(), persistent=False)
        
        centers = torch.linspace(0, self.n_bins - 1, self.n_bands).unsqueeze(0)
        freq_bins = torch.arange(self.n_bins).unsqueeze(1)
        filter_banks = torch.relu(1 - (self.n_bands - 1) * torch.abs(freq_bins - centers) / (self.n_bins - 1))
        filter_banks = filter_banks / filter_banks.sum(dim=0)
        self.register_buffer('filter_banks', filter_banks, persistent=False)

        eye = torch.eye(self.n_bands).unsqueeze(0)
        interp_matrix = F.interpolate(eye, size=self.n_bins, mode='linear', align_corners=True)
        self.register_buffer('interp_matrix', interp_matrix.squeeze(0), persistent=False)

    def _encode_base(self, x: torch.Tensor, sin=False) -> torch.Tensor:
        batch_size, T = x.shape
        pad_remainder = (self.hop_length - (T % self.hop_length)) % self.hop_length
        x_padded = F.pad(x.unsqueeze(1), (self.hop_length, self.hop_length + pad_remainder))

        weight = self.enc_kernel_sin if sin else self.enc_kernel
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            spec = F.conv1d(x_padded.float(), weight, stride=self.hop_length)
        return spec.transpose(1, 2) # (B, n_frames, n_bins)
    
    def _decode_base(self, spec: torch.Tensor, length=None) -> torch.Tensor:
        spec_in = spec.transpose(1, 2) # (B, n_bins, n_frames)
        with torch.amp.autocast(device_type=spec.device.type, enabled=False):
            x_recon_padded = F.conv_transpose1d(spec_in.float(), self.dec_kernel, stride=self.hop_length)
        x_recon_padded = x_recon_padded.squeeze(1)
        
        if length is not None:
            return x_recon_padded[:, self.hop_length : self.hop_length + length]
        return x_recon_padded[:, self.hop_length : -self.hop_length]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = self._encode_base(x)
        spec_sin = self._encode_base(x, sin=True)
        energy = spec.pow(2) + spec_sin.pow(2)
        
        with torch.amp.autocast(device_type=spec.device.type, enabled=False):
            band_energy = torch.matmul(energy, self.filter_banks).clamp(min=1e-6)
            log_mag = 0.5 * torch.log(band_energy)
            envelop = torch.matmul(log_mag, self.interp_matrix)
            norm_spec = spec * torch.exp(-envelop)

        feats = torch.cat((log_mag, norm_spec), dim=-1)
        return feats

    def inverse(self, feats: torch.Tensor, length: int = None) -> torch.Tensor:
        log_mag, norm_spec = torch.split(feats, [self.n_bands, self.n_bins], dim=-1)

        with torch.amp.autocast(device_type=feats.device.type, enabled=False):
            envelop = torch.matmul(log_mag, self.interp_matrix)
            spec = norm_spec * torch.exp(envelop)

        x_recon = self._decode_base(spec, length=length)
        return x_recon
    
if __name__ == "__main__":

    input_wav_path = "assets/test.wav"
    npy_save_path = "features.npy"
    output_wav_path = "rebuild.wav"
    
    model = BN_MDCT_Spectrogram()
    model.eval() 
    
    print("\n>>> 开始编解码测试流程...")
    
    waveform, sample_rate = torchaudio.load(input_wav_path)
    original_length = waveform.shape[1]
    print(f"[Step 1] 读取音频成功 | 形状: {waveform.shape} | 采样率: {sample_rate}")
    
    with torch.no_grad():
        # 提取特征
        feats = model(waveform)
    print(f"[Step 1] MDCT特征提取成功 | 特征形状: {feats.shape}")
    
    feats_np = feats.cpu().numpy()
    np.save(npy_save_path, feats_np)
    print(f"[Step 2] 特征已成功保存至根目录: {npy_save_path}")
    
    loaded_feats_np = np.load(npy_save_path)
    loaded_feats = torch.from_numpy(loaded_feats_np)
    print(f"[Step 3] 从文件读取特征成功 | 形状: {loaded_feats.shape}")
    
    with torch.no_grad():
        rebuilt_waveform = model.inverse(loaded_feats, length=original_length)
    
    # 计算重建精度损失
    mse_loss = torch.nn.MSELoss()(rebuilt_waveform, waveform)
    print(f"[Step 4] 重建精度损失 (MSE): {mse_loss.item():.6e}")
    
    # 计算信噪比 (SNR)
    signal_power = torch.mean(waveform ** 2)
    noise_power = torch.mean((rebuilt_waveform - waveform) ** 2)
    snr_db = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    print(f"[Step 4] 重建信噪比 (SNR): {snr_db.item():.2f} dB")
    
    # 保存重建后的音频
    torchaudio.save(output_wav_path, rebuilt_waveform, sample_rate)
    print(f"[Step 5] 音频已成功复原并保存至根目录: {output_wav_path} | 复原形状: {rebuilt_waveform.shape}")
    
    print("\n>>> 🎉 测试全部顺利完成！")
