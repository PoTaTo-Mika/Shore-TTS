import io
import json
import os
import random
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torchaudio
from torch.utils.data import DataLoader, IterableDataset

from shore_tts.utils.spectrogram import BN_MDCT_Spectrogram

_DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "mdct.json")

# webdataset 按扩展名作为 key，列出所有支持的格式
_AUDIO_KEYS = ("wav", "flac", "mp3", "m4a", "ogg", "opus")


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _discover_tar_files(data_path: str) -> List[str]:
    root = Path(data_path)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {data_path}")

    tar_files = sorted(str(path) for path in root.rglob("*.tar") if path.is_file())
    if not tar_files:
        raise FileNotFoundError(f"No *.tar files found under: {data_path}")
    return tar_files


def _assign_tar_files_by_size(tar_files: List[str], world_size: int) -> List[List[str]]:
    if world_size <= 1:
        return [list(tar_files)]

    shard_info = [(path, os.path.getsize(path)) for path in tar_files]
    shard_info.sort(key=lambda item: item[1], reverse=True)

    assignments: List[List[str]] = [[] for _ in range(world_size)]
    bucket_sizes = [0 for _ in range(world_size)]

    for path, size in shard_info:
        rank = min(range(world_size), key=lambda idx: (bucket_sizes[idx], len(assignments[idx]), idx))
        assignments[rank].append(path)
        bucket_sizes[rank] += size

    return assignments


class ShoreDataset(IterableDataset):
    """
    Streaming dataset for Shore-TTS.

    Reads paired (audio, .txt annotation) samples from tar shards,
    computes BN-MDCT D-spectrum on the fly, applying frame-length filtering.

    Tar内部格式：
        <key>.<ext>  —— 音频文件（wav/flac/mp3/m4a/ogg/opus）
        <key>.txt    —— 对应的文本标注（UTF-8）

    Args:
        data_path:      包含 *.tar 分片的目录，支持递归扫描子目录。
        config_path:    mdct.json 路径，默认读取 shore_tts/configs/mdct.json。
        sample_rate:    目标采样率（覆盖 config）。
        hop_length:     MDCT hop length（覆盖 config）。
        n_bands:        频带数量（覆盖 config）。
        min_length:     保留的最小帧数（含）。
        max_length:     保留的最大帧数（含）。
        shuffle_buffer: 内存内 shuffle 缓冲区大小。
        epoch_shuffle:  是否在每个 epoch 打乱分片顺序。
    """

    def __init__(
        self,
        data_path: str,
        config_path: str = _DEFAULT_CONFIG,
        sample_rate: Optional[int] = None,
        hop_length: Optional[int] = None,
        n_bands: Optional[int] = None,
        min_length: int = 10,
        max_length: int = 1000,
        shuffle_buffer: int = 1000,
        epoch_shuffle: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()

        cfg = _load_config(config_path)
        mdct_params = cfg.get("mdct_params", {})

        self.sample_rate   = sample_rate or cfg.get("sample_rate", 22050)
        self.min_length    = min_length
        self.max_length    = max_length
        self.shuffle_buffer = shuffle_buffer
        self.epoch_shuffle  = epoch_shuffle
        self.rank = rank
        self.world_size = world_size
        self._epoch = 0

        self.tar_files = _discover_tar_files(data_path)

        # BN_MDCT 只含 buffer，无可学习参数，在 CPU 上运行
        self.mdct = BN_MDCT_Spectrogram(
            hop_length = hop_length or mdct_params.get("hop_length", 1024),
            n_bands    = n_bands    or mdct_params.get("n_bands", 20),
        )
        self.mdct.eval()

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _iter_tar_samples(self, tar_path: str) -> Iterator[dict]:
        pending: Dict[str, dict] = defaultdict(dict)

        with tarfile.open(tar_path, mode="r:*") as archive:
            for member in archive:
                if not member.isreg():
                    continue

                base_name = Path(member.name).name
                key, ext = os.path.splitext(base_name)
                suffix = ext.lstrip(".").lower()
                if suffix not in _AUDIO_KEYS and suffix != "txt":
                    continue

                extracted = archive.extractfile(member)
                if extracted is None:
                    continue

                try:
                    data = extracted.read()
                finally:
                    extracted.close()

                sample = pending[key]
                sample["__key__"] = key
                sample[suffix] = data

                has_audio = any(audio_key in sample for audio_key in _AUDIO_KEYS)
                if has_audio and "txt" in sample:
                    yield dict(sample)
                    del pending[key]

    def _iter_samples(self, tar_files: Iterable[str], rng: random.Random) -> Iterator[Tuple[torch.Tensor, str, float, float]]:
        shuffle_buffer: List[Tuple[torch.Tensor, str, float, float]] = []

        for tar_path in tar_files:
            for sample in self._iter_tar_samples(tar_path):
                item = self._decode(sample)
                if not self._valid_length(item):
                    continue

                if self.shuffle_buffer <= 1:
                    yield item
                    continue

                shuffle_buffer.append(item)
                if len(shuffle_buffer) >= self.shuffle_buffer:
                    index = rng.randrange(len(shuffle_buffer))
                    yield shuffle_buffer.pop(index)

        while shuffle_buffer:
            index = rng.randrange(len(shuffle_buffer))
            yield shuffle_buffer.pop(index)

    def _decode(self, sample: dict) -> Tuple[torch.Tensor, str, float, float]:
        """将 tar 中的音频字节解码，在线提取 D谱，返回 (spec, text)。"""
        io_start = time.perf_counter()
        audio_bytes = next((sample[k] for k in _AUDIO_KEYS if k in sample), None)
        if audio_bytes is None:
            raise KeyError(f"tar sample 中未找到支持的音频格式: {_AUDIO_KEYS}")

        waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))

        # 混音为单声道 (1, T)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 按需重采样
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        text = sample["txt"].decode("utf-8").strip()
        io_time = time.perf_counter() - io_start

        mdct_start = time.perf_counter()
        with torch.no_grad():
            spec = self.mdct(waveform)  # (1, T, F)
        mdct_time = time.perf_counter() - mdct_start

        return spec.squeeze(0), text, io_time, mdct_time   # (T, F)

    def _valid_length(self, item: Tuple[torch.Tensor, str, float, float]) -> bool:
        T = item[0].shape[0]
        return self.min_length <= T <= self.max_length

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        tar_files = list(self.tar_files)
        rng = random.Random(self._epoch)
        if self.epoch_shuffle:
            rng.shuffle(tar_files)

        if self.world_size > 1:
            assignments = _assign_tar_files_by_size(tar_files, self.world_size)
            tar_files = assignments[self.rank]

        if worker_info is not None:
            tar_files = tar_files[worker_info.id :: worker_info.num_workers]

        worker_seed = self._epoch * 9973 + self.rank
        if worker_info is not None:
            worker_seed = worker_seed * 31 + worker_info.id
        sample_rng = random.Random(worker_seed)

        for spec, text, io_time, mdct_time in self._iter_samples(tar_files, sample_rng):
            yield spec.float(), text, io_time, mdct_time


def collate_fn(batch: List[Tuple[torch.Tensor, str, float, float]]) -> dict:
    """
    将 (spec, text) 列表整理成零填充批次。

    Returns:
        specs:   FloatTensor (B, T_max, F)  — 零填充 D谱
        lengths: LongTensor  (B,)           — 各样本真实帧数
        mask:    BoolTensor  (B, T_max)     — True 表示有效帧
        texts:   list[str]                  — 原始文本标注
    """
    specs, texts, io_times, mdct_times = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in specs], dtype=torch.long)
    T_max = int(lengths.max())
    F = specs[0].shape[1]

    padded = torch.zeros(len(specs), T_max, F)
    for i, spec in enumerate(specs):
        padded[i, : spec.shape[0]] = spec

    mask = torch.arange(T_max).unsqueeze(0) < lengths.unsqueeze(1)  # (B, T_max)

    return {
        "specs":   padded,       # (B, T_max, F)
        "lengths": lengths,      # (B,)
        "mask":    mask,         # (B, T_max)
        "texts":   list(texts),  # list[str]
        "io_time": torch.tensor(io_times, dtype=torch.float32).mean(),
        "mdct_time": torch.tensor(mdct_times, dtype=torch.float32).mean(),
    }


def build_dataloader(
    data_path: str,
    batch_size: int,
    config_path: str = _DEFAULT_CONFIG,
    sample_rate: Optional[int] = None,
    hop_length: Optional[int] = None,
    n_bands: Optional[int] = None,
    min_length: int = 10,
    max_length: int = 1000,
    shuffle_buffer: int = 1000,
    num_workers: int = 4,
    epoch_shuffle: bool = True,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:

    dataset = ShoreDataset(
        data_path=data_path,
        config_path=config_path,
        sample_rate=sample_rate,
        hop_length=hop_length,
        n_bands=n_bands,
        min_length=min_length,
        max_length=max_length,
        shuffle_buffer=shuffle_buffer,
        epoch_shuffle=epoch_shuffle,
        rank=rank,
        world_size=world_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


if __name__ == "__main__":
    import tarfile

    cfg = _load_config(_DEFAULT_CONFIG)
    SAMPLE_RATE = cfg.get("sample_rate", 22050)
    data_path = "assets"

    test_tar = os.path.join(data_path, "_test_shard.tar")
    os.makedirs(data_path, exist_ok=True)
    try:
        discovered = _discover_tar_files(data_path)
    except FileNotFoundError:
        discovered = []

    if not discovered:
        print("[setup] 未找到 tar 文件，生成随机测试 shard ...")
        with tarfile.open(test_tar, "w") as tf:
            for idx in range(8):
                # 生成随机时长 (1~5 秒) 的单声道音频
                duration = torch.randint(1, 6, (1,)).item()
                waveform = torch.randn(1, SAMPLE_RATE * duration) * 0.1
                text = f"测试样本 {idx}"

                # wav → bytes
                wav_buf = io.BytesIO()
                torchaudio.save(wav_buf, waveform, SAMPLE_RATE, format="wav")
                wav_bytes = wav_buf.getvalue()
                info = tarfile.TarInfo(name=f"{idx:04d}.wav")
                info.size = len(wav_bytes)
                tf.addfile(info, io.BytesIO(wav_bytes))

                # txt → bytes
                txt_bytes = text.encode("utf-8")
                info = tarfile.TarInfo(name=f"{idx:04d}.txt")
                info.size = len(txt_bytes)
                tf.addfile(info, io.BytesIO(txt_bytes))

        print(f"[setup] 已写入: {test_tar}")

    print(f"\n[test] data_path={data_path!r}  sample_rate={SAMPLE_RATE}  batch_size=4")
    loader = build_dataloader(
        data_path=data_path,
        batch_size=4,
        min_length=10,
        max_length=1024,
        shuffle_buffer=16,
        num_workers=0,
    )

    batch = next(iter(loader))
    print(f"  specs   shape : {batch['specs'].shape}   dtype={batch['specs'].dtype}")
    print(f"  lengths       : {batch['lengths'].tolist()}")
    print(f"  mask    shape : {batch['mask'].shape}")
    print(f"  texts         : {batch['texts']}")
    print(f"  io_time       : {float(batch['io_time']):.6f}s")
    print(f"  mdct_time     : {float(batch['mdct_time']):.6f}s")
    print("\n[test] OK")
