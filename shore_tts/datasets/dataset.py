import io
import json
import os
import random
from pathlib import Path
from typing import List, Optional

import torch
import torchaudio
from torch.utils.data import DataLoader, IterableDataset

import webdataset as wds

from shore_tts.utils.spectrogram import BN_MDCT_Spectrogram

_DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "mdct.json")

_AUDIO_KEYS = ("wav", "flac", "mp3", "m4a", "ogg", "opus")


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _discover_tar_files(data_path: str) -> List[str]:
    root = Path(data_path)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {data_path}")
    tar_files = sorted(str(p) for p in root.rglob("*.tar") if p.is_file())
    if not tar_files:
        raise FileNotFoundError(f"No *.tar files found under: {data_path}")
    return tar_files


def _assign_tar_files_by_size(tar_files: List[str], world_size: int) -> List[List[str]]:
    if world_size <= 1:
        return [list(tar_files)]
    shards = sorted(((p, os.path.getsize(p)) for p in tar_files), key=lambda x: x[1], reverse=True)
    assignments: List[List[str]] = [[] for _ in range(world_size)]
    sizes = [0] * world_size
    for path, size in shards:
        r = min(range(world_size), key=lambda i: (sizes[i], len(assignments[i]), i))
        assignments[r].append(path)
        sizes[r] += size
    return assignments


def _collate_batch(samples: List[dict]) -> dict:
    """零填充组批。返回 specs(B,T,F), lengths(B), mask(B,T), texts。"""
    specs = [s["spec"] for s in samples]
    texts = [s["text"] for s in samples]
    lengths = torch.tensor([s.shape[0] for s in specs], dtype=torch.long)
    T_max = int(lengths.max())
    F = specs[0].shape[1]
    padded = torch.zeros(len(specs), T_max, F)
    for i, sp in enumerate(specs):
        padded[i, : sp.shape[0]] = sp
    mask = torch.arange(T_max).unsqueeze(0) < lengths.unsqueeze(1)
    return {"specs": padded, "lengths": lengths, "mask": mask, "texts": texts}


class ShoreDataset(IterableDataset):

    def __init__(
        self,
        data_path: str,
        config_path: str = _DEFAULT_CONFIG,
        sample_rate: Optional[int] = None,
        hop_length: Optional[int] = None,
        n_bands: Optional[int] = None,
        min_length: int = 10,
        max_length: int = 1000,
        batch_size: int = 32,
        epoch_shuffle: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        cfg = _load_config(config_path)
        mdct = cfg.get("mdct_params", {})

        self.sample_rate = sample_rate or cfg.get("sample_rate", 22050)
        self.min_length = min_length
        self.max_length = max_length
        self.batch_size = batch_size
        self.epoch_shuffle = epoch_shuffle
        self.rank = rank
        self.world_size = world_size
        self._epoch = 0

        self.tar_files = _discover_tar_files(data_path)

        self.mdct = BN_MDCT_Spectrogram(
            hop_length=hop_length or mdct.get("hop_length", 1024),
            n_bands=n_bands or mdct.get("n_bands", 20),
        )
        self.mdct.eval()

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    @staticmethod
    def _extract_text(sample: dict) -> Optional[str]:
        """兼容 .txt 与 .json（Emilia）标注，提取 text 字段。"""
        if "txt" in sample:
            raw = sample["txt"]
            return raw.decode("utf-8").strip() if isinstance(raw, bytes) else str(raw).strip()
        if "json" in sample:
            raw = sample["json"]
            try:
                meta = json.loads(raw) if isinstance(raw, (bytes, str)) else raw if isinstance(raw, dict) else None
            except (json.JSONDecodeError, UnicodeDecodeError):
                return None
            return (meta.get("text") or "").strip() or None if meta else None
        return None

    def _decode_sample(self, sample: dict) -> Optional[dict]:
        """解析 + 解码 + MDCT + 帧长过滤，返回 {"spec": (T,F), "text": str} 或 None。"""
        audio_bytes = next((sample[k] for k in _AUDIO_KEYS if k in sample), None)
        if audio_bytes is None:
            return None
        text = self._extract_text(sample)
        if not text:
            return None

        try:
            waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
        except Exception:
            return None
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        with torch.no_grad():
            spec = self.mdct(waveform).squeeze(0)  # (T, F)
        if not (self.min_length <= spec.shape[0] <= self.max_length):
            return None
        return {"spec": spec, "text": text}

    def __iter__(self):
        tar_files = list(self.tar_files)
        if self.epoch_shuffle:
            random.Random(self._epoch).shuffle(tar_files)

        if self.world_size > 1:
            tar_files = _assign_tar_files_by_size(tar_files, self.world_size)[self.rank]

        wi = torch.utils.data.get_worker_info()
        if wi is not None:
            tar_files = tar_files[wi.id :: wi.num_workers]
        if not tar_files:
            return

        # 纯流式：逐样本从 tar 流中读取，攒够 batch_size 就 yield
        stream = wds.WebDataset(tar_files, shardshuffle=len(tar_files) if self.epoch_shuffle else 0)
        buf: List[dict] = []
        for raw in stream:
            decoded = self._decode_sample(raw)
            if decoded is None:
                continue
            buf.append(decoded)
            if len(buf) >= self.batch_size:
                yield _collate_batch(buf)
                buf = []

        # 收尾：不足一个 batch 的剩余样本也 yield
        if buf:
            yield _collate_batch(buf)


def collate_fn(batch):
    """DataLoader collate：dataset 已 yield 组好的 batch dict，直接取出。"""
    return batch[0]


def build_dataloader(
    data_path: str,
    config_path: str = _DEFAULT_CONFIG,
    sample_rate: Optional[int] = None,
    hop_length: Optional[int] = None,
    n_bands: Optional[int] = None,
    min_length: int = 10,
    max_length: int = 1000,
    batch_size: int = 32,
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
        batch_size=batch_size,
        epoch_shuffle=epoch_shuffle,
        rank=rank,
        world_size=world_size,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
