import bisect
import io
import json
import math
import os
import random
from pathlib import Path
from typing import List, Optional

import torch
import torchaudio
from torch.utils.data import DataLoader, IterableDataset

import webdataset as wds

_DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "mdct.json")

_AUDIO_KEYS = ("wav", "flac", "mp3", "m4a", "ogg", "opus")


def _identity_splitter(src):
    yield from src


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
    """零填充组批。返回 wavs(B,T_wav), wav_lengths(B), texts。"""
    wavs = [s["wav"] for s in samples]
    texts = [s["text"] for s in samples]
    wav_lengths = torch.tensor([w.shape[0] for w in wavs], dtype=torch.long)
    T_max = int(wav_lengths.max())
    padded = torch.zeros(len(wavs), T_max)
    for i, w in enumerate(wavs):
        padded[i, : w.shape[0]] = w
    return {"wavs": padded, "wav_lengths": wav_lengths, "texts": texts}


class ShoreDataset(IterableDataset):

    def __init__(
        self,
        data_path: str,
        config_path: str = _DEFAULT_CONFIG,
        sample_rate: Optional[int] = None,
        hop_length: Optional[int] = None,
        min_length: int = 10,
        max_length: int = 1000,
        batch_size: int = 32,
        n_buckets: int = 20,
        max_tokens_per_batch: Optional[int] = None,
        epoch_shuffle: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        cfg = _load_config(config_path)
        mdct = cfg.get("mdct_params", {})

        self.sample_rate = sample_rate or cfg.get("sample_rate", 22050)
        self.hop_length = int(hop_length or mdct.get("hop_length", 1024))
        self.min_length = min_length
        self.max_length = max_length
        self.batch_size = batch_size
        self.n_buckets = n_buckets
        self.max_tokens_per_batch = max_tokens_per_batch
        self.epoch_shuffle = epoch_shuffle
        self.rank = rank
        self.world_size = world_size
        self._epoch = 0

        self.tar_files = _discover_tar_files(data_path)

        # Precompute log-spaced bucket boundaries
        log_min = math.log10(max(self.min_length, 1))
        log_max = math.log10(self.max_length)
        step = (log_max - log_min) / self.n_buckets
        self._bucket_boundaries = [10 ** (log_min + i * step) for i in range(self.n_buckets + 1)]

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _assigned_tar_files_for_cycle(self, cycle: int) -> List[str]:
        tar_files = list(self.tar_files)
        if self.epoch_shuffle:
            random.Random(cycle).shuffle(tar_files)

        if self.world_size <= 1:
            return tar_files

        assignments = _assign_tar_files_by_size(tar_files, self.world_size)
        return assignments[(self.rank + cycle) % self.world_size]

    def _find_bucket(self, frame_length: int) -> int:
        idx = bisect.bisect_right(self._bucket_boundaries, frame_length) - 1
        return max(0, min(idx, self.n_buckets - 1))

    @staticmethod
    def _parse_json(sample: dict) -> Optional[dict]:
        if "json" not in sample:
            return None
        raw = sample["json"]
        try:
            return json.loads(raw) if isinstance(raw, (bytes, str)) else raw if isinstance(raw, dict) else None
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
            return None

    def _extract_duration(self, sample: dict) -> Optional[float]:
        meta = self._parse_json(sample)
        if meta and "duration" in meta:
            return float(meta["duration"])
        return None

    def _effective_batch_size(self, bucket: List[dict]) -> int:
        """Dynamic batch size: cap by max_tokens_per_batch if set."""
        if not self.max_tokens_per_batch or not bucket:
            return self.batch_size
        max_wav = max(s["wav"].shape[0] for s in bucket)
        return max(1, self.max_tokens_per_batch // max_wav)

    def _iter_cycle_batches(self, tar_files: List[str]):
        if not tar_files:
            return

        # Rank-level shard assignment is handled above; let WebDataset only split by DataLoader worker.
        stream = wds.WebDataset(
            tar_files,
            shardshuffle=len(tar_files) if self.epoch_shuffle else 0,
            nodesplitter=_identity_splitter,
        )
        buckets: List[List[dict]] = [[] for _ in range(self.n_buckets)]

        for raw in stream:
            # Pre-filter using json duration (skip expensive audio decode)
            duration = self._extract_duration(raw)
            if duration is not None:
                frame_len = int(duration * self.sample_rate / self.hop_length)
                if not (self.min_length <= frame_len <= self.max_length):
                    continue

            decoded = self._decode_sample(raw)
            if decoded is None:
                continue

            frame_len = decoded["wav"].shape[0] // self.hop_length
            idx = self._find_bucket(frame_len)
            buckets[idx].append(decoded)

            eff_bs = self._effective_batch_size(buckets[idx])
            while len(buckets[idx]) >= eff_bs:
                yield _collate_batch(buckets[idx][:eff_bs])
                buckets[idx] = buckets[idx][eff_bs:]
                if buckets[idx]:
                    eff_bs = self._effective_batch_size(buckets[idx])

        # Flush remaining
        for bucket in buckets:
            if bucket:
                yield _collate_batch(bucket)

    @staticmethod
    def _extract_text(sample: dict) -> Optional[str]:
        if "txt" in sample:
            raw = sample["txt"]
            return raw.decode("utf-8").strip() if isinstance(raw, bytes) else str(raw).strip()
        meta = ShoreDataset._parse_json(sample)
        return (meta.get("text") or "").strip() or None if meta else None

    def _decode_sample(self, sample: dict) -> Optional[dict]:
        """解析 + 解码 + 帧长预过滤，返回 {"wav": (T,), "text": str} 或 None。"""
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

        frame_length = waveform.shape[-1] // self.hop_length
        if not (self.min_length <= frame_length <= self.max_length):
            return None

        return {"wav": waveform.squeeze(0), "text": text}

    def __iter__(self):
        cycle = self._epoch
        while True:
            tar_files = self._assigned_tar_files_for_cycle(cycle)
            yield from self._iter_cycle_batches(tar_files)
            cycle += 1


def collate_fn(batch):
    """DataLoader collate：dataset 已 yield 组好的 batch dict，直接取出。"""
    return batch[0]


def build_dataloader(
    data_path: str,
    config_path: str = _DEFAULT_CONFIG,
    sample_rate: Optional[int] = None,
    hop_length: Optional[int] = None,
    min_length: int = 10,
    max_length: int = 1000,
    batch_size: int = 32,
    n_buckets: int = 20,
    max_tokens_per_batch: Optional[int] = None,
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
        max_tokens_per_batch=max_tokens_per_batch,
        min_length=min_length,
        max_length=max_length,
        batch_size=batch_size,
        n_buckets=n_buckets,
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
        prefetch_factor=4,
    )
