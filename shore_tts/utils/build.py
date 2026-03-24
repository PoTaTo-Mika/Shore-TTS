from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from shore_tts.datasets.dataset import build_dataloader
from shore_tts.models.cfm import CFM
from shore_tts.models.dit import DiT
from shore_tts.text.tokenizer import PinyinTokenizer


def load_json_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_config(path: str, config: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")


def set_seed(seed: int, rank: int = 0) -> None:
    seed = int(seed) + int(rank)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed(backend: str = "nccl") -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if distributed and not dist.is_initialized():
        dist.init_process_group(backend=backend)

    return distributed, rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def get_device(local_rank: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def load_mdct_config(path: str) -> dict[str, Any]:
    return load_json_config(path)


def get_mdct_feature_config(path: str) -> dict[str, int]:
    cfg = load_mdct_config(path)
    mdct_params = cfg.get("mdct_params", {})
    hop_length = int(mdct_params.get("hop_length", 441))
    n_bands = int(mdct_params.get("n_bands", 10))
    return {
        "hop_length": hop_length,
        "n_bands": n_bands,
        "spec_dim": hop_length + n_bands,
        "sample_rate": int(cfg.get("sample_rate", 44100)),
    }


def build_model(config: dict[str, Any], device: torch.device) -> CFM:
    mdct_cfg_path = config["data"]["mdct_config"]
    feature_cfg = get_mdct_feature_config(mdct_cfg_path)
    text_cfg = config.get("text", {})
    tokenizer_path = text_cfg.get("tokenizer_path")
    if not tokenizer_path:
        raise ValueError("Missing `text.tokenizer_path` in config. Shore-TTS now requires a pinyin tokenizer vocab.")

    tokenizer = PinyinTokenizer.load(
        tokenizer_path,
        polyphone=bool(text_cfg.get("polyphone", True)),
    )

    dit_cfg = dict(config["model"]["dit"])
    dit_cfg.setdefault("spec_dim", feature_cfg["spec_dim"])
    dit_cfg["text_num_embeds"] = tokenizer.vocab_size

    transformer = DiT(**dit_cfg)
    cfm_cfg = dict(config["model"].get("cfm", {}))
    cfm_cfg["num_channels"] = feature_cfg["spec_dim"]
    cfm_cfg["spec_kwargs"] = {
        "hop_length": feature_cfg["hop_length"],
        "n_bands": feature_cfg["n_bands"],
        "target_sample_rate": feature_cfg["sample_rate"],
    }
    cfm_cfg["vocab_char_map"] = tokenizer.token_to_id
    cfm_cfg["text_tokenizer"] = tokenizer

    model = CFM(transformer=transformer, **cfm_cfg)
    return model.to(device)


def wrap_ddp(model: torch.nn.Module, device: torch.device, distributed: bool) -> torch.nn.Module:
    if not distributed:
        return model

    ddp_kwargs = {}
    if device.type == "cuda":
        ddp_kwargs["device_ids"] = [device.index]
        ddp_kwargs["output_device"] = device.index
    return DDP(model, **ddp_kwargs)


def build_optimizer(config: dict[str, Any], model: torch.nn.Module) -> AdamW:
    optim_cfg = config["optim"]
    return AdamW(
        model.parameters(),
        lr=float(optim_cfg.get("lr", 2e-4)),
        betas=tuple(optim_cfg.get("betas", [0.9, 0.95])),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
    )


def build_scheduler(config: dict[str, Any], optimizer: AdamW) -> LambdaLR:
    sched_cfg = config.get("scheduler", {})
    warmup_steps = int(sched_cfg.get("warmup_steps", 0))
    min_lr_scale = float(sched_cfg.get("min_lr_scale", 0.1))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(step + 1, 1) / warmup_steps
        return min_lr_scale

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_train_dataloader(
    config: dict[str, Any],
    rank: int = 0,
    world_size: int = 1,
):
    data_cfg = config["data"]
    return build_dataloader(
        data_path=data_cfg["data_path"],
        batch_size=int(data_cfg.get("batch_size", 8)),
        config_path=data_cfg["mdct_config"],
        sample_rate=data_cfg.get("sample_rate"),
        hop_length=data_cfg.get("hop_length"),
        n_bands=data_cfg.get("n_bands"),
        min_length=int(data_cfg.get("min_length", 10)),
        max_length=int(data_cfg.get("max_length", 1000)),
        shuffle_buffer=int(data_cfg.get("shuffle_buffer", 1000)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        epoch_shuffle=bool(data_cfg.get("epoch_shuffle", True)),
        rank=rank,
        world_size=world_size,
    )


def checkpoint_state(
    model: torch.nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    raw_model = model.module if isinstance(model, DDP) else model
    return {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
    }


def save_checkpoint(
    save_dir: str,
    state: dict[str, Any],
    filename: str,
) -> str:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    return path


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: AdamW | None = None,
    scheduler: LambdaLR | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return int(checkpoint.get("epoch", 0)), int(checkpoint.get("global_step", 0))


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(device, non_blocking=True)
        else:
            output[key] = value
    return output


def is_main_process(rank: int) -> bool:
    return rank == 0
