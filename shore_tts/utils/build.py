from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from accelerate import Accelerator

from shore_tts.datasets.dataset import build_dataloader
from shore_tts.models.diffusion.cfm import CFM
from shore_tts.models.diffusion.dit import DiT
from shore_tts.models.gan.fwd import FastWaveD
from shore_tts.optimizer.muon import Muon_AdamW
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


def get_mdct_feature_config(path: str) -> dict[str, int]:
    cfg = load_json_config(path)
    mdct_params = cfg.get("mdct_params", {})
    hop_length = int(mdct_params.get("hop_length", 441))
    n_bands = int(mdct_params.get("n_bands", 10))
    return {
        "hop_length": hop_length,
        "n_bands": n_bands,
        "spec_dim": hop_length + n_bands,
        "sample_rate": int(cfg.get("sample_rate", 44100)),
    }


def build_model(config: dict[str, Any]) -> CFM:
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
    return model

def build_discriminator(config: dict[str, Any]) -> FastWaveD:
    fwd_cfg = dict(config["model"]["fwd"])
    return FastWaveD(**fwd_cfg)

def build_optimizer(config: dict[str, Any], model: torch.nn.Module):
    optim_cfg = config["optim"]
    optimizer_type = optim_cfg.get("optimizer_type", "adamw")
    lr = float(optim_cfg.get("lr", 2e-4))
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))

    if optimizer_type == "adamw":
        print("Using AdamW Optimizer...")
        fused = bool(optim_cfg.get("fused", True)) and torch.cuda.is_available()
        return AdamW(
            model.parameters(),
            lr=lr,
            betas=tuple(optim_cfg.get("betas", [0.9, 0.95])),
            weight_decay=weight_decay,
            fused=fused,
        )
    elif optimizer_type == "muon_adamw":
        print("Using Muon Optimizer...")
        muon_args = optim_cfg.get("muon_args", {})
        adamw_args = optim_cfg.get("adamw_args", {})
        return Muon_AdamW(
            model,
            lr=lr,
            weight_decay=weight_decay,
            muon_args=muon_args,
            adamw_args=adamw_args,
        )
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type!r}, please choose one of [muon_adamw] and [adamw]")


def build_scheduler(config: dict[str, Any], optimizer, num_processes: int = 1):
    sched_cfg = config.get("scheduler", {})
    warmup_steps = int(sched_cfg.get("warmup_steps", 0)) * num_processes
    final_lr_scale = float(sched_cfg.get("final_lr_scale", sched_cfg.get("min_lr_scale", 0.1)))
    total_steps = int(config.get("train", {}).get("max_steps", -1))
    total_steps = int(sched_cfg.get("total_steps", total_steps))

    if total_steps <= 0:
        total_steps = warmup_steps + 1
    warmup_steps = min(warmup_steps, max(total_steps - 1, 0))
    decay_steps = max(total_steps - warmup_steps, 1)

    schedulers = []
    milestones = []

    if warmup_steps > 0:
        schedulers.append(
            LinearLR(
                optimizer,
                start_factor=max(float(sched_cfg.get("warmup_start_factor", 1e-8)), 1e-8),
                end_factor=1.0,
                total_iters=warmup_steps,
            )
        )
        milestones.append(warmup_steps)

    schedulers.append(
        LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=final_lr_scale,
            total_iters=decay_steps,
        )
    )

    if len(schedulers) == 1:
        return schedulers[0]
    return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)


def build_train_dataloader(config: dict[str, Any], accelerator: Accelerator):
    data_cfg = config["data"]
    return build_dataloader(
        data_path=data_cfg["data_path"],
        config_path=data_cfg["mdct_config"],
        sample_rate=data_cfg.get("sample_rate"),
        hop_length=data_cfg.get("hop_length"),
        n_bands=data_cfg.get("n_bands"),
        min_length=int(data_cfg.get("min_length", 10)),
        max_length=int(data_cfg.get("max_length", 1000)),
        batch_size=int(data_cfg.get("batch_size", 32)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        epoch_shuffle=bool(data_cfg.get("epoch_shuffle", True)),
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
