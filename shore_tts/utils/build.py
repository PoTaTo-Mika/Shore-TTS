from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
import torchaudio
from tqdm.auto import tqdm

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
    fused = bool(optim_cfg.get("fused", True)) and torch.cuda.is_available()
    return AdamW(
        model.parameters(),
        lr=float(optim_cfg.get("lr", 2e-4)),
        betas=tuple(optim_cfg.get("betas", [0.9, 0.95])),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
        fused=fused,
    )


def build_scheduler(config: dict[str, Any], optimizer: AdamW):
    sched_cfg = config.get("scheduler", {})
    warmup_steps = int(sched_cfg.get("warmup_steps", 0))
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


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {}
        self._init_from_model(model)

    def _init_from_model(self, model: torch.nn.Module) -> None:
        for name, param in model.state_dict().items():
            if torch.is_tensor(param):
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        state_dict = model.state_dict()
        for name, param in state_dict.items():
            if not torch.is_tensor(param):
                continue
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
                continue
            ema_param = self.shadow[name]
            ema_param.lerp_(param.detach().to(device=ema_param.device, dtype=ema_param.dtype), 1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow": {name: tensor.clone() for name, tensor in self.shadow.items()},
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.decay = float(state_dict.get("decay", self.decay))
        self.shadow = {name: tensor.clone() for name, tensor in state_dict["shadow"].items()}

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)


def build_precision_components(
    config: dict[str, Any],
    device: torch.device,
) -> tuple[str, bool, torch.dtype | None, torch.amp.GradScaler | None]:
    train_cfg = config["train"]
    precision = str(train_cfg.get("precision", "fp16")).lower()
    allow_tf32 = bool(train_cfg.get("allow_tf32", True))

    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("`train.precision` must be one of: fp32, fp16, bf16.")
    if device.type != "cuda" and precision != "fp32":
        raise ValueError("Mixed precision training requires CUDA.")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")

    autocast_dtype = None
    scaler = None
    if precision == "fp16":
        autocast_dtype = torch.float16
        scaler = torch.amp.GradScaler("cuda")
    elif precision == "bf16":
        autocast_dtype = torch.bfloat16

    return precision, allow_tf32, autocast_dtype, scaler


def build_train_writer(config: dict[str, Any], rank: int) -> SummaryWriter | None:
    train_cfg = config["train"]
    if not is_main_process(rank):
        return None
    if not train_cfg.get("tensorboard", {}).get("enabled", True):
        return None

    log_dir = train_cfg.get("tensorboard", {}).get("log_dir", os.path.join(train_cfg["save_dir"], "tensorboard"))
    return SummaryWriter(log_dir=log_dir)


def log_train_setup(
    model: torch.nn.Module,
    rank: int,
    precision: str,
    allow_tf32: bool,
    device: torch.device,
) -> None:
    if not is_main_process(rank):
        return

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tqdm.write(f"[model] total_params={total_params:,} trainable_params={trainable_params:,}")
    tqdm.write(f"[train] precision={precision} tf32={allow_tf32} device={device}")


def resume_training_state(
    config: dict[str, Any],
    model: torch.nn.Module,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    rank: int,
    ema: EMA | None = None,
) -> tuple[int, int]:
    resume_from = config["train"].get("resume_from")
    if not resume_from:
        return 0, 0

    start_epoch, global_step = load_checkpoint(
        resume_from,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        map_location=device,
        ema=ema,
    )
    if is_main_process(rank):
        tqdm.write(f"[resume] epoch={start_epoch} global_step={global_step} ckpt={resume_from}")
    return start_epoch, global_step


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
    scheduler,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    ema: EMA | None = None,
) -> dict[str, Any]:
    raw_model = model.module if isinstance(model, DDP) else model
    state = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
    }
    if ema is not None:
        state["ema"] = ema.state_dict()
    return state


def save_checkpoint(
    save_dir: str,
    state: dict[str, Any],
    filename: str,
    keep_last_n: int = -1,
) -> str:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    if keep_last_n >= 0 and filename.startswith("step_") and filename.endswith(".pt"):
        rotate_checkpoints(save_dir, keep_last_n)
    return path


def rotate_checkpoints(save_dir: str, keep_last_n: int) -> None:
    if keep_last_n < 0:
        return
    ckpts = sorted(Path(save_dir).glob("step_*.pt"))
    while len(ckpts) > keep_last_n:
        ckpts[0].unlink(missing_ok=True)
        ckpts.pop(0)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: AdamW | None = None,
    scheduler = None,
    map_location: str | torch.device = "cpu",
    ema: EMA | None = None,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if ema is not None and "ema" in checkpoint:
        ema.load_state_dict(checkpoint["ema"])

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


@torch.no_grad()
def save_sample_outputs(
    model: torch.nn.Module,
    ema: EMA | None,
    batch: dict[str, Any],
    config: dict[str, Any],
    save_dir: str,
    global_step: int,
    device: torch.device,
) -> list[str]:
    sample_cfg = config["train"].get("log_samples", {})
    if not sample_cfg.get("enabled", False):
        return []

    raw_model = model.module if isinstance(model, DDP) else model
    original_state = None
    if ema is not None:
        original_state = {k: v.detach().clone() for k, v in raw_model.state_dict().items()}
        ema.copy_to(raw_model)

    was_training = raw_model.training
    raw_model.eval()

    sample_dir = Path(save_dir) / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    sample_index = int(sample_cfg.get("sample_index", 0))
    sample_index = min(sample_index, batch["specs"].shape[0] - 1)
    ref_spec = batch["specs"][sample_index : sample_index + 1].to(device)
    ref_len = int(batch["lengths"][sample_index].item())
    ref_text = batch["texts"][sample_index]
    duration = int(max(ref_len * float(sample_cfg.get("duration_factor", 2.0)), ref_len + 1))
    sample_steps = int(sample_cfg.get("sample_steps", 16))
    cfg_strength = float(sample_cfg.get("cfg_strength", 1.0))

    generated, _ = raw_model.sample(
        cond=ref_spec[:, :ref_len],
        text=[f"{ref_text} {ref_text}"],
        duration=duration,
        steps=sample_steps,
        cfg_strength=cfg_strength,
        lens=torch.tensor([ref_len], device=device, dtype=torch.long),
    )
    generated = generated[:, ref_len:, :]
    gen_audio = raw_model.spec.inverse(generated, length=generated.shape[1] * raw_model.spec.hop_length).cpu()
    ref_audio = raw_model.spec.inverse(
        ref_spec[:, :ref_len],
        length=ref_len * raw_model.spec.hop_length,
    ).cpu()

    sample_rate = int(
        config["data"].get("sample_rate") or get_mdct_feature_config(config["data"]["mdct_config"])["sample_rate"]
    )
    ref_path = sample_dir / f"step_{global_step:08d}_ref.wav"
    gen_path = sample_dir / f"step_{global_step:08d}_gen.wav"
    torchaudio.save(str(ref_path), ref_audio, sample_rate)
    torchaudio.save(str(gen_path), gen_audio, sample_rate)

    if original_state is not None:
        raw_model.load_state_dict(original_state, strict=True)
    if was_training:
        raw_model.train()

    return [str(ref_path), str(gen_path)]
