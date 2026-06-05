from __future__ import annotations

import contextlib
import gc
import json
import os
import shutil
from pathlib import Path
from typing import Any
import signal

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchaudio
from tqdm.auto import tqdm

from ema_pytorch import EMA

from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, DistributedDataParallelKwargs

from shore_tts.utils.build import (
    build_optimizer,
    build_scheduler,
    build_train_dataloader,
)
from shore_tts.utils.loss import FeatureMatchingLoss


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: dict[str, Any],
    ):
        train_cfg = config["train"]
        optim_cfg = config["optim"]

        # Accelerator
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        precision = str(train_cfg.get("precision", "bf16")).lower()
        mp_map = {"fp32": "no", "fp16": "fp16", "bf16": "bf16"}
        mixed_precision = mp_map.get(precision, "no")
        grad_accum = int(train_cfg.get("grad_accumulation_steps", 1))

        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=grad_accum,
            dataloader_config=DataLoaderConfiguration(
                split_batches=False,
                dispatch_batches=False,
                even_batches=False,
            ),
            kwargs_handlers=[ddp_kwargs],
        )

        # Training params from config
        self.epochs = int(train_cfg["epochs"])
        self.max_steps = int(train_cfg.get("max_steps", -1))
        self.save_per_updates = int(train_cfg.get("save_every_steps", 50000))
        self.last_per_updates = int(train_cfg.get("last_per_updates", 1000))
        self.keep_last_n_checkpoints = int(train_cfg.get("keep_last_n_checkpoints", -1))
        self.log_every_steps = int(train_cfg.get("log_every_steps", 10))
        self.max_grad_norm = float(optim_cfg.get("grad_clip", 1.0))
        self.checkpoint_path = train_cfg.get("save_dir", "checkpoints")
        self.randomize_data_on_resume = bool(train_cfg.get("randomize_data_on_resume", True))
        self.data_cycle_seed = 0
        self.config = config

        # Optimizer & scheduler (before prepare)
        self.optimizer = build_optimizer(config, model)
        self.scheduler = build_scheduler(config, self.optimizer, self.accelerator.num_processes)

        # Prepare model + optimizer with Accelerate (dataloader prepared later in train())
        self.model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        # EMA on main process only (after prepare so model is on correct device)
        # Keep EMA in fp32 to avoid precision loss in bf16 mixed-precision training
        if self.is_main:
            self.ema_model = EMA(self.accelerator.unwrap_model(self.model), include_online_model=False)
            self.ema_model.to(self.accelerator.device, dtype=torch.float32)
        else:
            self.ema_model = None

        # TensorBoard config (writer created lazily in train() so we know resume step)
        self._tb_cfg = train_cfg.get("tensorboard", {})
        self.writer = None

        if self.is_main:
            raw_model = self.accelerator.unwrap_model(self.model)
            total_params = sum(p.numel() for p in raw_model.parameters())
            trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
            print(f"[model] total_params={total_params:,} trainable_params={trainable_params:,}")
            print(f"[train] mixed_precision={self.accelerator.mixed_precision} device={self.accelerator.device}")
            if grad_accum > 1:
                print(f"[train] gradient_accumulation_steps={grad_accum}")

    @property
    def is_main(self) -> bool:
        return self.accelerator.is_main_process

    # ------------------------------------------------------------------
    # Checkpoint: save as folder  checkpoints/step_00050000/
    #   ├── config.json        (training config used)
    #   ├── vocab.json         (tokenizer vocabulary)
    #   └── model.pt           (model + optimizer + scheduler + ema + update)
    # ------------------------------------------------------------------

    def save_checkpoint(self, update: int, epoch: int = 0, last: bool = False) -> None:
        self.accelerator.wait_for_everyone()
        if self.is_main:
            raw_model = self.accelerator.unwrap_model(self.model)
            state = {
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "epoch": epoch,
                "update": update,
                "data_seed": self.data_cycle_seed,
            }
            if self.ema_model is not None:
                state["ema_model_state_dict"] = self.ema_model.state_dict()

            Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)

            if last:
                # model_last.pt — lightweight, just model weights for quick resume
                self.accelerator.save(state, f"{self.checkpoint_path}/model_last.pt")
                print(f"Saved last checkpoint at update {update}")
            else:
                # Full checkpoint folder
                ckpt_dir = f"{self.checkpoint_path}/step_{update:08d}"
                Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

                # 1) Save model weights
                self.accelerator.save(state, f"{ckpt_dir}/model.pt")

                # 2) Save training config
                config_path = os.path.join(ckpt_dir, "config.json")
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)

                # 3) Copy vocab.json from tokenizer_path
                tokenizer_path = self.config.get("text", {}).get("tokenizer_path")
                if tokenizer_path and os.path.exists(tokenizer_path):
                    shutil.copy2(tokenizer_path, os.path.join(ckpt_dir, "vocab.json"))

                print(f"[checkpoint] update={update} saved {ckpt_dir}/")

                # Rotate old checkpoint folders
                if self.keep_last_n_checkpoints > 0:
                    self._rotate_checkpoints()
        self.accelerator.wait_for_everyone()

    def _rotate_checkpoints(self) -> None:
        ckpt_dirs = sorted(
            d for d in Path(self.checkpoint_path).iterdir()
            if d.is_dir() and d.name.startswith("step_")
        )
        while len(ckpt_dirs) > self.keep_last_n_checkpoints:
            oldest = ckpt_dirs.pop(0)
            shutil.rmtree(oldest)
            print(f"Removed old checkpoint: {oldest.name}")

    def load_checkpoint(self) -> int:
        resume_from = self.config["train"].get("resume_from")
        if not resume_from or not os.path.exists(resume_from):
            return 0

        self.accelerator.wait_for_everyone()

        # Support both folder format (step_00050000/model.pt) and legacy single file
        if os.path.isdir(resume_from):
            model_path = os.path.join(resume_from, "model.pt")
            # If resuming from a folder, also reload config if present
            config_path = os.path.join(resume_from, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    saved_config = json.load(f)
                # Update resume config from saved (keep resume_from pointing to this folder)
                self.config.update(saved_config)
                self.config["train"]["resume_from"] = resume_from
        else:
            model_path = resume_from

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

        # Support both new key names and legacy
        model_key = "model_state_dict" if "model_state_dict" in checkpoint else "model"
        raw_model = self.accelerator.unwrap_model(self.model)
        raw_model.load_state_dict(checkpoint[model_key])

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Support both new and legacy EMA keys
        if self.ema_model is not None:
            ema_key = None
            if "ema_model_state_dict" in checkpoint:
                ema_key = "ema_model_state_dict"
            elif "ema" in checkpoint:
                ema_key = "ema"
            if ema_key is not None:
                self.ema_model.load_state_dict(checkpoint[ema_key])

        update = int(checkpoint.get("update", checkpoint.get("global_step", 0)))
        self.data_cycle_seed = int(checkpoint.get("data_seed", update))
        del checkpoint
        gc.collect()

        if self.is_main:
            print(f"[resume] update={update} ckpt={resume_from}")

        return update

    def _broadcast_data_seed(self, seed: int) -> int:
        if self.accelerator.num_processes <= 1:
            return int(seed)
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return int(seed)

        seed_tensor = torch.tensor([seed], device=self.accelerator.device, dtype=torch.long)
        torch.distributed.broadcast(seed_tensor, src=0)
        return int(seed_tensor.item())

    def _resolve_data_seed(self, start_update: int) -> int:
        resume_from = self.config["train"].get("resume_from")
        resumed = bool(resume_from and os.path.exists(resume_from))
        if not resumed:
            return 0
        if not self.randomize_data_on_resume:
            return self.data_cycle_seed

        seed = 0
        if self.is_main:
            seed = int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())
        seed = self._broadcast_data_seed(seed)

        if self.is_main:
            print(f"[resume] randomize_data_on_resume=True data_seed={seed} update={start_update}")

        return seed

    @torch.no_grad()
    def _save_sample_outputs(self, batch: dict, global_update: int) -> list[str]:
        if not self.is_main:
            return []

        sample_cfg = self.config["train"].get("log_samples", {})
        if not sample_cfg.get("enabled", False):
            return []

        max_sample_frames = int(sample_cfg.get("max_sample_frames", 4800))
        raw_model = self.accelerator.unwrap_model(self.model)
        frame_lens = batch["wav_lengths"] // raw_model.spec.hop_length
        eligible = (frame_lens < max_sample_frames).nonzero(as_tuple=False).flatten()
        if eligible.numel() == 0:
            print(
                f"[samples] update={global_update} skipped: "
                f"no sample shorter than {max_sample_frames} frames in current batch"
            )
            return []

        sample_index = int(sample_cfg.get("sample_index", 0))
        sample_index = int(eligible[sample_index % eligible.numel()].item())

        original_state = None
        if self.ema_model is not None:
            original_state = {k: v.detach().clone() for k, v in raw_model.state_dict().items()}
            self.ema_model.copy_params_from_ema_to_model()

        was_training = raw_model.training
        raw_model.eval()

        sample_dir = Path(self.checkpoint_path) / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)

        device = self.accelerator.device
        ref_wav = batch["wavs"][sample_index : sample_index + 1]
        ref_wav_len = int(batch["wav_lengths"][sample_index].item())
        ref_text = batch["texts"][sample_index]
        hop = raw_model.spec.hop_length
        ref_len = (ref_wav_len + hop - 1) // hop + 1
        duration = int(max(ref_len * float(sample_cfg.get("duration_factor", 2.0)), ref_len + 1))
        sample_steps = int(sample_cfg.get("sample_steps", 16))
        cfg_strength = float(sample_cfg.get("cfg_strength", 1.0))

        with torch.inference_mode(), self.accelerator.autocast():
            generated, _ = raw_model.sample(
                cond=ref_wav[:, :ref_wav_len],
                text=[f"{ref_text} {ref_text}"],
                duration=duration,
                steps=sample_steps,
                cfg_strength=cfg_strength,
                lens=torch.tensor([ref_wav_len], device=device, dtype=torch.long),
            )
            generated = generated[:, ref_len:, :]
            gen_audio = raw_model.spec.inverse(
                generated.permute(0, 2, 1), length=generated.shape[1] * hop
            ).cpu()
            ref_audio = ref_wav[:, :ref_wav_len].cpu()

        sample_rate = raw_model.spec.target_sample_rate
        ref_path = str(sample_dir / f"step_{global_update:08d}_ref.wav")
        gen_path = str(sample_dir / f"step_{global_update:08d}_gen.wav")
        torchaudio.save(ref_path, ref_audio, sample_rate)
        torchaudio.save(gen_path, gen_audio, sample_rate)

        if original_state is not None:
            raw_model.load_state_dict(original_state, strict=True)
        if was_training:
            raw_model.train()

        return [ref_path, gen_path]

    def train(self, train_dataloader=None) -> None:
        if train_dataloader is None:
            train_dataloader = build_train_dataloader(self.config, self.accelerator)

        # Prepare dataloader + scheduler with Accelerate
        train_dataloader, self.scheduler = self.accelerator.prepare(train_dataloader, self.scheduler)

        # Load checkpoint
        start_update = self.load_checkpoint()
        global_update = start_update

        # Create TensorBoard writer (after checkpoint load so we know the resume step)
        if self.is_main and self._tb_cfg.get("enabled", True):
            log_dir = self._tb_cfg.get("log_dir", os.path.join(self.checkpoint_path, "tensorboard"))
            purge_step = start_update if start_update > 0 else None
            self.writer = SummaryWriter(log_dir=log_dir, purge_step=purge_step)

        self.data_cycle_seed = self._resolve_data_seed(start_update)
        if hasattr(train_dataloader, "dataset") and hasattr(train_dataloader.dataset, "set_epoch"):
            train_dataloader.dataset.set_epoch(self.data_cycle_seed)

        progress = None
        if self.accelerator.is_local_main_process:
            total = self.max_steps if self.max_steps > 0 else None
            progress = tqdm(
                total=total,
                initial=global_update,
                dynamic_ncols=True,
                mininterval=0.5,
                unit="update",
                desc="train",
            )

        stop_requested = False

        def _handle_sigint(signum, frame):
            nonlocal stop_requested
            if stop_requested:
                raise KeyboardInterrupt
            stop_requested = True
            if self.accelerator.is_local_main_process:
                print("\n[SIGINT] Graceful stop requested. Press Ctrl+C again to force quit.", flush=True)

        old_handler = signal.signal(signal.SIGINT, _handle_sigint)

        try:
            self.model.train()
            train_iterator = iter(train_dataloader)

            while self.max_steps <= 0 or global_update < self.max_steps:
                if stop_requested:
                    break
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_dataloader)
                    continue

                with self.accelerator.accumulate(self.model):
                    loss, _, _, loss_low, loss_high = self.model(
                        inp=batch["wavs"],
                        text=batch["texts"],
                        lens=batch["wav_lengths"],
                    )
                    self.accelerator.backward(loss)

                    grad_norm = 0.0
                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.max_grad_norm if self.max_grad_norm > 0 else float("inf"),
                        ).item()

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    if self.ema_model is not None:
                        self.ema_model.update()
                    global_update += 1

                    if progress is not None:
                        progress.update(1)
                        progress.set_postfix(
                            update=str(global_update),
                            loss=f"{loss.item():.4f}",
                        )

                # TensorBoard logging
                if (
                    self.accelerator.is_local_main_process
                    and self.writer is not None
                    and self.log_every_steps > 0
                    and global_update % self.log_every_steps == 0
                ):
                    self.writer.add_scalar("train/loss", loss.item(), global_update)
                    self.writer.add_scalar("train/loss_low_freq", loss_low.item(), global_update)
                    self.writer.add_scalar("train/loss_high_freq", loss_high.item(), global_update)
                    self.writer.add_scalar("train/grad_norm", grad_norm, global_update)
                    self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], global_update)

                # Update model_last.pt periodically
                if (
                    self.last_per_updates > 0
                    and global_update % self.last_per_updates == 0
                    and self.accelerator.sync_gradients
                ):
                    self.save_checkpoint(global_update, last=True)

                # Save full checkpoint folder periodically
                if (
                    self.save_per_updates > 0
                    and global_update % self.save_per_updates == 0
                    and self.accelerator.sync_gradients
                ):
                    self.save_checkpoint(global_update)

                    sample_paths = self._save_sample_outputs(batch, global_update)
                    if sample_paths:
                        print(f"[samples] update={global_update} saved {' '.join(sample_paths)}")
                    self.accelerator.wait_for_everyone()

            if global_update > start_update:
                self.save_checkpoint(global_update, last=True)

        finally:
            signal.signal(signal.SIGINT, old_handler)
            if progress is not None:
                progress.close()
            if self.writer is not None:
                self.writer.close()
            self.accelerator.end_training()

class DiscriminatorTrainer(Trainer):
    """GAN discriminator trainer that trains a waveform-domain discriminator against a frozen CFM generator.

    The generator is loaded from a pre-trained checkpoint (EMA weights only) and kept frozen.
    Only the discriminator is trained, using a combination of L2 adversarial loss and feature
    matching loss.

    Training follows the standard GAN pattern: text + voice reference → generate audio →
    truncate both real and fake waveforms to the shorter length → D judges real vs fake.
    """

    def __init__(
        self,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        config: dict[str, Any],
    ):
        self.gen = generator
        for p in self.gen.parameters():
            p.requires_grad = False
        self.gen.eval()

        self.gen_steps = int(config.get("train", {}).get("gen_steps", 16))
        self.gen_cfg_strength = float(config.get("train", {}).get("gen_cfg_strength", 1.0))
        self.ref_ratio = float(config.get("train", {}).get("ref_ratio", 0.3))
        self.fm_loss_weight = float(config.get("train", {}).get("fm_loss_weight", 1.0))
        self.adv_loss_weight = float(config.get("train", {}).get("adv_loss_weight", 1.0))

        # Use disc_optim section for the discriminator optimizer (fall back to optim)
        disc_optim = config.get("disc_optim", dict(config["optim"]))
        saved_optim = config["optim"]
        config["optim"] = disc_optim
        super().__init__(discriminator, config)
        config["optim"] = saved_optim

        # Move frozen generator to the accelerator device
        self.gen = self.gen.to(self.accelerator.device)

        if self.is_main:
            d_total = sum(p.numel() for p in discriminator.parameters())
            d_trainable = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
            print(f"[discriminator] total_params={d_total:,} trainable_params={d_trainable:,}")

    # ------------------------------------------------------------------
    # Checkpoint: load generator EMA weights from a pre-trained CFM ckpt
    # ------------------------------------------------------------------

    def load_checkpoint(self) -> int:
        resume_from = self.config["train"].get("resume_from")
        if not resume_from or not os.path.exists(resume_from):
            raise FileNotFoundError(
                f"Generator checkpoint not found: {resume_from}. "
                f"Set train.resume_from to a pre-trained CFM checkpoint directory."
            )

        self.accelerator.wait_for_everyone()

        if os.path.isdir(resume_from):
            model_path = os.path.join(resume_from, "model.pt")
        else:
            model_path = resume_from

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        # Prefer EMA weights (better inference quality), but fall back to model_state_dict.
        # EMA only tracks trainable parameters, not buffers like loss_fn.weights,
        # so we must use strict=False when loading EMA.
        ema_key = None
        if "ema_model_state_dict" in checkpoint:
            ema_key = "ema_model_state_dict"
        elif "ema" in checkpoint:
            ema_key = "ema"

        if ema_key is not None:
            state_dict = checkpoint[ema_key]
            # Strip ema_model. prefix from EMA-tracked parameters
            if any(k.startswith("ema_model.") for k in state_dict):
                state_dict = {
                    k.removeprefix("ema_model."): v
                    for k, v in state_dict.items()
                    if k.startswith("ema_model.")
                }
            missing, unexpected = self.gen.load_state_dict(state_dict, strict=False)
            if self.is_main:
                print(f"[disc_trainer] Loaded generator EMA weights from {resume_from}")
                if missing:
                    print(f"[disc_trainer] Missing keys (using fresh init): {missing}")
                if unexpected:
                    print(f"[disc_trainer] Unexpected keys (ignored): {unexpected}")
        else:
            if self.is_main:
                print("[disc_trainer] WARNING: No EMA weights in checkpoint, loading model weights instead")
            model_key = "model_state_dict" if "model_state_dict" in checkpoint else "model"
            self.gen.load_state_dict(checkpoint[model_key], strict=True)

        del checkpoint
        gc.collect()

        # ---- Resume discriminator training if a discriminator checkpoint is provided ----
        disc_resume_from = self.config["train"].get("disc_resume_from")
        if disc_resume_from and os.path.exists(disc_resume_from):
            if os.path.isdir(disc_resume_from):
                disc_model_path = os.path.join(disc_resume_from, "discriminator.pt")
            else:
                disc_model_path = disc_resume_from

            disc_checkpoint = torch.load(disc_model_path, map_location="cpu", weights_only=True)

            raw_disc = self.accelerator.unwrap_model(self.model)
            raw_disc.load_state_dict(disc_checkpoint["discriminator_state_dict"])

            if "optimizer_state_dict" in disc_checkpoint:
                try:
                    self.optimizer.load_state_dict(disc_checkpoint["optimizer_state_dict"])
                except Exception:
                    if self.is_main:
                        print("[disc_resume] WARNING: Failed to load optimizer state (config may have changed). Using fresh optimizer.")
            if "scheduler_state_dict" in disc_checkpoint:
                try:
                    self.scheduler.load_state_dict(disc_checkpoint["scheduler_state_dict"])
                except Exception:
                    if self.is_main:
                        print("[disc_resume] WARNING: Failed to load scheduler state (config may have changed). Using fresh scheduler.")

            update = int(disc_checkpoint.get("update", 0))
            self.data_cycle_seed = int(disc_checkpoint.get("data_seed", update))

            del disc_checkpoint
            gc.collect()

            if self.is_main:
                print(f"[disc_resume] update={update} ckpt={disc_resume_from}")

            return update

        # Discriminator starts from step 0 — the generator's step is irrelevant here.
        return 0

    # ------------------------------------------------------------------
    # Checkpoint: save discriminator-only weights
    # ------------------------------------------------------------------

    def save_checkpoint(self, update: int, epoch: int = 0, last: bool = False) -> None:
        self.accelerator.wait_for_everyone()
        if self.is_main:
            raw_disc = self.accelerator.unwrap_model(self.model)
            state = {
                "discriminator_state_dict": raw_disc.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "epoch": epoch,
                "update": update,
                "data_seed": self.data_cycle_seed,
            }

            Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)

            if last:
                self.accelerator.save(state, f"{self.checkpoint_path}/discriminator_last.pt")
                print(f"Saved discriminator last checkpoint at update {update}")
            else:
                ckpt_dir = f"{self.checkpoint_path}/disc_step_{update:08d}"
                Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

                self.accelerator.save(state, f"{ckpt_dir}/discriminator.pt")

                config_path = os.path.join(ckpt_dir, "config.json")
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)

                print(f"[disc_checkpoint] update={update} saved to {ckpt_dir}/")

                if self.keep_last_n_checkpoints > 0:
                    self._rotate_disc_checkpoints()
        self.accelerator.wait_for_everyone()

    def _rotate_disc_checkpoints(self) -> None:
        ckpt_dirs = sorted(
            d for d in Path(self.checkpoint_path).iterdir()
            if d.is_dir() and d.name.startswith("disc_step_")
        )
        while len(ckpt_dirs) > self.keep_last_n_checkpoints:
            oldest = ckpt_dirs.pop(0)
            shutil.rmtree(oldest)
            print(f"Removed old discriminator checkpoint: {oldest.name}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, train_dataloader=None) -> None:
        if train_dataloader is None:
            train_dataloader = build_train_dataloader(self.config, self.accelerator)

        train_dataloader = self.accelerator.prepare(train_dataloader)

        start_update = self.load_checkpoint()
        global_update = start_update

        # TensorBoard
        if self.is_main and self._tb_cfg.get("enabled", True):
            log_dir = self._tb_cfg.get(
                "log_dir", os.path.join(self.checkpoint_path, "disc_tensorboard")
            )
            purge_step = start_update if start_update > 0 else None
            self.writer = SummaryWriter(log_dir=log_dir, purge_step=purge_step)

        # Progress bar
        progress = None
        if self.accelerator.is_local_main_process:
            total = self.max_steps if self.max_steps > 0 else None
            progress = tqdm(
                total=total,
                initial=global_update,
                dynamic_ncols=True,
                mininterval=0.5,
                unit="update",
                desc="disc_train",
            )

        fm_loss_fn = FeatureMatchingLoss()

        self.model.train()
        train_iterator = iter(train_dataloader)

        stop_requested = False

        def _handle_sigint(signum, frame):
            nonlocal stop_requested
            if stop_requested:
                raise KeyboardInterrupt
            stop_requested = True
            if self.accelerator.is_local_main_process:
                print("\n[SIGINT] Graceful stop requested. Press Ctrl+C again to force quit.", flush=True)

        old_handler = signal.signal(signal.SIGINT, _handle_sigint)

        try:
            while self.max_steps <= 0 or global_update < self.max_steps:
                if stop_requested:
                    break
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_dataloader)
                    continue

                wavs = batch["wavs"]             # (B, T_pad_wav) raw waveforms
                wav_lengths = batch["wav_lengths"] # (B,) sample lengths
                texts = batch["texts"]            # list[str]

                # Compute MDCT features on GPU from raw waveforms
                with torch.no_grad():
                    specs_feat, lengths = self.gen._to_spec(wavs, wav_lengths)

                hop = self.gen.spec.hop_length

                B = wavs.shape[0]

                # ---- Split: first ref_ratio of each sample = voice reference ----
                ref_len = (lengths.float() * self.ref_ratio).long().clamp(min=1)

                # ---- Generate with frozen generator (voice reference + text) ----
                with torch.no_grad(), self.accelerator.autocast():
                    with open(os.devnull, 'w') as devnull, contextlib.redirect_stderr(devnull):
                        generated, _trajectory = self.gen.sample(
                            cond=specs_feat,
                            text=texts,
                            duration=lengths,
                            lens=ref_len,
                            steps=self.gen_steps,
                            cfg_strength=self.gen_cfg_strength,
                        )

                gen_max = generated.shape[1]

                # ---- Per-sample: extract tails beyond reference, convert to waveform, truncate to min ----
                real_wavs = []
                fake_wavs = []
                for i in range(B):
                    r = ref_len[i].item()
                    t = lengths[i].item()

                    # Real tail: frames after the voice reference
                    real_tail = specs_feat[i, r:t]  # (real_tail_frames, F)
                    # Fake tail: generated frames after the voice reference
                    fake_tail = generated[i, r:min(t, gen_max)]

                    min_f = min(real_tail.shape[0], fake_tail.shape[0])
                    if min_f < 4:  # skip samples that are too short
                        continue

                    real_tail = real_tail[:min_f]
                    fake_tail = fake_tail[:min_f]

                    # Convert each sample to waveform individually (no MDCT-domain padding)
                    r_wav = self.gen.spec.inverse(
                        real_tail.unsqueeze(0).permute(0, 2, 1),
                        length=min_f * hop,
                    ).squeeze(0)  # (T_wav,)
                    f_wav = self.gen.spec.inverse(
                        fake_tail.unsqueeze(0).permute(0, 2, 1),
                        length=min_f * hop,
                    ).squeeze(0)

                    # Truncate to the shorter waveform
                    min_wav = min(r_wav.shape[-1], f_wav.shape[-1])
                    real_wavs.append(r_wav[..., :min_wav])
                    fake_wavs.append(f_wav[..., :min_wav])

                if len(real_wavs) == 0:
                    continue

                # ---- Pad waveforms into a batch ----
                real_wav = torch.nn.utils.rnn.pad_sequence(real_wavs, batch_first=True)
                fake_wav = torch.nn.utils.rnn.pad_sequence(fake_wavs, batch_first=True)
                real_wav = real_wav.unsqueeze(1)  # (B', 1, T_wav)
                fake_wav = fake_wav.unsqueeze(1)

                # ---- Discriminator forward ----
                d_real_scores, fmap_real_nested = self.model(real_wav)
                d_fake_scores, fmap_fake_nested = self.model(fake_wav)

                d_real = d_real_scores[0]  # (B'*segments,)
                d_fake = d_fake_scores[0]
                fmap_real = fmap_real_nested[0]   # list[Tensor] per discriminator layer
                fmap_fake = fmap_fake_nested[0]

                # ---- Loss computation ----
                # L2 adversarial loss (LS-GAN style)
                loss_real = F.mse_loss(d_real, torch.ones_like(d_real))
                loss_fake = F.mse_loss(d_fake, torch.zeros_like(d_fake))
                adv_loss = loss_real + loss_fake

                # Feature matching loss
                fm_loss = fm_loss_fn(fmap_real, fmap_fake)

                loss = self.adv_loss_weight * adv_loss + self.fm_loss_weight * fm_loss

                # ---- Accuracy ----
                with torch.no_grad():
                    real_acc = (d_real > 0.5).float().mean()
                    fake_acc = (d_fake < 0.5).float().mean()
                    accuracy = (real_acc + fake_acc) / 2

                # ---- Backward ----
                self.accelerator.backward(loss)

                grad_norm = 0.0
                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm if self.max_grad_norm > 0 else float("inf"),
                    ).item()

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    global_update += 1

                    if progress is not None:
                        progress.update(1)
                        progress.set_postfix(
                            update=str(global_update),
                            loss=f"{loss.item():.4f}",
                            acc=f"{accuracy.item():.2%}",
                        )

                # ---- TensorBoard ----
                if (
                    self.accelerator.is_local_main_process
                    and self.writer is not None
                    and self.log_every_steps > 0
                    and global_update % self.log_every_steps == 0
                ):
                    self.writer.add_scalar("disc/loss", loss.item(), global_update)
                    self.writer.add_scalar("disc/adv_loss", adv_loss.item(), global_update)
                    self.writer.add_scalar("disc/fm_loss", fm_loss.item(), global_update)
                    self.writer.add_scalar("disc/accuracy", accuracy.item(), global_update)
                    self.writer.add_scalar("disc/grad_norm", grad_norm, global_update)
                    self.writer.add_scalar("disc/lr", self.scheduler.get_last_lr()[0], global_update)

                # ---- Save model_last periodically ----
                if (
                    self.last_per_updates > 0
                    and global_update % self.last_per_updates == 0
                    and self.accelerator.sync_gradients
                ):
                    self.save_checkpoint(global_update, last=True)

                # ---- Save full checkpoint periodically ----
                if (
                    self.save_per_updates > 0
                    and global_update % self.save_per_updates == 0
                    and self.accelerator.sync_gradients
                ):
                    self.save_checkpoint(global_update)

            if global_update > start_update:
                self.save_checkpoint(global_update, last=True)

        finally:
            signal.signal(signal.SIGINT, old_handler)
            if progress is not None:
                progress.close()
            if self.writer is not None:
                self.writer.close()
            self.accelerator.end_training()
