from __future__ import annotations

import gc
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter
import torchaudio
from tqdm.auto import tqdm

from ema_pytorch import EMA

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from shore_tts.utils.build import (
    build_optimizer,
    build_scheduler,
    build_train_dataloader,
    get_mdct_feature_config,
)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: dict[str, Any],
    ):
        train_cfg = config["train"]
        optim_cfg = config["optim"]

        # Accelerator
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        precision = str(train_cfg.get("precision", "fp16")).lower()
        mp_map = {"fp32": "no", "fp16": "fp16", "bf16": "bf16"}
        mixed_precision = mp_map.get(precision, "no")
        grad_accum = int(train_cfg.get("grad_accumulation_steps", 1))

        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=grad_accum,
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
        self.config = config

        # Optimizer & scheduler (before prepare)
        self.optimizer = build_optimizer(config, model)
        self.scheduler = build_scheduler(config, self.optimizer, self.accelerator.num_processes)

        # Prepare model + optimizer with Accelerate (dataloader prepared later in train())
        self.model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        # EMA on main process only (after prepare so model is on correct device)
        if self.is_main:
            self.ema_model = EMA(self.accelerator.unwrap_model(self.model), include_online_model=False)
            self.ema_model.to(self.accelerator.device)
        else:
            self.ema_model = None

        # TensorBoard writer
        self.writer = None
        if self.is_main:
            tb_cfg = train_cfg.get("tensorboard", {})
            if tb_cfg.get("enabled", True):
                log_dir = tb_cfg.get("log_dir", os.path.join(self.checkpoint_path, "tensorboard"))
                self.writer = SummaryWriter(log_dir=log_dir)

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
        if not self.is_main:
            return

        raw_model = self.accelerator.unwrap_model(self.model)
        state = {
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": epoch,
            "update": update,
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
        del checkpoint
        gc.collect()

        if self.is_main:
            print(f"[resume] update={update} ckpt={resume_from}")

        return update

    @torch.no_grad()
    def _save_sample_outputs(self, batch: dict, global_update: int) -> list[str]:
        if not self.is_main:
            return []

        sample_cfg = self.config["train"].get("log_samples", {})
        if not sample_cfg.get("enabled", False):
            return []

        raw_model = self.accelerator.unwrap_model(self.model)
        original_state = None
        if self.ema_model is not None:
            original_state = {k: v.detach().clone() for k, v in raw_model.state_dict().items()}
            self.ema_model.copy_to(raw_model)

        was_training = raw_model.training
        raw_model.eval()

        sample_dir = Path(self.checkpoint_path) / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)

        device = self.accelerator.device
        sample_index = int(sample_cfg.get("sample_index", 0))
        sample_index = min(sample_index, batch["specs"].shape[0] - 1)
        ref_spec = batch["specs"][sample_index : sample_index + 1]
        ref_len = int(batch["lengths"][sample_index].item())
        ref_text = batch["texts"][sample_index]
        duration = int(max(ref_len * float(sample_cfg.get("duration_factor", 2.0)), ref_len + 1))
        sample_steps = int(sample_cfg.get("sample_steps", 16))
        cfg_strength = float(sample_cfg.get("cfg_strength", 1.0))

        with torch.inference_mode(), self.accelerator.autocast():
            generated, _ = raw_model.sample(
                cond=ref_spec[:, :ref_len],
                text=[f"{ref_text} {ref_text}"],
                duration=duration,
                steps=sample_steps,
                cfg_strength=cfg_strength,
                lens=torch.tensor([ref_len], device=device, dtype=torch.long),
            )
            generated = generated[:, ref_len:, :]
            gen_audio = raw_model.spec.inverse(
                generated.permute(0, 2, 1), length=generated.shape[1] * raw_model.spec.hop_length
            ).cpu()
            ref_audio = raw_model.spec.inverse(
                ref_spec[:, :ref_len].permute(0, 2, 1), length=ref_len * raw_model.spec.hop_length
            ).cpu()

        sample_rate = int(
            self.config["data"].get("sample_rate")
            or get_mdct_feature_config(self.config["data"]["mdct_config"])["sample_rate"]
        )
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

        # Calculate starting epoch from update count
        grad_accum = self.accelerator.gradient_accumulation_steps
        try:
            steps_per_epoch = len(train_dataloader)
        except TypeError:
            steps_per_epoch = None
        if steps_per_epoch is not None:
            start_epoch = (start_update * grad_accum) // steps_per_epoch
            skipped_batches = (start_update * grad_accum) % steps_per_epoch
        else:
            start_epoch = 0
            skipped_batches = 0

        # Skip already-processed batches on resume
        if start_update > 0 and skipped_batches > 0:
            train_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batches)

        progress = None
        if self.accelerator.is_local_main_process:
            total = self.max_steps if self.max_steps > 0 else None
            progress = tqdm(
                total=total,
                initial=global_update,
                dynamic_ncols=True,
                mininterval=0.5,
                unit="update",
                desc=f"epoch {start_epoch}",
            )

        try:
            for epoch in range(start_epoch, self.epochs):
                self.model.train()

                # Set epoch for reproducible shuffling
                if hasattr(train_dataloader, "batch_sampler") and hasattr(
                    train_dataloader.batch_sampler, "set_epoch"
                ):
                    train_dataloader.batch_sampler.set_epoch(epoch)
                if hasattr(train_dataloader, "dataset") and hasattr(train_dataloader.dataset, "set_epoch"):
                    train_dataloader.dataset.set_epoch(epoch)

                if progress is not None:
                    progress.set_description(f"epoch {epoch}")

                for batch in train_dataloader:
                    with self.accelerator.accumulate(self.model):
                        loss, _, _ = self.model(
                            inp=batch["specs"],
                            text=batch["texts"],
                            lens=batch["lengths"],
                        )
                        self.accelerator.backward(loss)

                        if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

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
                        self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], global_update)

                    # Update model_last.pt periodically
                    if (
                        self.last_per_updates > 0
                        and global_update % self.last_per_updates == 0
                        and self.accelerator.sync_gradients
                    ):
                        self.save_checkpoint(global_update, epoch=epoch, last=True)

                    # Save full checkpoint folder periodically
                    if (
                        self.save_per_updates > 0
                        and global_update % self.save_per_updates == 0
                        and self.accelerator.sync_gradients
                    ):
                        self.save_checkpoint(global_update, epoch=epoch)

                        sample_paths = self._save_sample_outputs(batch, global_update)
                        if sample_paths:
                            print(f"[samples] update={global_update} saved {' '.join(sample_paths)}")

                    if self.max_steps > 0 and global_update >= self.max_steps:
                        break

                # Save end-of-epoch last checkpoint
                self.save_checkpoint(global_update, epoch=epoch + 1, last=True)

                if self.max_steps > 0 and global_update >= self.max_steps:
                    break

        finally:
            if progress is not None:
                progress.close()
            if self.writer is not None:
                self.writer.close()
            self.accelerator.end_training()
