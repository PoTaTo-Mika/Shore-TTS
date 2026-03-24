from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..')) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.environ['PROJECT_ROOT'] = project_root

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore")

from shore_tts.utils.build import (
    build_model,
    build_optimizer,
    build_scheduler,
    build_train_dataloader,
    checkpoint_state,
    cleanup_distributed,
    get_device,
    init_distributed,
    is_main_process,
    load_checkpoint,
    load_json_config,
    move_batch_to_device,
    save_checkpoint,
    set_seed,
    wrap_ddp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="shore_tts/configs/pretrain.json",
        help="Path to training config JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json_config(args.config)

    ddp_cfg = config.get("ddp", {})
    distributed, rank, world_size, local_rank = init_distributed(ddp_cfg.get("backend", "nccl"))
    device = get_device(local_rank)

    try:
        set_seed(int(config.get("seed", 42)), rank=rank)

        model = build_model(config, device)
        raw_model = model
        optimizer = build_optimizer(config, model)
        scheduler = build_scheduler(config, optimizer)
        model = wrap_ddp(model, device, distributed)
        dataloader = build_train_dataloader(config, rank=rank, world_size=world_size)

        train_cfg = config["train"]
        save_dir = train_cfg["save_dir"]
        save_every_steps = int(train_cfg.get("save_every_steps", 1000))
        log_every_steps = int(train_cfg.get("log_every_steps", 10))
        timing_every_steps = int(train_cfg.get("timing_every_steps", 100))
        max_steps = int(train_cfg.get("max_steps", -1))
        grad_clip = float(config["optim"].get("grad_clip", 1.0))

        writer = None
        if is_main_process(rank) and train_cfg.get("tensorboard", {}).get("enabled", True):
            log_dir = train_cfg.get("tensorboard", {}).get("log_dir", os.path.join(save_dir, "tensorboard"))
            writer = SummaryWriter(log_dir=log_dir)

        progress = None
        if is_main_process(rank):
            total_params = sum(p.numel() for p in raw_model.parameters())
            trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
            tqdm.write(f"[model] total_params={total_params:,} trainable_params={trainable_params:,}")

        start_epoch = 0
        global_step = 0
        resume_from = train_cfg.get("resume_from")
        if resume_from:
            start_epoch, global_step = load_checkpoint(
                resume_from,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                map_location=device,
            )
            if is_main_process(rank):
                tqdm.write(f"[resume] epoch={start_epoch} global_step={global_step} ckpt={resume_from}")

        if is_main_process(rank):
            total_steps = max_steps if max_steps > 0 else None
            progress = tqdm(
                total=total_steps,
                initial=global_step,
                dynamic_ncols=True,
                mininterval=0.5,
                unit="it",
                desc=f"epoch {start_epoch}",
            )

        timing_window = {
            "data_fetch": 0.0,
            "cpu_io": 0.0,
            "cpu_mdct": 0.0,
            "host_to_device": 0.0,
            "gpu_step": 0.0,
            "count": 0,
        }

        for epoch in range(start_epoch, int(train_cfg["epochs"])):
            model.train()
            if hasattr(dataloader, "dataset") and hasattr(dataloader.dataset, "set_epoch"):
                dataloader.dataset.set_epoch(epoch)
            data_iter = iter(dataloader)
            if progress is not None:
                progress.set_description(f"epoch {epoch}")

            join_context = model.join() if distributed and hasattr(model, "join") else nullcontext()
            with join_context:
                while True:
                    fetch_start = time.perf_counter()
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        break
                    data_fetch_time = time.perf_counter() - fetch_start

                    cpu_io_time = float(batch.get("io_time", 0.0))
                    cpu_mdct_time = float(batch.get("mdct_time", 0.0))

                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    h2d_start = time.perf_counter()
                    batch = move_batch_to_device(batch, device)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    host_to_device_time = time.perf_counter() - h2d_start

                    optimizer.zero_grad(set_to_none=True)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    gpu_start = time.perf_counter()
                    loss, _, _ = model(
                        inp=batch["specs"],
                        text=batch["texts"],
                        lens=batch["lengths"],
                    )
                    loss.backward()

                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                    optimizer.step()
                    scheduler.step()
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    gpu_step_time = time.perf_counter() - gpu_start
                    global_step += 1
                    latest_loss = float(loss.detach())
                    latest_lr = scheduler.get_last_lr()[0]

                    timing_window["data_fetch"] += data_fetch_time
                    timing_window["cpu_io"] += cpu_io_time
                    timing_window["cpu_mdct"] += cpu_mdct_time
                    timing_window["host_to_device"] += host_to_device_time
                    timing_window["gpu_step"] += gpu_step_time
                    timing_window["count"] += 1
                    if progress is not None:
                        progress.update(1)

                    if writer is not None:
                        writer.add_scalar("train/loss", latest_loss, global_step)
                        writer.add_scalar("train/lr", latest_lr, global_step)
                        writer.add_scalar("time/data_fetch_s", data_fetch_time, global_step)
                        writer.add_scalar("time/cpu_io_s", cpu_io_time, global_step)
                        writer.add_scalar("time/cpu_mdct_s", cpu_mdct_time, global_step)
                        writer.add_scalar("time/host_to_device_s", host_to_device_time, global_step)
                        writer.add_scalar("time/gpu_step_s", gpu_step_time, global_step)

                    should_refresh_progress = False
                    if is_main_process(rank) and log_every_steps > 0 and global_step % log_every_steps == 0:
                        should_refresh_progress = True
                    if is_main_process(rank) and timing_every_steps > 0 and global_step % timing_every_steps == 0:
                        should_refresh_progress = True

                    if is_main_process(rank) and should_refresh_progress and progress is not None:
                        count = max(timing_window["count"], 1)
                        progress.set_postfix(
                            {
                                "step": global_step,
                                "loss": f"{latest_loss:.4f}",
                                "lr": f"{latest_lr:.2e}",
                                "fetch": f"{timing_window['data_fetch'] / count:.3f}s",
                                "io": f"{timing_window['cpu_io'] / count:.3f}s",
                                "mdct": f"{timing_window['cpu_mdct'] / count:.3f}s",
                                "h2d": f"{timing_window['host_to_device'] / count:.3f}s",
                                "gpu": f"{timing_window['gpu_step'] / count:.3f}s",
                            },
                            refresh=False,
                        )
                        for key in timing_window:
                            timing_window[key] = 0.0 if key != "count" else 0

                    if is_main_process(rank) and save_every_steps > 0 and global_step % save_every_steps == 0:
                        state = checkpoint_state(model, optimizer, scheduler, epoch, global_step, config)
                        path = save_checkpoint(save_dir, state, f"step_{global_step:08d}.pt")
                        tqdm.write(f"[checkpoint] step={global_step} saved {path}")

                    if max_steps > 0 and global_step >= max_steps:
                        break

            if is_main_process(rank):
                state = checkpoint_state(model, optimizer, scheduler, epoch + 1, global_step, config)
                path = save_checkpoint(save_dir, state, "last.pt")
                tqdm.write(f"[checkpoint] epoch={epoch + 1} step={global_step} saved {path}")

            if max_steps > 0 and global_step >= max_steps:
                break

    finally:
        if "progress" in locals() and progress is not None:
            progress.close()
        if "writer" in locals() and writer is not None:
            writer.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
