from __future__ import annotations

import argparse
import json
from pathlib import Path
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["PROJECT_ROOT"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torchaudio

from shore_tts.models.cfm import CFM
from shore_tts.models.dit import DiT
from shore_tts.text.tokenizer import PinyinTokenizer


def _find_checkpoint(base: Path) -> tuple[Path, Path, Path] | None:
    config, vocab = base / "config.json", base / "vocab.json"
    if not (config.is_file() and vocab.is_file()):
        return None
    for name in ("model.pth", "model.pt"):
        model = base / name
        if model.is_file():
            return config, vocab, model
    return None


def locate_checkpoint_files(checkpoint_dir: str | Path) -> tuple[Path, Path, Path]:
    checkpoint_dir = Path(checkpoint_dir)
    result = _find_checkpoint(checkpoint_dir)
    if result:
        return result
    for child in sorted(checkpoint_dir.iterdir()):
        if child.is_dir():
            result = _find_checkpoint(child)
            if result:
                return result
    raise FileNotFoundError(f"Could not locate checkpoint files in {checkpoint_dir}")


def load_model(checkpoint_dir: str | Path, device: torch.device) -> tuple[CFM, int]:
    config_path, vocab_path, model_path = locate_checkpoint_files(checkpoint_dir)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # MDCT config: embedded path > local mdct.json > defaults
    mdct_cfg = config.get("data", {}).get("mdct_config")
    if mdct_cfg and Path(mdct_cfg).is_file():
        with open(mdct_cfg, "r", encoding="utf-8") as f:
            mdct = json.load(f)
    else:
        local = Path(checkpoint_dir) / "mdct.json"
        if local.is_file():
            with open(local, "r", encoding="utf-8") as f:
                mdct = json.load(f)
        else:
            mdct = {"mdct_params": {"hop_length": 100, "n_bands": 20}, "sample_rate": 24000}

    hop_length = int(mdct.get("mdct_params", {}).get("hop_length", 100))
    n_bands = int(mdct.get("mdct_params", {}).get("n_bands", 20))
    spec_dim = hop_length + n_bands
    sample_rate = int(mdct.get("sample_rate", 24000))

    # Tokenizer
    text_cfg = config.get("text", {})
    tokenizer = PinyinTokenizer.load(str(vocab_path), polyphone=bool(text_cfg.get("polyphone", True)))

    # DiT
    dit_cfg = dict(config["model"]["dit"])
    dit_cfg.setdefault("spec_dim", spec_dim)
    dit_cfg["text_num_embeds"] = tokenizer.vocab_size
    if dit_cfg.get("attn_backend") == "flash_attn":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            dit_cfg["attn_backend"] = "torch"
            print("[info] flash_attn unavailable, falling back to 'torch'")
    transformer = DiT(**dit_cfg)

    # CFM
    cfm_cfg = dict(config["model"].get("cfm", {}))
    cfm_cfg.update(num_channels=spec_dim, vocab_char_map=tokenizer.token_to_id, text_tokenizer=tokenizer)
    cfm_cfg["spec_kwargs"] = {"hop_length": hop_length, "n_bands": n_bands, "target_sample_rate": sample_rate}
    model = CFM(transformer=transformer, **cfm_cfg)

    # Weights
    raw = torch.load(str(model_path), map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        if "ema_model_state_dict" in raw:
            state_dict = {k.removeprefix("ema_model."): v for k, v in raw["ema_model_state_dict"].items() if k.startswith("ema_model.")}
            print("[info] Loaded EMA weights.")
        else:
            state_dict = raw["model_state_dict"]
            print("[warning] No EMA weights found; quality may be worse.")
    else:
        state_dict = raw
        print("[info] Loaded pre-extracted weights.")

    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model, sample_rate


@torch.no_grad()
def infer(
    model: CFM,
    text: str,
    ref_audio: str | Path | None = None,
    ref_text: str | None = None,
    steps: int = 32,
    cfg_strength: float = 1.0,
    sway_sampling_coef: float | None = None,
    seed: int | None = None,
    max_duration: int = 65536,
    speed: float = 1.0,
    fix_duration: float | None = None,
    duration_factor: float | None = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, int]:
    if seed is not None:
        torch.manual_seed(seed)

    target_sr = model.spec.target_sample_rate
    hop_length = model.spec.hop_length

    # Reference audio
    if ref_audio is not None:
        ref_waveform, ref_sr = torchaudio.load(str(ref_audio))
        if ref_sr != target_sr:
            ref_waveform = torchaudio.functional.resample(ref_waveform, ref_sr, target_sr)
        if ref_waveform.shape[0] > 1:
            ref_waveform = ref_waveform.mean(dim=0, keepdim=True)
        ref_waveform = ref_waveform.to(device)
    else:
        ref_waveform = torch.zeros(1, target_sr, device=device)

    full_text = f"{ref_text} {text}" if (ref_text and ref_audio) else text

    ref_spec = model.spec(ref_waveform).permute(0, 2, 1)  # (1, T, F)
    ref_len = ref_spec.shape[1]

    # Duration estimation
    if fix_duration is not None:
        duration = int(fix_duration * target_sr / hop_length)
    elif duration_factor is not None:
        duration = int(ref_len * duration_factor)
    else:
        ref_text_len = len(ref_text.encode("utf-8")) if ref_text else 0
        gen_text_len = len(text.encode("utf-8"))
        if ref_text_len > 0 and ref_audio is not None:
            gen_frames = int(ref_len / ref_text_len * gen_text_len / speed)
        else:
            text_tokens = model.tokenize_text([full_text], device)
            n_tokens = int((text_tokens != -1).sum(dim=-1).item())
            gen_frames = int(n_tokens * 6 / speed)
        duration = ref_len + gen_frames

    duration = max(duration, ref_len + 1)

    autocast_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
        generated, _ = model.sample(
            cond=ref_spec[:, :ref_len],
            text=[full_text],
            duration=duration,
            steps=steps,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            seed=seed,
            max_duration=max_duration,
            lens=torch.tensor([ref_len], device=device, dtype=torch.long),
        )

    generated = generated[:, ref_len:, :]
    wav = model.spec.inverse(generated.permute(0, 2, 1), length=generated.shape[1] * hop_length).cpu()
    return wav, target_sr


def main() -> None:
    p = argparse.ArgumentParser(description="Shore-TTS inference")
    p.add_argument("--checkpoint", default="checkpoints/Shore-TTS-0.1")
    p.add_argument("--text", default="你们有人想做我的颜料吗？")
    p.add_argument("--ref_audio", default="./assets/test.wav")
    p.add_argument("--ref_text", default="你们有人想做我的颜料吗")
    p.add_argument("--output", default="test.wav")
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--cfg_strength", type=float, default=1.0)
    p.add_argument("--sway_sampling_coef", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--fix_duration", type=float, default=None)
    p.add_argument("--duration_factor", type=float, default=2.0)
    args = p.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    print(f"Loading model from {args.checkpoint} ...")
    model, sample_rate = load_model(args.checkpoint, device)
    print(f"Model loaded. sample_rate={sample_rate} device={device}")

    print(f"Synthesizing: {args.text}")
    wav, sr = infer(
        model=model, text=args.text, ref_audio=args.ref_audio, ref_text=args.ref_text,
        steps=args.steps, cfg_strength=args.cfg_strength, sway_sampling_coef=args.sway_sampling_coef,
        seed=args.seed, speed=args.speed, fix_duration=args.fix_duration,
        duration_factor=args.duration_factor, device=device,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out), wav, sr)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
