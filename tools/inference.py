"""Shore-TTS inference script.

Usage:
    python -m shore_tts.tools.inference \
        --checkpoint checkpoints/Shore-TTS-0.1 \
        --text "你好世界" \
        --output output.wav

    # With reference audio for voice cloning
    python -m shore_tts.tools.inference \
        --checkpoint checkpoints/Shore-TTS-0.1 \
        --text "你好世界" \
        --ref_audio ref.wav \
        --ref_text "参考文本" \
        --output output.wav
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..')) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.environ['PROJECT_ROOT'] = project_root

import torch
import torchaudio

from shore_tts.models.cfm import CFM
from shore_tts.models.dit import DiT
from shore_tts.text.tokenizer import PinyinTokenizer


def locate_checkpoint_files(checkpoint_dir: str | Path) -> tuple[Path, Path, Path]:
    """Locate config.json, vocab.json, and model.pth inside *checkpoint_dir*.

    Looks directly inside the directory first, then one level deep for
    sub-directories that contain the expected files.
    """
    checkpoint_dir = Path(checkpoint_dir)

    def _find(base: Path) -> tuple[Path, Path, Path] | None:
        config = base / "config.json"
        vocab = base / "vocab.json"
        model = base / "model.pth"
        if config.is_file() and vocab.is_file() and model.is_file():
            return config, vocab, model
        return None

    result = _find(checkpoint_dir)
    if result is not None:
        return result

    # Search one level deep
    for child in sorted(checkpoint_dir.iterdir()):
        if child.is_dir():
            result = _find(child)
            if result is not None:
                return result

    raise FileNotFoundError(
        f"Could not find config.json, vocab.json, and model.pth in {checkpoint_dir}"
    )


def load_model(
    checkpoint_dir: str | Path,
    device: torch.device,
) -> tuple[CFM, int]:
    """Build the CFM model from checkpoint_dir/config.json + vocab.json,
    load model.pth weights, and return (model, sample_rate).
    """
    config_path, vocab_path, model_path = locate_checkpoint_files(checkpoint_dir)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # ---- MDCT feature config (from the embedded data.mdct_config or defaults) ----
    mdct_cfg = config.get("data", {}).get("mdct_config")
    if mdct_cfg and Path(mdct_cfg).is_file():
        with open(mdct_cfg, "r", encoding="utf-8") as f:
            mdct = json.load(f)
    else:
        # Fallback to the config bundled inside the checkpoint dir
        local_mdct = Path(checkpoint_dir) / "mdct.json"
        if local_mdct.is_file():
            with open(local_mdct, "r", encoding="utf-8") as f:
                mdct = json.load(f)
        else:
            # Sensible defaults matching the pretrain config
            mdct = {"mdct_params": {"hop_length": 100, "n_bands": 20}, "sample_rate": 24000}

    mdct_params = mdct.get("mdct_params", {})
    hop_length = int(mdct_params.get("hop_length", 100))
    n_bands = int(mdct_params.get("n_bands", 20))
    spec_dim = hop_length + n_bands
    sample_rate = int(mdct.get("sample_rate", 24000))

    # ---- Tokenizer ----
    text_cfg = config.get("text", {})
    tokenizer = PinyinTokenizer.load(
        str(vocab_path),
        polyphone=bool(text_cfg.get("polyphone", True)),
    )

    # ---- DiT ----
    dit_cfg = dict(config["model"]["dit"])
    dit_cfg.setdefault("spec_dim", spec_dim)
    dit_cfg["text_num_embeds"] = tokenizer.vocab_size
    # Flash attention is not available on Windows; fall back to PyTorch native.
    if dit_cfg.get("attn_backend") == "flash_attn":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            dit_cfg["attn_backend"] = "torch"
            print("[info] flash_attn not available, using 'torch' attention backend.")
    transformer = DiT(**dit_cfg)

    # ---- CFM ----
    cfm_cfg = dict(config["model"].get("cfm", {}))
    cfm_cfg["num_channels"] = spec_dim
    cfm_cfg["spec_kwargs"] = {
        "hop_length": hop_length,
        "n_bands": n_bands,
        "target_sample_rate": sample_rate,
    }
    cfm_cfg["vocab_char_map"] = tokenizer.token_to_id
    cfm_cfg["text_tokenizer"] = tokenizer
    model = CFM(transformer=transformer, **cfm_cfg)

    # ---- Load weights ----
    state_dict = torch.load(str(model_path), map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

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
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, int]:
    """Run inference and return (waveform, sample_rate).

    The waveform tensor has shape ``(1, T)`` and lives on CPU.

    Args:
        speed: Speech speed factor. Higher values produce faster (shorter)
            audio, lower values produce slower (longer) audio. Default 1.0.
        fix_duration: If set, force the total output duration (ref + gen) to
            this value in seconds. Overrides the proportional duration estimate.
    """
    if seed is not None:
        torch.manual_seed(seed)

    target_sr = model.spec.target_sample_rate
    hop_length = model.spec.hop_length

    # ---- Reference audio ----
    if ref_audio is not None:
        ref_waveform, ref_sr = torchaudio.load(str(ref_audio))
        if ref_sr != target_sr:
            ref_waveform = torchaudio.functional.resample(ref_waveform, ref_sr, target_sr)
        if ref_waveform.shape[0] > 1:
            ref_waveform = ref_waveform.mean(dim=0, keepdim=True)
        ref_waveform = ref_waveform.to(device)  # (1, T)
    else:
        # No reference audio: use an empty 1-second silence as cond
        ref_waveform = torch.zeros(1, target_sr, device=device)

    # ---- Build the full text prompt (ref_text + target_text) ----
    if ref_text is not None and ref_audio is not None:
        full_text = f"{ref_text} {text}"
    else:
        full_text = text

    # ---- Compute reference spec length for duration ----
    ref_spec = model.spec(ref_waveform)  # (1, F, T)
    ref_spec = ref_spec.permute(0, 2, 1)  # (1, T, F)
    ref_len = ref_spec.shape[1]

    # ---- Estimate duration ----
    if fix_duration is not None:
        duration = int(fix_duration * target_sr / hop_length)
    else:
        # Proportional estimation inspired by F5-TTS:
        # ref_frames / ref_text_bytes gives "frames per text byte" from the
        # reference, then scale by the generation text byte length / speed.
        ref_text_len = len(ref_text.encode("utf-8")) if ref_text is not None else 0
        gen_text_len = len(text.encode("utf-8"))
        if ref_text_len > 0 and ref_audio is not None:
            gen_frames = int(ref_len / ref_text_len * gen_text_len / speed)
        else:
            # No reference text or no reference audio: estimate from token count
            # ~6 MDCT frames per pinyin token at normal speed (≈25 ms/token)
            text_tokens = model.tokenize_text([full_text], device)
            n_text_tokens = int((text_tokens != -1).sum(dim=-1).item())
            gen_frames = int(n_text_tokens * 6 / speed)
        duration = ref_len + gen_frames

    # Ensure at least one frame beyond the reference
    duration = max(duration, ref_len + 1)

    # ---- Sample ----
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

    # Slice off the reference portion to get only generated audio
    generated = generated[:, ref_len:, :]

    # Inverse MDCT -> waveform
    # generated is (B, T, D); MDCTSpec.inverse expects (B, F, T)
    wav = model.spec.inverse(generated.permute(0, 2, 1), length=generated.shape[1] * hop_length).cpu()

    return wav, target_sr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shore-TTS inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/Shore-TTS-0.1",
        help="Path to the checkpoint directory containing config.json, vocab.json, and model.pth.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="你是谁？请支持明日方舟。",
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--ref_audio",
        type=str,
        default=None,
        help="Path to reference audio for voice cloning (optional).",
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        default=None,
        help="Transcript of the reference audio (required if --ref_audio is provided).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test.wav",
        help="Output wav file path.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=32,
        help="Number of ODE solver steps (default: 32).",
    )
    parser.add_argument(
        "--cfg_strength",
        type=float,
        default=1.0,
        help="Classifier-free guidance strength (default: 1.0).",
    )
    parser.add_argument(
        "--sway_sampling_coef",
        type=float,
        default=None,
        help="Sway sampling coefficient (default: None, disabled).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed factor. Higher = faster, lower = slower (default: 1.0).",
    )
    parser.add_argument(
        "--fix_duration",
        type=float,
        default=None,
        help="Fix total output duration in seconds (overrides proportional estimate).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    print(f"Loading model from {args.checkpoint} ...")
    model, sample_rate = load_model(args.checkpoint, device)
    print(f"Model loaded. sample_rate={sample_rate} device={device}")

    print(f"Synthesizing: {args.text}")
    wav, sr = infer(
        model=model,
        text=args.text,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        steps=args.steps,
        cfg_strength=args.cfg_strength,
        sway_sampling_coef=args.sway_sampling_coef,
        seed=args.seed,
        speed=args.speed,
        fix_duration=args.fix_duration,
        device=device,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), wav, sr)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
