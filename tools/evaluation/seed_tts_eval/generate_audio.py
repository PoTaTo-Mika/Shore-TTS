"""
Generate evaluation audio from SeedTTS testset using Shore-TTS.

Usage:
    python generate_audio.py /path/to/seedtts_testset [--checkpoint CHECKPOINT] [--steps 16] [--device cuda]

The testset folder is expected to have this structure:
    seedtts_testset/
    ├── zh/
    │   ├── meta.lst                          (4 fields: utt|prompt_text|prompt_wav|gt_text)
    │   ├── non_para_reconstruct_meta.lst     (5 fields: utt|prompt_text|prompt_wav|gt_text|gt_wav)
    │   ├── hardcase.lst                      (4 fields, same as meta.lst)
    │   ├── prompt-wavs/
    │   └── wavs/
    └── en/
        ├── meta.lst
        ├── non_para_reconstruct_meta.lst
        ├── prompt-wavs/
        └── wavs/

Output is saved to tools/evaluation/eval_audio/seed_tts_eval/ preserving the task structure:
    eval_audio/seed_tts_eval/
    ├── zh/
    │   ├── meta/
    │   ├── non_para_reconstruct/
    │   └── hardcase/
    └── en/
        ├── meta/
        └── non_para_reconstruct/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

# project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.inference import infer, load_model

# recognized meta files and their output sub-directory names
TASK_FILES = {
    "meta.lst": "meta",
    "non_para_reconstruct_meta.lst": "non_para_reconstruct",
    "hardcase.lst": "hardcase",
}

OUTPUT_ROOT = PROJECT_ROOT / "tools" / "evaluation" / "eval_audio" / "seed_tts_eval"


def discover_tasks(folder: Path) -> list[tuple[str, Path, Path]]:
    """Return list of (lang, task_name, meta_file_path) for every discovered task."""
    tasks = []
    for lang_dir in sorted(folder.iterdir()):
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        for filename, task_name in TASK_FILES.items():
            meta = lang_dir / filename
            if meta.is_file():
                tasks.append((lang, task_name, meta))
    return tasks


def parse_meta(meta_path: Path) -> list[dict]:
    """Parse a meta.lst file into a list of dicts."""
    entries = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) == 5:
                utt, prompt_text, prompt_wav, gt_text, _gt_wav = parts
            elif len(parts) == 4:
                utt, prompt_text, prompt_wav, gt_text = parts
            else:
                continue
            # resolve relative prompt_wav path against the meta file's directory
            if not os.path.isabs(prompt_wav):
                prompt_wav = str(meta_path.parent / prompt_wav)
            entries.append({
                "utt": utt,
                "prompt_text": prompt_text,
                "prompt_wav": prompt_wav,
                "gt_text": gt_text,
            })
    return entries


def main():
    parser = argparse.ArgumentParser(description="Generate SeedTTS eval audio with Shore-TTS")
    parser.add_argument("folder", type=Path, help="Path to seedtts_testset folder")
    parser.add_argument("--checkpoint", type=str, default=str(PROJECT_ROOT / "checkpoints" / "pretrain-200M"),
                        help="Shore-TTS checkpoint directory")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--cfg_strength", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lang", type=str, default=None, choices=["zh", "en"],
                        help="Only process a specific language")
    parser.add_argument("--task", type=str, default=None,
                        choices=list(TASK_FILES.values()),
                        help="Only process a specific task")
    args = parser.parse_args()

    if not args.folder.is_dir():
        parser.error(f"Folder not found: {args.folder}")

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    print(f"Loading model from {args.checkpoint} ...")
    model, sample_rate = load_model(args.checkpoint, device)
    print(f"Model loaded. sample_rate={sample_rate} device={device}")

    tasks = discover_tasks(args.folder)
    if not tasks:
        print("No task files found in the given folder.")
        return

    for lang, task_name, meta_path in tasks:
        if args.lang and lang != args.lang:
            continue
        if args.task and task_name != args.task:
            continue

        entries = parse_meta(meta_path)
        out_dir = OUTPUT_ROOT / lang / task_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{lang}/{task_name}] {len(entries)} samples -> {out_dir}")

        for entry in tqdm(entries, desc=f"{lang}/{task_name}"):
            out_path = out_dir / f"{entry['utt']}.wav"
            if out_path.exists():
                continue

            try:
                wav, sr = infer(
                    model=model,
                    text=entry["gt_text"],
                    ref_audio=entry["prompt_wav"],
                    ref_text=entry["prompt_text"],
                    steps=args.steps,
                    cfg_strength=args.cfg_strength,
                    device=device,
                )
                torchaudio.save(str(out_path), wav, sr)
            except Exception as e:
                print(f"  FAILED {entry['utt']}: {e}")


if __name__ == "__main__":
    main()
