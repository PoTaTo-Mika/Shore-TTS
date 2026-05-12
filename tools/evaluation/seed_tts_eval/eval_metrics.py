"""
Evaluate SeedTTS metrics (WER + SIM) for generated audio.

Usage:
    python eval_metrics.py --testset /root/seedtts_testset --eval_repo /root/seed-tts-eval --wavlm_ckpt /path/to/wavlm_large_finetune.pth

    # WER only (no wavlm checkpoint needed)
    python eval_metrics.py --metrics wer

    # Single language / task
    python eval_metrics.py --lang zh --task hardcase --metrics wer
"""

from __future__ import annotations

import argparse
import os
import string
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TASK_FILES = {
    "meta.lst": "meta",
    "non_para_reconstruct_meta.lst": "non_para_reconstruct",
    "hardcase.lst": "hardcase",
}

# reverse mapping: task dir name -> meta.lst filename
TASK_TO_METAFILE = {v: k for k, v in TASK_FILES.items()}


def discover_tasks(testset_dir: Path, gen_dir: Path) -> list[tuple[str, str, Path, Path]]:
    """Return (lang, task_name, meta_path, gen_wav_dir) for every discovered task."""
    tasks = []
    for lang_dir in sorted(testset_dir.iterdir()):
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        for task_dir in sorted(gen_dir.joinpath(lang).iterdir()):
            if not task_dir.is_dir():
                continue
            meta_file = TASK_TO_METAFILE.get(task_dir.name)
            if meta_file is None:
                continue
            meta_path = lang_dir / meta_file
            if not meta_path.is_file():
                continue
            tasks.append((lang, task_dir.name, meta_path, task_dir))
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
            if not os.path.isabs(prompt_wav):
                prompt_wav = str(meta_path.parent / prompt_wav)
            entries.append({
                "utt": utt,
                "prompt_text": prompt_text,
                "prompt_wav": prompt_wav,
                "gt_text": gt_text,
            })
    return entries


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------

def load_asr_model(lang: str, device: str):
    if lang == "zh":
        from funasr import AutoModel
        return AutoModel(model="paraformer-zh")
    else:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        model_id = "openai/whisper-large-v3"
        processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
        return (processor, model)


def transcribe(asr_model, wav_path: str, lang: str, device: str) -> str:
    if lang == "zh":
        import zhconv
        res = asr_model.generate(input=wav_path, batch_size_s=300)
        text = res[0]["text"]
        return zhconv.convert(text, "zh-cn")
    else:
        import soundfile as sf
        import scipy.signal
        processor, model = asr_model
        wav, sr = sf.read(wav_path)
        if sr != 16000:
            wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
        input_features = processor(wav, sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to(device, dtype=model.dtype)
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
        predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
        return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


def compute_wer(hypo: str, truth: str, lang: str) -> float:
    from jiwer import wer
    from zhon.hanzi import punctuation as zh_punctuation

    table = str.maketrans("", "", zh_punctuation + string.punctuation.replace("'", ""))
    truth = truth.translate(table).replace("  ", " ")
    hypo = hypo.translate(table).replace("  ", " ")

    if lang == "zh":
        truth = " ".join(truth)
        hypo = " ".join(hypo)
    else:
        truth = truth.lower()
        hypo = hypo.lower()

    return wer(truth, hypo)


def load_sim_model(wavlm_ckpt: str, eval_repo: str, device: str):
    sys.path.insert(0, str(Path(eval_repo) / "thirdparty" / "UniSpeech" / "downstreams" / "speaker_verification"))
    from verification import init_model
    return init_model("wavlm_large", wavlm_ckpt).to(device).eval()


def compute_sim(sim_model, wav1_path: str, wav2_path: str, device: str) -> float:
    import librosa
    from torchaudio.transforms import Resample
    import torch.nn.functional as F

    wav1, sr1 = librosa.load(wav1_path, sr=None, mono=False)
    wav2, sr2 = librosa.load(wav2_path, sr=None, mono=False)
    if len(wav1.shape) == 2:
        wav1 = wav1[:, 0]
    if len(wav2.shape) == 2:
        wav2 = wav2[0, :]

    wav1 = torch.from_numpy(wav1).unsqueeze(0).float()
    wav2 = torch.from_numpy(wav2).unsqueeze(0).float()
    wav1 = Resample(orig_freq=sr1, new_freq=16000)(wav1)
    wav2 = Resample(orig_freq=sr2, new_freq=16000)(wav2)

    with torch.no_grad():
        emb1 = sim_model(wav1.to(device))
        emb2 = sim_model(wav2.to(device))
    return F.cosine_similarity(emb1, emb2).item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate SeedTTS metrics")
    parser.add_argument("--testset", type=Path, default=Path("/root/seedtts_testset"),
                        help="SeedTTS testset folder")
    parser.add_argument("--gen_dir", type=Path,
                        default=PROJECT_ROOT / "tools" / "evaluation" / "eval_audio" / "seed_tts_eval",
                        help="Generated audio root (from generate_audio.py)")
    parser.add_argument("--eval_repo", type=Path, default=Path("/root/seed-tts-eval"),
                        help="seed-tts-eval repo path (for WavLM thirdparty code)")
    parser.add_argument("--wavlm_ckpt", type=str, default=None,
                        help="Path to wavlm_large_finetune.pth (required for SIM)")
    parser.add_argument("--metrics", type=str, default="wer,sim",
                        help="Comma-separated metrics to compute: wer, sim")
    parser.add_argument("--lang", type=str, default=None, choices=["zh", "en"],
                        help="Only evaluate a specific language")
    parser.add_argument("--task", type=str, default=None,
                        choices=list(TASK_TO_METAFILE.keys()),
                        help="Only evaluate a specific task")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    metrics = [m.strip() for m in args.metrics.split(",")]

    if "sim" in metrics and not args.wavlm_ckpt:
        print("Error: --wavlm_ckpt is required for SIM metric.")
        print("Download from: https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view")
        sys.exit(1)

    tasks = discover_tasks(args.testset, args.gen_dir)
    if not tasks:
        print(f"No tasks found. Check --testset and --gen_dir paths.")
        sys.exit(1)

    # group by language for ASR model reuse
    from collections import defaultdict
    by_lang = defaultdict(list)
    for lang, task_name, meta_path, gen_dir in tasks:
        if args.lang and lang != args.lang:
            continue
        if args.task and task_name != args.task:
            continue
        by_lang[lang].append((task_name, meta_path, gen_dir))

    print("=" * 60)
    print("SeedTTS Evaluation")
    print(f"  Testset:  {args.testset}")
    print(f"  Gen dir:  {args.gen_dir}")
    print(f"  Metrics:  {metrics}")
    print(f"  Device:   {device}")
    print("=" * 60)

    all_results = {}

    for lang, task_list in sorted(by_lang.items()):
        if "wer" in metrics:
            print(f"\n--- WER [{lang}] ---")
            print(f"Loading ASR model for {lang} ...")
            asr_model = load_asr_model(lang, device)
            print("ASR model loaded.")

        if "sim" in metrics:
            print(f"\nLoading WavLM model ...")
            sim_model = load_sim_model(args.wavlm_ckpt, args.eval_repo, device)
            print("WavLM model loaded.")

        for task_name, meta_path, gen_dir in task_list:
            entries = parse_meta(meta_path)
            n_gen = sum(1 for e in entries if (gen_dir / f'{e["utt"]}.wav').exists())
            print(f"\n>>> {lang}/{task_name}  ({len(entries)} entries, {n_gen} generated)")

            task_key = f"{lang}/{task_name}"
            all_results[task_key] = {}

            if "wer" in metrics:
                wers = []
                for entry in tqdm(entries, desc=f"WER [{lang}/{task_name}]"):
                    wav_path = gen_dir / f"{entry['utt']}.wav"
                    if not wav_path.exists():
                        continue
                    try:
                        hypo = transcribe(asr_model, str(wav_path), lang, device)
                        wer = compute_wer(hypo, entry["gt_text"], lang)
                        wers.append(wer)
                    except Exception as e:
                        print(f"  FAILED {entry['utt']}: {e}")

                avg_wer = round(np.mean(wers) * 100, 3) if wers else float("nan")
                all_results[task_key]["WER(%)"] = avg_wer
                all_results[task_key]["WER_n"] = len(wers)
                print(f"  WER: {avg_wer}%  ({len(wers)} samples)")

            if "sim" in metrics:
                sims = []
                for entry in tqdm(entries, desc=f"SIM [{lang}/{task_name}]"):
                    gen_path = gen_dir / f"{entry['utt']}.wav"
                    ref_path = entry["prompt_wav"]
                    if not gen_path.exists() or not os.path.exists(ref_path):
                        continue
                    try:
                        sim = compute_sim(sim_model, str(gen_path), ref_path, device)
                        sims.append(sim)
                    except Exception as e:
                        print(f"  FAILED {entry['utt']}: {e}")

                avg_sim = round(np.mean(sims), 4) if sims else float("nan")
                all_results[task_key]["SIM"] = avg_sim
                all_results[task_key]["SIM_n"] = len(sims)
                print(f"  SIM: {avg_sim}  ({len(sims)} samples)")

    # summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    header = f"{'Task':<35} {'WER(%)':<10} {'SIM':<10}"
    print(header)
    print("-" * len(header))
    for task_key, res in sorted(all_results.items()):
        wer_str = f"{res.get('WER(%)', '-'):>8}" if "WER(%)" in res else "       -"
        sim_str = f"{res.get('SIM', '-'):>8}" if "SIM" in res else "       -"
        print(f"{task_key:<35} {wer_str} {sim_str}")


if __name__ == "__main__":
    main()
