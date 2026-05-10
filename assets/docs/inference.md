# Inference

Please follow these steps to use the model:

## 0. Install the env

```bash
pip install -r requirements.txt
```

## 1. Download models.

```bash
hf download PoTaTo721/Shore-TTS-0.1 --local-dir ./checkpoints/Shore-TTS-0.1
```

## 2. Inference with cli

```bash
python tools/inference.py \
    --checkpoint checkpoints/Shore-TTS-0.1 \
    --text "你们有人想做我的颜料吗？" \
    --ref_audio ./assets/test.wav \
    --ref_text "你们有人想做我的颜料吗" \
    --output test.wav
```

### CLI Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--checkpoint` | `checkpoints/Shore-TTS-0.1` | Path to the model checkpoint directory |
| `--text` | `你们有人想做我的颜料吗？` | Text to synthesize |
| `--ref_audio` | `./assets/test.wav` | Path to reference audio for voice cloning |
| `--ref_text` | `你们有人想做我的颜料吗` | Transcript of the reference audio |
| `--output` | `test.wav` | Output audio file path |
| `--steps` | `16` | Number of ODE sampling steps |
| `--cfg_strength` | `1.0` | Classifier-free guidance strength |
| `--sway_sampling_coef` | `None` | Sway sampling coefficient |
| `--seed` | `None` | Random seed for reproducibility |
| `--speed` | `1.0` | Speech speed factor |
| `--fix_duration` | `None` | Fix output duration in seconds |
| `--duration_factor` | `2.0` | Duration = ref_len * duration_factor |
| `--device` | `None` | Device to use (auto-detected if not set) |

## 3. Inference with Python API

```python
import torch
import torchaudio
from tools.inference import load_model, infer

# Load model
model, sample_rate = load_model("checkpoints/Shore-TTS-0.1", device=torch.device("cuda"))

# Synthesize
wav, sr = infer(
    model=model,
    text="你们有人想做我的颜料吗？",
    ref_audio="./assets/test.wav",
    ref_text="你们有人想做我的颜料吗",
    steps=16,
    speed=1.0,
    device=torch.device("cuda"),
)

# Save output
torchaudio.save("output.wav", wav, sr)
```

### API Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model` | `CFM` | — | Loaded model instance |
| `text` | `str` | — | Text to synthesize |
| `ref_audio` | `str \| Path \| None` | `None` | Reference audio path for voice cloning |
| `ref_text` | `str \| None` | `None` | Transcript of the reference audio |
| `steps` | `int` | `32` | Number of ODE sampling steps |
| `cfg_strength` | `float` | `1.0` | Classifier-free guidance strength |
| `sway_sampling_coef` | `float \| None` | `None` | Sway sampling coefficient |
| `seed` | `int \| None` | `None` | Random seed for reproducibility |
| `max_duration` | `int` | `65536` | Maximum duration in frames |
| `speed` | `float` | `1.0` | Speech speed factor |
| `fix_duration` | `float \| None` | `None` | Fix output duration in seconds |
| `duration_factor` | `float \| None` | `None` | Duration = ref_len * duration_factor |
| `device` | `torch.device` | `cpu` | Device to run inference on |

## 4. Tips

- **Voice cloning**: Provide `ref_audio` and `ref_text` together for best quality. The reference audio should be clean and match the transcript.
- **Duration control**: Currently our model can't predict audio's duration, you need to provide target duration for better performance.