# Training

Please follow these steps to train the model:

## 0. Install the env

```bash
pip install -r requirements.txt
```

For multi-GPU training, make sure `accelerate` is configured, but I recommend to keep the default config:

```bash
accelerate config
```

## 1. Prepare the dataset

Shore-TTS uses WebDataset format (`.tar` files). Each tar archive should contain:

- An audio file (`.wav`, `.mp3`, `.m4a`, `.ogg`, or `.opus`)
- A text transcript (`.txt` or `.json` with a `text` field)

Example tar structure for a sample:

```
sample_001.wav    # audio
sample_001.txt    # transcript
```

Or with JSON metadata:

```
sample_001.wav
sample_001.json   # {"text": "你好世界"}
```

Place all `.tar` files under a single directory (e.g., `/data/Emilia`).

### Build vocab

Before training, generate the pinyin vocabulary file:

```bash
python -m shore_tts.text.tokenizer \
    --data_path /data/Emilia \
    --output_path checkpoints/vocab/vocab.json
```

Or just use the file we provided.

## 2. Configure training

Create or modify a training config JSON. The default config is at `shore_tts/configs/pretrain.json`.

## 3. Start training

Single GPU:

```bash
python shore_tts/train.py --config shore_tts/configs/pretrain.json
```

Multi-GPU with `accelerate`:

```bash
accelerate launch shore_tts/train.py --config shore_tts/configs/pretrain.json
```

Or with `torchrun` (not tested):

```bash
torchrun --nproc_per_node=8 shore_tts/train.py --config shore_tts/configs/pretrain.json
```

## 4. Resume training

Set `resume_from` in your config to the checkpoint path and re-run:

```json
{
  "train": {
    "resume_from": "checkpoints/pretrain-200M/model_last.pt"
  }
}
```

This restores model weights, optimizer state, scheduler state, EMA weights, and the training step counter.

## 5. Logging

You can use tensorboard to watch the metrics.

Logged metrics:

- `train/loss` — total CFM loss
- `train/loss_low_freq` — low-frequency band loss
- `train/loss_high_freq` — high-frequency band loss
- `train/grad_norm` — gradient norm after clipping
- `train/lr` — current learning rate
