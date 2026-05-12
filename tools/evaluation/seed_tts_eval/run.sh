#!/usr/bin/env bash
set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CHECKPOINT="${CHECKPOINT:-Shore-TTS-0.1}"
TESTSET="${TESTSET:-/root/seedtts_testset}"
EVAL_REPO="${EVAL_REPO:-/root/seed-tts-eval}"
CKPT_DIR="$PROJECT_ROOT/checkpoints"
STEPS="${STEPS:-16}"
CFG_STRENGTH="${CFG_STRENGTH:-1.0}"
METRICS="${METRICS:-wer,sim}"

# ── Resolve checkpoint ───────────────────────────────────────────────────────
if [[ "$CHECKPOINT" == /* ]]; then
    CKPT_PATH="$CHECKPOINT"
else
    CKPT_PATH="$CKPT_DIR/$CHECKPOINT"
fi

if [ ! -d "$CKPT_PATH" ]; then
    echo "Error: checkpoint not found at $CKPT_PATH"
    exit 1
fi

# ── Download WavLM model for SIM metric ──────────────────────────────────────
WAVLM_DIR="$CKPT_DIR/wavlm"
WAVLM_CKPT="$WAVLM_DIR/wavlm_large_finetune.pth"

if [[ "$METRICS" == *sim* ]]; then
    if [ ! -f "$WAVLM_CKPT" ]; then
        echo "Downloading WavLM checkpoint for SIM metric ..."
        mkdir -p "$WAVLM_DIR"
        pip install -q gdown 2>/dev/null
        gdown "1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP" -O "$WAVLM_CKPT"
        if [ ! -f "$WAVLM_CKPT" ]; then
            echo "Error: failed to download wavlm_large_finetune.pth"
            echo "Please manually download from:"
            echo "  https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view"
            echo "and place it at $WAVLM_CKPT"
            exit 1
        fi
        echo "WavLM checkpoint saved to $WAVLM_CKPT"
    else
        echo "WavLM checkpoint already exists at $WAVLM_CKPT"
    fi
fi

# ── Check testset ────────────────────────────────────────────────────────────
if [ ! -d "$TESTSET" ]; then
    echo "Error: testset not found at $TESTSET"
    echo "Please download the SeedTTS testset and place it at $TESTSET"
    exit 1
fi

if [ ! -d "$EVAL_REPO" ]; then
    echo "Error: seed-tts-eval repo not found at $EVAL_REPO"
    echo "Please clone https://github.com/BytedanceSpeech/seed-tts-eval to $EVAL_REPO"
    exit 1
fi

# ── Step 1: Generate audio ───────────────────────────────────────────────────
echo "============================================"
echo "Step 1/2: Generating audio with Shore-TTS"
echo "  Checkpoint:  $CKPT_PATH"
echo "  Testset:     $TESTSET"
echo "  Steps:       $STEPS"
echo "  CFG strength: $CFG_STRENGTH"
echo "============================================"

python "$SCRIPT_DIR/generate_audio.py" "$TESTSET" \
    --checkpoint "$CKPT_PATH" \
    --steps "$STEPS" \
    --cfg_strength "$CFG_STRENGTH"

# ── Step 2: Evaluate metrics ─────────────────────────────────────────────────
EVAL_ARGS=(
    --testset "$TESTSET"
    --eval_repo "$EVAL_REPO"
    --metrics "$METRICS"
)

if [[ "$METRICS" == *sim* ]]; then
    EVAL_ARGS+=(--wavlm_ckpt "$WAVLM_CKPT")
fi

echo ""
echo "============================================"
echo "Step 2/2: Evaluating metrics"
echo "  Metrics:  $METRICS"
echo "============================================"

python "$SCRIPT_DIR/eval_metrics.py" "${EVAL_ARGS[@]}"

echo ""
echo "Done!"
