import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Extract model weights from training checkpoint for inference.")
    parser.add_argument("checkpoint", type=str, help="Path to the training checkpoint (e.g. model.pt)")
    args = parser.parse_args()

    src = Path(args.checkpoint)
    if not src.exists():
        raise FileNotFoundError(f"Checkpoint not found: {src}")

    ckpt = torch.load(src, map_location="cpu", weights_only=True)

    if "ema_model_state_dict" in ckpt:
        # ema_pytorch stores keys with "ema_model." prefix plus extra keys like "initted", "step"
        raw = ckpt["ema_model_state_dict"]
        state_dict = {k.replace("ema_model.", "", 1): v for k, v in raw.items() if k.startswith("ema_model.")}
        print("Extracted ema_model_state_dict (stripped 'ema_model.' prefix)")
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        print("ema_model_state_dict not found, falling back to model_state_dict")
    else:
        raise KeyError("No model weights found in checkpoint (expected 'ema_model_state_dict' or 'model_state_dict')")

    dst = src.with_suffix(".pth")
    torch.save(state_dict, dst)
    print(f"Saved to {dst}")


if __name__ == "__main__":
    main()
