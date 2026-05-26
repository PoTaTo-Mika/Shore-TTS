from __future__ import annotations

import argparse
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.environ["PROJECT_ROOT"] = project_root

import warnings
warnings.filterwarnings("ignore")

from shore_tts.utils.build import build_model, build_discriminator, load_json_config, set_seed
from shore_tts.utils.trainer import Trainer, DiscriminatorTrainer


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

    if "fwd" in config.get("model", {}):
        generator = build_model(config)
        discriminator = build_discriminator(config)
        trainer = DiscriminatorTrainer(generator, discriminator, config)
    else:
        model = build_model(config)
        trainer = Trainer(model, config)

    set_seed(int(config.get("seed", 42)), trainer.accelerator.process_index)
    trainer.train()


if __name__ == "__main__":
    main()
