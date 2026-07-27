from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .pixel_direct_model import build_pixel_direct_from_checkpoint
from .train_pixel_direct import (
    PixelDirectTrainConfig,
    evaluate_collision_quadrants,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a passive pixel checkpoint on collision-centered quadrant splits."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples-per-quadrant", type=int, default=32)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    config_values = {
        key: value
        for key, value in summary["config"].items()
        if key in PixelDirectTrainConfig.__dataclass_fields__
    }
    config = PixelDirectTrainConfig(
        **config_values,
        collision_quadrant_eval_samples=args.samples_per_quadrant,
    )
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    model = build_pixel_direct_from_checkpoint(checkpoint).to(device).eval()
    result = evaluate_collision_quadrants(model, config, device)
    payload = {
        "checkpoint": args.checkpoint.name,
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "device": str(device),
        "evaluation": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
