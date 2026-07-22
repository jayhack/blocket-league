from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .data import make_passive_clip
from .env import PALETTE
from .pixel_direct_model import build_pixel_direct_from_checkpoint
from .pixel_probe import PLAYER_CLASSES, _rollout, _visual_centroid
from .train_pixel_direct import frames_to_classes, palette_tensor


SCENARIOS = (
    {
        "id": "collision",
        "title": "Collision",
        "description": "The discs meet and exchange momentum.",
        "seed": 30_000_014,
        "event": "contact",
    },
    {
        "id": "wall-bounce",
        "title": "Wall bounce",
        "description": "Repeated wall contacts redirect both trajectories.",
        "seed": 30_000_003,
        "event": "wall bounce",
    },
    {
        "id": "goal-reset",
        "title": "Goal + reset",
        "description": "The puck scores, pauses, and the imagined world restarts.",
        "seed": 30_000_019,
        "event": "goal and kickoff",
    },
)


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_classes(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    video = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float().div(127.5).sub(1)
    return frames_to_classes(video[None].to(device), palette_tensor(device))


def _to_rgb(classes: torch.Tensor) -> np.ndarray:
    palette = np.stack(tuple(PALETTE.values()))
    return palette[classes.detach().long().cpu().numpy()]


@torch.no_grad()
def render_hallucination_assets(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    input_frames: int = 12,
    output_frames: int = 36,
    device_name: str = "auto",
) -> dict[str, Any]:
    device = _device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_pixel_direct_from_checkpoint(checkpoint).to(device).eval().requires_grad_(False)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "version": 1,
        "frameSize": model.config.image_size,
        "inputFrames": input_frames,
        "hallucinationFrames": output_frames,
        "checkpointStep": int(checkpoint.get("step", 0)),
        "scenarios": [],
    }

    for scenario in SCENARIOS:
        clip = make_passive_clip(
            int(scenario["seed"]),
            context_frames=input_frames,
            future_frames=output_frames,
            image_size=model.config.image_size,
        )
        context = _to_classes(clip["context"], device)
        prediction = _rollout(model, context, output_frames)
        target = _to_classes(clip["target"], device)
        sequence = torch.cat((context, prediction), dim=1)[0]
        rgb = _to_rgb(sequence)
        frame_size = model.config.image_size
        atlas = np.empty((frame_size, (input_frames + output_frames) * frame_size, 3), dtype=np.uint8)
        for index, frame in enumerate(rgb):
            atlas[:, index * frame_size : (index + 1) * frame_size] = frame
        atlas_name = f"{scenario['id']}.png"
        Image.fromarray(atlas).save(output_dir / atlas_name, optimize=True)

        predicted_player = _visual_centroid(prediction, PLAYER_CLASSES)
        target_player = _visual_centroid(target, PLAYER_CLASSES)
        predicted_puck = _visual_centroid(prediction, (7, 8))
        target_puck = _visual_centroid(target, (7, 8))
        position_error = torch.cat(
            (
                (predicted_player - target_player).norm(dim=-1),
                (predicted_puck - target_puck).norm(dim=-1),
            ),
            dim=0,
        ).mean()
        manifest["scenarios"].append(
            {
                **scenario,
                "atlas": f"/blocket-league/hallucinations/{atlas_name}",
                "meanEntityErrorPixels": round(float(position_error), 2),
            }
        )

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Render long direct-pixel hallucination filmstrips")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--input-frames", type=int, default=12)
    parser.add_argument("--output-frames", type=int, default=36)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = render_hallucination_assets(
        args.checkpoint,
        args.output,
        input_frames=args.input_frames,
        output_frames=args.output_frames,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
