from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .data import make_passive_clip
from .metrics import trajectory_metrics
from .pixel_direct_model import build_pixel_direct_from_checkpoint
from .train_pixel_direct import classes_to_video, frames_to_classes, palette_tensor, rollout_pixel_classes


EVENT_NAMES = ("coast", "coast", "collision", "wall bounce", "goal", "kickoff")
SCENARIOS = (
    {
        "id": "collision",
        "title": "Collision",
        "description": "The discs meet and exchange momentum.",
        "seed": 2_000_005,
        "goal_centered": False,
    },
    {
        "id": "wall-bounce",
        "title": "Wall bounce",
        "description": "A moving body reaches the arena boundary and reverses.",
        "seed": 2_000_001,
        "goal_centered": False,
    },
    {
        "id": "goal-reset",
        "title": "Goal + reset",
        "description": "The sequence crosses a score, pause, and deterministic moving kickoff.",
        "seed": 2_000_000,
        "goal_centered": True,
    },
)


def _video_to_uint8(video: torch.Tensor) -> np.ndarray:
    return (
        video.detach()
        .float()
        .clamp(-1, 1)
        .add(1)
        .mul(127.5)
        .round()
        .byte()
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )


def _video_tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(value.copy())
        .permute(0, 3, 1, 2)
        .float()
        .div(127.5)
        .sub(1.0)
        .to(device)
    )


@torch.no_grad()
def render_experiment_assets(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    rollout_frames: int = 64,
    playback_fps: int = 8,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_pixel_direct_from_checkpoint(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    config = model.config
    palette = palette_tensor(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_boundary = min(12, rollout_frames)
    manifest: dict[str, Any] = {
        "version": 1,
        "frameSize": config.image_size,
        "contextFrames": config.history_frames,
        "futureFrames": rollout_frames,
        "playbackFps": playback_fps,
        "checkpointStep": int(checkpoint.get("step", 0)),
        "generationLabel": "deterministic pixel AR",
        "metricBoundary": metric_boundary,
        "scenarios": [],
    }

    for scenario in SCENARIOS:
        clip = make_passive_clip(
            int(scenario["seed"]),
            context_frames=config.history_frames,
            future_frames=rollout_frames,
            image_size=config.image_size,
            goal_centered=bool(scenario["goal_centered"]),
        )
        context = _video_tensor(clip["context"], device)[None]
        truth = _video_tensor(clip["target"], device)[None]
        states = torch.from_numpy(clip["state"].copy()).float().to(device)[None]
        classes = frames_to_classes(context, palette)
        prediction_classes = rollout_pixel_classes(model, classes, rollout_frames)
        prediction = classes_to_video(prediction_classes, palette)

        context_frames = _video_to_uint8(context[0])
        rows = (
            np.concatenate((context_frames, _video_to_uint8(truth[0])), axis=0),
            np.concatenate((context_frames, _video_to_uint8(prediction[0])), axis=0),
        )
        total_frames = config.history_frames + rollout_frames
        atlas = np.empty(
            (len(rows) * config.image_size, total_frames * config.image_size, 3),
            dtype=np.uint8,
        )
        for row_index, row in enumerate(rows):
            for column, frame in enumerate(row):
                top = row_index * config.image_size
                left = column * config.image_size
                atlas[top : top + config.image_size, left : left + config.image_size] = frame

        atlas_name = f"{scenario['id']}.png"
        Image.fromarray(atlas).save(output_dir / atlas_name, optimize=True)
        metrics = trajectory_metrics(prediction, states)
        short_metrics = trajectory_metrics(
            prediction[:, :metric_boundary],
            states[:, :metric_boundary],
        )
        events = [
            EVENT_NAMES[int(event)] if int(event) < len(EVENT_NAMES) else "coast"
            for event in clip["events"][:rollout_frames]
        ]
        manifest["scenarios"].append(
            {
                "id": scenario["id"],
                "title": scenario["title"],
                "description": scenario["description"],
                "seed": scenario["seed"],
                "atlas": atlas_name,
                "events": events,
                "lanes": [
                    {
                        "id": "truth",
                        "label": "Ground truth",
                        "kind": "truth",
                        "playerErrorPx": 0.0,
                        "puckErrorPx": 0.0,
                    },
                    {
                        "id": "checkpoint",
                        "label": "Nano prediction",
                        "kind": "sample",
                        "playerErrorPx": round(metrics["player_position_error_px"], 2),
                        "puckErrorPx": round(metrics["puck_position_error_px"], 2),
                        "shortPlayerErrorPx": round(
                            short_metrics["player_position_error_px"], 2
                        ),
                        "shortPuckErrorPx": round(
                            short_metrics["puck_position_error_px"], 2
                        ),
                    },
                ],
            }
        )

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a registered experiment manifest and sample atlases."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--rollout-frames", type=int, default=64)
    parser.add_argument("--playback-fps", type=int, default=8)
    args = parser.parse_args()
    manifest = render_experiment_assets(
        args.checkpoint,
        args.output_dir,
        rollout_frames=args.rollout_frames,
        playback_fps=args.playback_fps,
    )
    print(json.dumps({"output": str(args.output_dir), "scenarios": len(manifest["scenarios"])}, indent=2))


if __name__ == "__main__":
    main()

