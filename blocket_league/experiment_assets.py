from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .data import make_passive_clip, points_in_quadrant
from .metrics import trajectory_metrics
from .pixel_direct_model import build_pixel_direct_from_checkpoint
from .train_pixel_direct import classes_to_video, frames_to_classes, palette_tensor, rollout_pixel_classes


EVENT_NAMES = ("coast", "coast", "collision", "wall bounce", "goal", "kickoff")
DEFAULT_SCENARIOS = (
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

DIRECTION_SCENARIOS = (
    {
        "id": "east-north-20",
        "title": "−20°",
        "description": "Held-out eastward motion, twenty degrees above the horizontal.",
        "seed": 4_800_000,
        "puck_angle_center_degrees": -20.0,
    },
    {
        "id": "due-east",
        "title": "0°",
        "description": "Due-east puck motion at the center of the excluded sixty-degree wedge.",
        "seed": 4_900_000,
        "puck_angle_center_degrees": 0.0,
    },
    {
        "id": "east-south-20",
        "title": "+20°",
        "description": "Held-out eastward motion, twenty degrees below the horizontal.",
        "seed": 5_000_000,
        "puck_angle_center_degrees": 20.0,
    },
)

COLLISION_QUADRANT_SCENARIOS = (
    {
        "id": "upper-left",
        "title": "Upper Left",
        "description": "A comparison collision centered in the upper-left quadrant.",
        "seed": 6_000_000,
        "collision_quadrant": "upper-left",
    },
    {
        "id": "upper-right-1",
        "title": "Far Upper Right · 1",
        "description": "Held-out collision deep in the upper-right corner, sample one.",
        "seed": 7_000_000,
        "collision_quadrant": "upper-right",
        "collision_min_x": 0.70,
        "collision_max_y": 0.30,
    },
    {
        "id": "upper-right-2",
        "title": "Far Upper Right · 2",
        "description": "Held-out collision deep in the upper-right corner, sample two.",
        "seed": 7_700_003,
        "collision_quadrant": "upper-right",
        "collision_min_x": 0.70,
        "collision_max_y": 0.30,
    },
    {
        "id": "upper-right-3",
        "title": "Far Upper Right · 3",
        "description": "Held-out collision deep in the upper-right corner, sample three.",
        "seed": 8_400_007,
        "collision_quadrant": "upper-right",
        "collision_min_x": 0.70,
        "collision_max_y": 0.30,
    },
    {
        "id": "upper-right-4",
        "title": "Far Upper Right · 4",
        "description": "Held-out collision deep in the upper-right corner, sample four.",
        "seed": 9_100_009,
        "collision_quadrant": "upper-right",
        "collision_min_x": 0.70,
        "collision_max_y": 0.30,
    },
    {
        "id": "lower-left",
        "title": "Lower Left",
        "description": "A comparison collision centered in the lower-left quadrant.",
        "seed": 8_000_000,
        "collision_quadrant": "lower-left",
    },
    {
        "id": "lower-right",
        "title": "Lower Right",
        "description": "A comparison collision centered in the lower-right quadrant.",
        "seed": 9_000_000,
        "collision_quadrant": "lower-right",
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


def _direction_clip(
    scenario: dict[str, Any],
    *,
    context_frames: int,
    future_frames: int,
    image_size: int,
) -> dict[str, np.ndarray]:
    """Find a clip whose complete observed puck motion stays in the held-out wedge."""
    center = float(scenario["puck_angle_center_degrees"])
    for attempt in range(512):
        clip = make_passive_clip(
            int(scenario["seed"]) + attempt * 9_973,
            context_frames=context_frames,
            future_frames=future_frames,
            image_size=image_size,
            puck_angle_center_degrees=center,
            puck_angle_width_degrees=4.0,
        )
        velocities = clip["all_state"][:context_frames, 6:8]
        speeds = np.linalg.norm(velocities, axis=1)
        angles = np.rad2deg(np.arctan2(velocities[:, 1], velocities[:, 0]))
        east_delta = (angles + 180.0) % 360.0 - 180.0
        if np.all(speeds > 0.05) and np.all(np.abs(east_delta) < 30.0):
            return clip
    raise RuntimeError(f"could not find a held-out direction clip for {scenario['id']}")


def _collision_quadrant_clip(
    scenario: dict[str, Any],
    *,
    context_frames: int,
    future_frames: int,
    image_size: int,
) -> dict[str, np.ndarray]:
    """Find a clip with an early collision centered in the requested quadrant."""
    quadrant = str(scenario["collision_quadrant"])
    metric_frames = min(12, future_frames)
    for attempt in range(2_048):
        clip = make_passive_clip(
            int(scenario["seed"]) + attempt * 9_973,
            context_frames=context_frames,
            future_frames=future_frames,
            image_size=image_size,
        )
        impact_indices = np.flatnonzero(clip["events"][:metric_frames] == 2)
        for impact_index in impact_indices:
            state = clip["state"][impact_index]
            midpoint = ((state[:2] + state[4:6]) / 2.0)[None]
            location_matches = bool(points_in_quadrant(midpoint, quadrant)[0])
            if "collision_min_x" in scenario:
                location_matches = location_matches and bool(
                    midpoint[0, 0] >= float(scenario["collision_min_x"])
                )
            if "collision_max_y" in scenario:
                location_matches = location_matches and bool(
                    midpoint[0, 1] <= float(scenario["collision_max_y"])
                )
            if location_matches:
                return clip
    raise RuntimeError(f"could not find a collision clip in {quadrant}")


@torch.no_grad()
def render_experiment_assets(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    rollout_frames: int = 64,
    playback_fps: int = 8,
    scenario_set: str = "default",
    lane_label: str = "Checkpoint prediction",
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

    if scenario_set == "default":
        scenarios = DEFAULT_SCENARIOS
    elif scenario_set == "direction-holdout":
        scenarios = DIRECTION_SCENARIOS
    elif scenario_set == "collision-quadrants":
        scenarios = COLLISION_QUADRANT_SCENARIOS
    else:
        raise ValueError(f"unknown scenario_set: {scenario_set}")

    for scenario in scenarios:
        if scenario_set == "direction-holdout":
            clip = _direction_clip(
                scenario,
                context_frames=config.history_frames,
                future_frames=rollout_frames,
                image_size=config.image_size,
            )
        elif scenario_set == "collision-quadrants":
            clip = _collision_quadrant_clip(
                scenario,
                context_frames=config.history_frames,
                future_frames=rollout_frames,
                image_size=config.image_size,
            )
        else:
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
                        "label": lane_label,
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
    parser.add_argument(
        "--scenario-set",
        choices=("default", "direction-holdout", "collision-quadrants"),
        default="default",
    )
    parser.add_argument("--lane-label", default="Checkpoint prediction")
    args = parser.parse_args()
    manifest = render_experiment_assets(
        args.checkpoint,
        args.output_dir,
        rollout_frames=args.rollout_frames,
        playback_fps=args.playback_fps,
        scenario_set=args.scenario_set,
        lane_label=args.lane_label,
    )
    print(json.dumps({"output": str(args.output_dir), "scenarios": len(manifest["scenarios"])}, indent=2))


if __name__ == "__main__":
    main()
