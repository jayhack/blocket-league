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
from .pixel_probe import _context_batch, _fit_ridge, _visual_centroid
from .position_write_probe import (
    PLAYER_CLASSES,
    PUCK_CLASSES,
    _block_features,
)
from .train_pixel_direct import frames_to_classes, palette_tensor

SCENARIOS = (
    {
        "id": "collision",
        "title": "Collision",
        "description": "The coordinate readout follows both discs through contact.",
        "seed": 30_000_044,
    },
    {
        "id": "wall-bounce",
        "title": "Wall bounce",
        "description": "The decoded positions turn with the hallucinated bounce.",
        "seed": 30_000_003,
    },
    {
        "id": "goal-reset",
        "title": "Goal + reset",
        "description": "The readout tracks the puck into the goal and through reset.",
        "seed": 30_000_019,
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
    video = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2)
    normalized = video[None].to(device=device, dtype=torch.float32).div(127.5).sub(1)
    return frames_to_classes(normalized, palette_tensor(device))


def _to_rgb(classes: torch.Tensor) -> np.ndarray:
    palette = np.stack(tuple(PALETTE.values()))
    return palette[classes.detach().long().cpu().numpy()]


def _predict(fit, features: torch.Tensor) -> torch.Tensor:
    mean, scale, weight = fit
    normalized = (features.cpu() - mean) / scale
    augmented = torch.cat(
        (normalized, torch.ones(features.shape[0], 1)),
        dim=1,
    )
    return augmented @ weight


@torch.no_grad()
def render_position_rollout_assets(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    fit_samples: int = 2048,
    batch_size: int = 32,
    block: int = 5,
    input_frames: int = 8,
    output_frames: int = 20,
    device_name: str = "auto",
) -> dict[str, Any]:
    device = _device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_pixel_direct_from_checkpoint(checkpoint).to(device).eval().requires_grad_(False)
    block_index = block - 1

    feature_chunks = []
    target_chunks = []
    fit_seeds = [51_000_007 + index * 9_973 for index in range(fit_samples)]
    for start in range(0, fit_samples, batch_size):
        classes = _context_batch(fit_seeds[start : start + batch_size], model, device)
        features = _block_features(model, classes, block_index=block_index)
        player = _visual_centroid(classes[:, -1], PLAYER_CLASSES)
        puck = _visual_centroid(classes[:, -1], PUCK_CLASSES)
        feature_chunks.append(features["fixed_bottom_right"].cpu())
        target_chunks.append(torch.cat((player, puck), dim=1).cpu())
    fit = _fit_ridge(torch.cat(feature_chunks), torch.cat(target_chunks))

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "version": 1,
        "frameSize": model.config.image_size,
        "inputFrames": input_frames,
        "hallucinationFrames": output_frames,
        "playbackFps": 7,
        "checkpointStep": int(checkpoint["step"]),
        "fitSamples": fit_samples,
        "block": block,
        "readout": (
            "linear ridge from the fixed bottom-right token after the final "
            "observed frame"
        ),
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
        history = context
        generated = []
        decoded = [None] * (input_frames - 1)

        initial_features = _block_features(
            model,
            history[:, -model.config.history_frames :],
            block_index=block_index,
        )
        initial_position = _predict(
            fit, initial_features["fixed_bottom_right"]
        )[0]
        decoded.append([round(float(value), 3) for value in initial_position])

        for _ in range(output_frames):
            current = history[:, -model.config.history_frames :]
            next_frame = model(current)[:, -1].argmax(dim=1)
            generated.append(next_frame)
            history = torch.cat((history, next_frame[:, None]), dim=1)
            features = _block_features(
                model,
                history[:, -model.config.history_frames :],
                block_index=block_index,
            )
            position = _predict(fit, features["fixed_bottom_right"])[0]
            decoded.append([round(float(value), 3) for value in position])

        generated_tensor = torch.stack(generated, dim=1)
        sequence = torch.cat((context, generated_tensor), dim=1)[0]
        frame_size = model.config.image_size
        atlas = np.empty(
            (frame_size, (input_frames + output_frames) * frame_size, 3),
            dtype=np.uint8,
        )
        for index, frame in enumerate(_to_rgb(sequence)):
            atlas[:, index * frame_size : (index + 1) * frame_size] = frame
        atlas_name = f"{scenario['id']}.png"
        Image.fromarray(atlas).save(output_dir / atlas_name, optimize=True)

        visual_player = _visual_centroid(sequence, PLAYER_CLASSES)
        visual_puck = _visual_centroid(sequence, PUCK_CLASSES)
        visual = torch.cat((visual_player, visual_puck), dim=1)
        decoded_tensor = torch.tensor(
            [values for values in decoded if values is not None],
            dtype=torch.float32,
        )
        visual_tail = visual[input_frames - 1 :].cpu()
        player_error = torch.linalg.vector_norm(
            decoded_tensor[:, :2] - visual_tail[:, :2], dim=1
        )
        puck_error = torch.linalg.vector_norm(
            decoded_tensor[:, 2:] - visual_tail[:, 2:], dim=1
        )

        manifest["scenarios"].append(
            {
                **scenario,
                "atlas": f"/blocket-league/position-rollouts/{atlas_name}",
                "decodedPositions": decoded,
                "meanPlayerErrorPx": round(float(player_error.mean()), 2),
                "meanPuckErrorPx": round(float(puck_error.mean()), 2),
                "meanEntityErrorPx": round(
                    float(torch.cat((player_error, puck_error)).mean()), 2
                ),
            }
        )

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fit-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block", type=int, default=5)
    parser.add_argument("--input-frames", type=int, default=8)
    parser.add_argument("--output-frames", type=int, default=20)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = render_position_rollout_assets(
        args.checkpoint,
        args.output,
        fit_samples=args.fit_samples,
        batch_size=args.batch_size,
        block=args.block,
        input_frames=args.input_frames,
        output_frames=args.output_frames,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
