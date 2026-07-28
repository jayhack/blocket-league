from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .pixel_direct_model import (
    DirectPixelTransformer,
    PixelDirectConfig,
    build_pixel_direct_from_checkpoint,
)
from .pixel_probe import _context_batch, _fit_ridge, _visual_centroid

ENTITY_CLASSES = {"player": (5, 6), "puck": (7, 8)}
READOUTS = ("fixed_bottom_right", "spatial_mean", "player_token", "puck_token")


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _position_token(model, position: torch.Tensor) -> torch.Tensor:
    patch_x = (position[:, 0] / model.config.patch_size).long().clamp(
        0, model.config.grid_size - 1
    )
    patch_y = (position[:, 1] / model.config.patch_size).long().clamp(
        0, model.config.grid_size - 1
    )
    return patch_y * model.config.grid_size + patch_x


@torch.no_grad()
def _readouts(model, classes: torch.Tensor, positions: dict[str, torch.Tensor]):
    tokens = (
        model.patch_projection(model.patch_tokens(classes))
        + model.spatial_position
        + model.temporal_position[:, : classes.shape[1]]
    )
    states = [tokens]
    for block in model.blocks:
        tokens = block(tokens)
        states.append(tokens)
    batch = torch.arange(classes.shape[0], device=classes.device)
    player_token = _position_token(model, positions["player"])
    puck_token = _position_token(model, positions["puck"])
    result = {readout: [] for readout in READOUTS}
    for state in states:
        result["fixed_bottom_right"].append(state[:, -1, -1].float().cpu())
        result["spatial_mean"].append(state[:, -1].mean(dim=1).float().cpu())
        result["player_token"].append(
            state[batch, -1, player_token].float().cpu()
        )
        result["puck_token"].append(state[batch, -1, puck_token].float().cpu())
    return result


@torch.no_grad()
def _collect(model, seeds: list[int], batch_size: int, device: torch.device):
    feature_chunks = {
        readout: [[] for _ in range(model.config.depth + 1)]
        for readout in READOUTS
    }
    position_chunks = {entity: [] for entity in ENTITY_CLASSES}
    for start in range(0, len(seeds), batch_size):
        classes = _context_batch(seeds[start : start + batch_size], model, device)
        positions = {
            entity: _visual_centroid(classes[:, -1], values)
            for entity, values in ENTITY_CLASSES.items()
        }
        readouts = _readouts(model, classes, positions)
        for readout, stages in readouts.items():
            for stage, features in enumerate(stages):
                feature_chunks[readout][stage].append(features)
        for entity, position in positions.items():
            position_chunks[entity].append(position.cpu())
        done = min(start + batch_size, len(seeds))
        if done % 512 == 0 or done == len(seeds):
            print(f"collected {done}/{len(seeds)}", flush=True)
    return (
        {
            readout: [torch.cat(chunks) for chunks in stages]
            for readout, stages in feature_chunks.items()
        },
        {
            entity: torch.cat(chunks)
            for entity, chunks in position_chunks.items()
        },
    )


def _predict(fit, features: torch.Tensor) -> torch.Tensor:
    mean, scale, weight = fit
    normalized = (features - mean) / scale
    augmented = torch.cat(
        (normalized, torch.ones(features.shape[0], 1)),
        dim=1,
    )
    return augmented @ weight


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    residual = target - prediction
    axis_total = (target - target.mean(dim=0, keepdim=True)).square().sum(dim=0)
    axis_r2 = 1 - residual.square().sum(dim=0) / axis_total.clamp_min(1e-8)
    total_r2 = 1 - residual.square().sum() / (
        target - target.mean(dim=0, keepdim=True)
    ).square().sum().clamp_min(1e-8)
    return {
        "r2": float(total_r2),
        "xR2": float(axis_r2[0]),
        "yR2": float(axis_r2[1]),
        "rmsePx": float(residual.square().mean().sqrt()),
        "medianEuclideanErrorPx": float(
            torch.linalg.vector_norm(residual, dim=1).median()
        ),
    }


def _evaluate(
    features: list[torch.Tensor],
    target: torch.Tensor,
    fit_mask: torch.Tensor,
    test_mask: torch.Tensor,
) -> list[dict[str, Any]]:
    rows = []
    for stage, values in enumerate(features):
        fit = _fit_ridge(values[fit_mask], target[fit_mask])
        rows.append(
            {
                "stage": "embedding" if stage == 0 else f"block_{stage}",
                **_metrics(_predict(fit, values[test_mask]), target[test_mask]),
            }
        )
    return rows


def _model_results(
    model,
    seeds: list[int],
    *,
    fit_samples: int,
    quadrant_fit_samples: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    features, positions = _collect(model, seeds, batch_size, device)
    index = torch.arange(len(seeds))
    ordinary_fit = index < fit_samples
    ordinary_test = index >= quadrant_fit_samples
    result = {}
    for entity in ENTITY_CLASSES:
        target = positions[entity]
        upper_right = (target[:, 0] >= model.config.image_size / 2) & (
            target[:, 1] < model.config.image_size / 2
        )
        quadrant_fit = (index < quadrant_fit_samples) & ~upper_right
        quadrant_test = (index >= quadrant_fit_samples) & upper_right
        readouts = ("fixed_bottom_right", "spatial_mean", f"{entity}_token")
        result[entity] = {
            "sampleCounts": {
                "ordinaryFit": int(ordinary_fit.sum()),
                "ordinaryTest": int(ordinary_test.sum()),
                "quadrantFit": int(quadrant_fit.sum()),
                "quadrantTest": int(quadrant_test.sum()),
            },
            "readouts": {
                readout: {
                    "ordinaryHeldOutTrajectories": _evaluate(
                        features[readout], target, ordinary_fit, ordinary_test
                    ),
                    "upperRightNeverSeenByProbe": _evaluate(
                        features[readout], target, quadrant_fit, quadrant_test
                    ),
                }
                for readout in readouts
            },
        }
    return result


def run_position_geometry_probe(
    checkpoint_path: Path,
    output_path: Path,
    *,
    samples: int = 4096,
    fit_samples: int = 2048,
    quadrant_fit_samples: int = 3072,
    batch_size: int = 32,
    seed: int = 24_071_991,
    device_name: str = "auto",
) -> dict[str, Any]:
    device = _device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    trained = build_pixel_direct_from_checkpoint(checkpoint).to(device).eval().requires_grad_(False)
    seeds = [seed + index * 9_973 for index in range(samples)]
    print("trained model", flush=True)
    trained_result = _model_results(
        trained,
        seeds,
        fit_samples=fit_samples,
        quadrant_fit_samples=quadrant_fit_samples,
        batch_size=batch_size,
        device=device,
    )
    del trained
    if device.type == "mps":
        torch.mps.empty_cache()

    torch.manual_seed(91_337)
    random_model = DirectPixelTransformer(
        PixelDirectConfig(**checkpoint["model_config"])
    ).to(device).eval().requires_grad_(False)
    print("random-weight control", flush=True)
    random_result = _model_results(
        random_model,
        seeds,
        fit_samples=fit_samples,
        quadrant_fit_samples=quadrant_fit_samples,
        batch_size=batch_size,
        device=device,
    )
    result = {
        "version": 1,
        "question": (
            "Can a linear decoder recover Cartesian player and puck coordinates "
            "from one fixed final-frame token?"
        ),
        "checkpoint": str(checkpoint_path),
        "checkpointStep": int(checkpoint["step"]),
        "samples": samples,
        "labelSource": "centroids measured from rendered categorical pixels",
        "upperRightDefinition": "x >= 32 px and y < 32 px",
        "trained": trained_result,
        "randomWeightControl": random_result,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--fit-samples", type=int, default=2048)
    parser.add_argument("--quadrant-fit-samples", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_position_geometry_probe(
        args.checkpoint,
        args.output,
        samples=args.samples,
        fit_samples=args.fit_samples,
        quadrant_fit_samples=args.quadrant_fit_samples,
        batch_size=args.batch_size,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
