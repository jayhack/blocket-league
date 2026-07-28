from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .pixel_direct_model import build_pixel_direct_from_checkpoint
from .pixel_probe import _context_batch, _fit_ridge, _soft_centroid, _visual_centroid

PUCK_CLASSES = (7, 8)
PLAYER_CLASSES = (5, 6)
READOUTS = ("fixed_bottom_right", "spatial_mean", "puck_token")
INTERVENTIONS = (*READOUTS, "puck_token_jacobian")


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _puck_token(model, classes: torch.Tensor) -> torch.Tensor:
    position = _visual_centroid(classes[:, -1], PUCK_CLASSES)
    patch_x = (position[:, 0] / model.config.patch_size).long().clamp(0, model.config.grid_size - 1)
    patch_y = (position[:, 1] / model.config.patch_size).long().clamp(0, model.config.grid_size - 1)
    return patch_y * model.config.grid_size + patch_x


def _mask(model, classes: torch.Tensor, readout: str) -> torch.Tensor:
    mask = torch.zeros(
        classes.shape[0],
        classes.shape[1],
        model.config.grid_size**2,
        device=classes.device,
    )
    if readout == "fixed_bottom_right":
        mask[:, -1, -1] = 1
    elif readout == "spatial_mean":
        mask[:, -1] = 1
    elif readout in ("puck_token", "puck_token_jacobian"):
        batch = torch.arange(classes.shape[0], device=classes.device)
        mask[batch, -1, _puck_token(model, classes)] = 1
    else:
        raise ValueError(f"Unknown readout: {readout}")
    return mask


@torch.no_grad()
def _block_features(
    model,
    classes: torch.Tensor,
    *,
    block_index: int,
) -> dict[str, torch.Tensor]:
    tokens = (
        model.patch_projection(model.patch_tokens(classes))
        + model.spatial_position
        + model.temporal_position[:, : classes.shape[1]]
    )
    for index, block in enumerate(model.blocks):
        tokens = block(tokens)
        if index == block_index:
            break
    batch = torch.arange(classes.shape[0], device=classes.device)
    puck_token = _puck_token(model, classes)
    return {
        "fixed_bottom_right": tokens[:, -1, -1].float(),
        "spatial_mean": tokens[:, -1].mean(dim=1).float(),
        "puck_token": tokens[batch, -1, puck_token].float(),
    }


def _raw_probe_jacobian(fit) -> torch.Tensor:
    """Return d(predicted x,y)/d(hidden), accounting for feature standardization."""

    _, scale, weight = fit
    return weight[:-1] / scale.squeeze(0)[:, None]


def _minimal_write(fit, desired_delta: torch.Tensor) -> torch.Tensor:
    """Minimum-norm hidden write that produces desired_delta under the linear probe."""

    jacobian = _raw_probe_jacobian(fit)
    gram = jacobian.T @ jacobian
    return jacobian @ torch.linalg.solve(gram + 1e-6 * torch.eye(2), desired_delta)


def _entity_mass(classes: torch.Tensor, values: tuple[int, ...]) -> torch.Tensor:
    return sum(classes.eq(value) for value in values).sum(dim=(-2, -1)).float()


def _puck_jacobian_directions(
    model,
    seeds: list[int],
    *,
    block_index: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    gradient_sums = [torch.zeros(model.config.hidden_size, device=device) for _ in range(2)]
    samples = 0
    for start in range(0, len(seeds), batch_size):
        classes = _context_batch(seeds[start : start + batch_size], model, device)
        write = torch.zeros(
            classes.shape[0],
            model.config.hidden_size,
            device=device,
            requires_grad=True,
        )
        logits = model(
            classes,
            intervention_block=block_index,
            intervention=write,
            intervention_mask=_mask(model, classes, "puck_token"),
        )[:, -1]
        position = _soft_centroid(logits, PUCK_CLASSES)
        for axis in range(2):
            gradient = torch.autograd.grad(
                position[:, axis].sum(),
                write,
                retain_graph=axis == 0,
            )[0]
            gradient_sums[axis] += gradient.sum(dim=0)
        samples += classes.shape[0]
    x = gradient_sums[0] / samples
    y = gradient_sums[1] / samples
    return {
        "x_plus": x / x.norm().clamp_min(1e-8),
        "x_minus": -x / x.norm().clamp_min(1e-8),
        "y_plus": y / y.norm().clamp_min(1e-8),
        "y_minus": -y / y.norm().clamp_min(1e-8),
    }


def _effect(
    baseline: torch.Tensor,
    intervention: torch.Tensor,
) -> dict[str, float]:
    baseline_puck = _visual_centroid(baseline, PUCK_CLASSES)
    intervention_puck = _visual_centroid(intervention, PUCK_CLASSES)
    baseline_player = _visual_centroid(baseline, PLAYER_CLASSES)
    intervention_player = _visual_centroid(intervention, PLAYER_CLASSES)
    delta = intervention_puck - baseline_puck
    player_delta = intervention_player - baseline_player
    baseline_mass = _entity_mass(baseline, PUCK_CLASSES)
    intervention_mass = _entity_mass(intervention, PUCK_CLASSES)
    return {
        "meanPuckDeltaX": float(delta[:, 0].mean()),
        "meanPuckDeltaY": float(delta[:, 1].mean()),
        "medianPuckDisplacement": float(torch.linalg.vector_norm(delta, dim=1).median()),
        "meanPlayerCollateralDisplacement": float(
            torch.linalg.vector_norm(player_delta, dim=1).mean()
        ),
        "expectedXSignFraction": float((delta[:, 0] > 0).float().mean()),
        "expectedYSignFraction": float((delta[:, 1] > 0).float().mean()),
        "meanPuckPixelMassRatio": float(
            (intervention_mass / baseline_mass.clamp_min(1)).mean()
        ),
        "puckPresentFraction": float((intervention_mass > 0).float().mean()),
    }


@torch.no_grad()
def _next_frame(model, classes: torch.Tensor, block_index: int, readout: str, write):
    logits = model(
        classes,
        intervention_block=block_index,
        intervention=write,
        intervention_mask=_mask(model, classes, readout),
    )[:, -1]
    return logits.argmax(dim=1)


@torch.no_grad()
def _rollout(
    model,
    classes: torch.Tensor,
    *,
    frames: int,
    block_index: int,
    readout: str | None = None,
    write: torch.Tensor | None = None,
    write_frames: int = 1,
) -> torch.Tensor:
    history = classes
    generated = []
    for step in range(frames):
        current = history[:, -model.config.history_frames :]
        if readout is not None and write is not None and step < write_frames:
            next_frame = _next_frame(model, current, block_index, readout, write)
        else:
            logits = model(current)[:, -1]
            next_frame = logits.argmax(dim=1)
        generated.append(next_frame)
        history = torch.cat((history, next_frame[:, None]), dim=1)
    return torch.stack(generated, dim=1)


def run_position_write_probe(
    checkpoint_path: Path,
    output_path: Path,
    *,
    fit_samples: int = 1024,
    test_samples: int = 128,
    batch_size: int = 32,
    block: int = 5,
    rollout_frames: int = 12,
    rollout_strength: float = 8.0,
    device_name: str = "auto",
) -> dict[str, Any]:
    device = _device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_pixel_direct_from_checkpoint(checkpoint).to(device).eval().requires_grad_(False)
    block_index = block - 1
    fit_seeds = [31_000_003 + index * 9_973 for index in range(fit_samples)]
    test_seeds = [41_000_009 + index * 9_973 for index in range(test_samples)]

    feature_chunks = {readout: [] for readout in READOUTS}
    target_chunks = []
    for start in range(0, fit_samples, batch_size):
        classes = _context_batch(fit_seeds[start : start + batch_size], model, device)
        features = _block_features(model, classes, block_index=block_index)
        for readout in READOUTS:
            feature_chunks[readout].append(features[readout].cpu())
        target_chunks.append(_visual_centroid(classes[:, -1], PUCK_CLASSES).cpu())
    fit_features = {
        readout: torch.cat(chunks)
        for readout, chunks in feature_chunks.items()
    }
    fit_targets = torch.cat(target_chunks)
    fits = {
        readout: _fit_ridge(features, fit_targets)
        for readout, features in fit_features.items()
    }

    directions = {}
    direction_norms = {}
    for readout, fit in fits.items():
        directions[readout] = {}
        direction_norms[readout] = {}
        for axis, delta in {
            "x_plus": (1.0, 0.0),
            "x_minus": (-1.0, 0.0),
            "y_plus": (0.0, 1.0),
            "y_minus": (0.0, -1.0),
        }.items():
            probe_pixel = _minimal_write(fit, torch.tensor(delta, dtype=torch.float32))
            direction_norms[readout][axis] = float(probe_pixel.norm())
            directions[readout][axis] = (
                probe_pixel / probe_pixel.norm().clamp_min(1e-8)
            ).to(device)
    directions["puck_token_jacobian"] = _puck_jacobian_directions(
        model,
        fit_seeds,
        block_index=block_index,
        batch_size=batch_size,
        device=device,
    )

    next_frame_effects = {
        readout: {
            axis: {}
            for axis in ("x_plus", "x_minus", "y_plus", "y_minus")
        }
        for readout in INTERVENTIONS
    }
    rollout_chunks = {
        readout: {
            axis: []
            for axis in ("x_plus", "x_minus", "y_plus", "y_minus")
        }
        for readout in INTERVENTIONS
    }
    baseline_rollout_chunks = []
    strengths = (1.0, 4.0, 8.0, 16.0)

    for start in range(0, test_samples, batch_size):
        classes = _context_batch(test_seeds[start : start + batch_size], model, device)
        baseline_next = model(classes)[:, -1].argmax(dim=1)
        baseline_rollout = _rollout(
            model,
            classes,
            frames=rollout_frames,
            block_index=block_index,
        )
        baseline_rollout_chunks.append(baseline_rollout.cpu())
        for readout in INTERVENTIONS:
            for axis, unit_direction in directions[readout].items():
                for strength in strengths:
                    intervention = _next_frame(
                        model,
                        classes,
                        block_index,
                        readout,
                        unit_direction * strength,
                    )
                    next_frame_effects[readout][axis].setdefault(str(strength), []).append(
                        (baseline_next.cpu(), intervention.cpu())
                    )
                rollout = _rollout(
                    model,
                    classes,
                    frames=rollout_frames,
                    block_index=block_index,
                    readout=readout,
                    write=unit_direction * rollout_strength,
                    write_frames=1,
                )
                rollout_chunks[readout][axis].append(rollout.cpu())

    next_frame_summary = {}
    for readout in INTERVENTIONS:
        next_frame_summary[readout] = {}
        for axis, dose_chunks in next_frame_effects[readout].items():
            next_frame_summary[readout][axis] = {}
            for strength, chunks in dose_chunks.items():
                baseline = torch.cat([chunk[0] for chunk in chunks])
                intervention = torch.cat([chunk[1] for chunk in chunks])
                effect = _effect(baseline, intervention)
                expected_axis = 0 if axis.startswith("x") else 1
                expected_sign = -1 if axis.endswith("minus") else 1
                delta_key = "meanPuckDeltaX" if expected_axis == 0 else "meanPuckDeltaY"
                effect["expectedAxisMeanDelta"] = effect[delta_key]
                positions = _visual_centroid(intervention, PUCK_CLASSES) - _visual_centroid(
                    baseline, PUCK_CLASSES
                )
                effect["expectedSignFraction"] = float(
                    (positions[:, expected_axis] * expected_sign > 0).float().mean()
                )
                next_frame_summary[readout][axis][strength] = effect

    baseline_rollouts = torch.cat(baseline_rollout_chunks)
    rollout_summary = {}
    for readout in INTERVENTIONS:
        rollout_summary[readout] = {}
        for axis, chunks in rollout_chunks[readout].items():
            intervention = torch.cat(chunks)
            expected_axis = 0 if axis.startswith("x") else 1
            expected_sign = -1 if axis.endswith("minus") else 1
            baseline_position = _visual_centroid(baseline_rollouts, PUCK_CLASSES)
            intervention_position = _visual_centroid(intervention, PUCK_CLASSES)
            delta = intervention_position - baseline_position
            player_delta = _visual_centroid(
                intervention, PLAYER_CLASSES
            ) - _visual_centroid(baseline_rollouts, PLAYER_CLASSES)
            rollout_summary[readout][axis] = {
                "expectedAxisMeanDeltaByFrame": [
                    float(value)
                    for value in delta[:, :, expected_axis].mean(dim=0)
                ],
                "orthogonalAxisMeanDeltaByFrame": [
                    float(value)
                    for value in delta[:, :, 1 - expected_axis].mean(dim=0)
                ],
                "expectedSignFractionAtFinal": float(
                    (delta[:, -1, expected_axis] * expected_sign > 0).float().mean()
                ),
                "finalMeanPlayerCollateralDisplacement": float(
                    torch.linalg.vector_norm(player_delta[:, -1], dim=1).mean()
                ),
                "puckPresentFractionAtFinal": float(
                    (_entity_mass(intervention[:, -1], PUCK_CLASSES) > 0).float().mean()
                ),
            }

    result = {
        "version": 1,
        "question": (
            "Does writing along linearly decoded puck-position directions causally "
            "change the model's rendered puck position?"
        ),
        "checkpoint": str(checkpoint_path),
        "checkpointStep": int(checkpoint["step"]),
        "block": block,
        "fitSamples": fit_samples,
        "testSamples": test_samples,
        "labelSource": "puck centroids measured from rendered categorical pixels",
        "directionMethod": (
            "minimum-norm hidden direction whose linear ridge decoder predicts "
            "a pure x or y coordinate change, normalized to unit activation norm"
        ),
        "readouts": {
            "fixed_bottom_right": "one fixed bottom-right final-frame token",
            "spatial_mean": "same write applied to every final-frame spatial token",
            "puck_token": "token containing the rendered puck centroid",
            "puck_token_jacobian": (
                "same puck token, but oriented by the average downstream gradient "
                "of the next rendered puck centroid"
            ),
        },
        "nextFrameActivationStrengthSweep": list(strengths),
        "nextFrame": next_frame_summary,
        "rollout": {
            "writeFrames": 1,
            "activationStrength": rollout_strength,
            "frames": rollout_frames,
            "effects": rollout_summary,
        },
        "directionNormPerProbePredictedPixel": {
            **direction_norms,
            "puck_token_jacobian": "not a linear-probe direction",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fit-samples", type=int, default=1024)
    parser.add_argument("--test-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block", type=int, default=5)
    parser.add_argument("--rollout-frames", type=int, default=12)
    parser.add_argument("--rollout-strength", type=float, default=8.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = run_position_write_probe(
        args.checkpoint,
        args.output,
        fit_samples=args.fit_samples,
        test_samples=args.test_samples,
        batch_size=args.batch_size,
        block=args.block,
        rollout_frames=args.rollout_frames,
        rollout_strength=args.rollout_strength,
        device_name=args.device,
    )
    for readout in INTERVENTIONS:
        x_effect = result["nextFrame"][readout]["x_plus"]["8.0"]
        rollout = result["rollout"]["effects"][readout]["x_plus"]
        print(
            f"{readout}: strength-8 +x probe write -> next-frame x "
            f"{x_effect['expectedAxisMeanDelta']:+.3f} px; final rollout x "
            f"{rollout['expectedAxisMeanDeltaByFrame'][-1]:+.3f} px"
        )


if __name__ == "__main__":
    main()
