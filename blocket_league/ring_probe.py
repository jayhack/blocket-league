from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .pixel_direct_model import build_pixel_direct_from_checkpoint
from .pixel_probe import (
    PLAYER_CLASSES,
    _context_batch,
    _entity_token_mask,
    _rollout,
    _visual_centroid,
)


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _gather_player_token(values: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(values.shape[0], device=values.device)
    return values[batch, -1, token].float()


@torch.no_grad()
def _activations(model, classes: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    mask = _entity_token_mask(model, classes)[:, -1]
    token = mask.argmax(dim=1)
    tokens = (
        model.patch_projection(model.patch_tokens(classes))
        + model.spatial_position
        + model.temporal_position[:, : classes.shape[1]]
    )
    residuals = [_gather_player_token(tokens, token)]
    mlp_features: list[torch.Tensor] = []
    for block in model.blocks:
        captured: list[torch.Tensor] = []

        def capture(_module, _inputs, output):
            captured.append(output)

        handle = block.mlp[1].register_forward_hook(capture)
        tokens = block(tokens)
        handle.remove()
        residuals.append(_gather_player_token(tokens, token))
        mlp_features.append(_gather_player_token(captured[0], token))
    return residuals, mlp_features


def _targets(classes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = _visual_centroid(classes, PLAYER_CLASSES)
    velocity = positions[:, -1] - positions[:, -2]
    speed = velocity.norm(dim=1)
    direction = velocity / speed[:, None].clamp_min(1e-6)
    return velocity.cpu(), speed.cpu(), direction.cpu()


def _fit_ridge(x: torch.Tensor, y: torch.Tensor, ridge: float = 1e-2):
    mean = x.mean(0, keepdim=True)
    scale = x.std(0, keepdim=True).clamp_min(1e-5)
    normalized = (x - mean) / scale
    augmented = torch.cat((normalized, torch.ones(normalized.shape[0], 1)), dim=1)
    eye = torch.eye(augmented.shape[1])
    eye[-1, -1] = 0
    weight = torch.linalg.solve(augmented.T @ augmented + ridge * eye, augmented.T @ y)
    return mean, scale, weight


def _predict(fit, x: torch.Tensor) -> torch.Tensor:
    mean, scale, weight = fit
    normalized = (x - mean) / scale
    return torch.cat((normalized, torch.ones(normalized.shape[0], 1)), dim=1) @ weight


def _r2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    residual = (target - prediction).square().sum()
    total = (target - target.mean(0, keepdim=True)).square().sum().clamp_min(1e-8)
    return float(1 - residual / total)


def _angular_mae(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction / prediction.norm(dim=1, keepdim=True).clamp_min(1e-8)
    target = target / target.norm(dim=1, keepdim=True).clamp_min(1e-8)
    radians = torch.acos((prediction * target).sum(dim=1).clamp(-1, 1))
    return float(torch.rad2deg(radians).mean())


def _layer_probes(
    fit_features: list[torch.Tensor],
    test_features: list[torch.Tensor],
    fit_speed: torch.Tensor,
    test_speed: torch.Tensor,
    fit_direction: torch.Tensor,
    test_direction: torch.Tensor,
) -> list[dict[str, float | int | str]]:
    rows = []
    for index, (fit_x, test_x) in enumerate(zip(fit_features, test_features, strict=True)):
        direction_fit = _fit_ridge(fit_x, fit_direction)
        direction_prediction = _predict(direction_fit, test_x)
        speed_fit = _fit_ridge(fit_x, fit_speed[:, None])
        speed_prediction = _predict(speed_fit, test_x)
        rows.append(
            {
                "stage": "patch embedding" if index == 0 else f"block {index}",
                "depth": index,
                "direction_r2": _r2(direction_prediction, test_direction),
                "direction_mae_degrees": _angular_mae(direction_prediction, test_direction),
                "speed_r2": _r2(speed_prediction, test_speed[:, None]),
            }
        )
    return rows


def _unit_tuning(
    fit_x: torch.Tensor,
    test_x: torch.Tensor,
    fit_direction: torch.Tensor,
    test_direction: torch.Tensor,
    *,
    bins: int = 24,
) -> dict[str, Any]:
    design_train = torch.cat((torch.ones(fit_x.shape[0], 1), fit_direction), dim=1)
    design_test = torch.cat((torch.ones(test_x.shape[0], 1), test_direction), dim=1)
    eye = torch.eye(3)
    eye[0, 0] = 0
    beta = torch.linalg.solve(design_train.T @ design_train + 1e-3 * eye, design_train.T @ fit_x)
    prediction = design_test @ beta
    residual = (test_x - prediction).square().sum(dim=0)
    total = (test_x - test_x.mean(0, keepdim=True)).square().sum(dim=0).clamp_min(1e-8)
    unit_r2 = 1 - residual / total
    gain = beta[1:].square().sum(dim=0).sqrt()
    preferred = torch.atan2(beta[2], beta[1])
    tuned = unit_r2 > 0.1

    edges = torch.linspace(-torch.pi, torch.pi, bins + 1)
    bin_index = torch.bucketize(preferred, edges[1:-1])
    weights = gain * unit_r2.clamp_min(0)
    histogram = torch.zeros(bins)
    histogram.scatter_add_(0, bin_index, weights)
    probabilities = histogram / histogram.sum().clamp_min(1e-8)
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum() / np.log(bins)

    preferred_tuned = preferred[tuned].sort().values
    if preferred_tuned.numel() > 1:
        wrapped = torch.cat((preferred_tuned, preferred_tuned[:1] + 2 * torch.pi))
        largest_gap = torch.diff(wrapped).max()
    else:
        largest_gap = torch.tensor(2 * torch.pi)

    angles = torch.atan2(fit_direction[:, 1], fit_direction[:, 0])
    angle_index = torch.bucketize(angles, edges[1:-1])
    means = []
    centers = []
    for index in range(bins):
        selected = angle_index == index
        if selected.sum() < 2:
            continue
        means.append(fit_x[selected].mean(0))
        centers.append(float((edges[index] + edges[index + 1]) / 2))
    mean_matrix = torch.stack(means)
    mean_matrix = (mean_matrix - mean_matrix.mean(0, keepdim=True)) / fit_x.std(0).clamp_min(1e-5)
    _, singular, right = torch.linalg.svd(mean_matrix, full_matrices=False)
    coordinates = mean_matrix @ right[:2].T
    radius = coordinates.norm(dim=1)
    phases = torch.atan2(coordinates[:, 1], coordinates[:, 0]).numpy()
    winding = abs(float(np.diff(np.unwrap(np.r_[phases, phases[0]])).sum() / (2 * np.pi)))

    distances = torch.cdist(mean_matrix, mean_matrix)
    center_tensor = torch.tensor(centers)
    delta = center_tensor[:, None] - center_tensor[None, :]
    chord = 2 * torch.sin(delta.abs() / 2).abs()
    triangle = torch.triu_indices(distances.shape[0], distances.shape[1], offset=1)
    distance_values = distances[triangle[0], triangle[1]]
    chord_values = chord[triangle[0], triangle[1]]
    distance_correlation = float(torch.corrcoef(torch.stack((distance_values, chord_values)))[0, 1])
    top2_variance = float(singular[:2].square().sum() / singular.square().sum().clamp_min(1e-8))
    coordinate_scale = coordinates.norm(dim=1).max().clamp_min(1e-8)
    normalized_coordinates = coordinates / coordinate_scale

    return {
        "units": int(fit_x.shape[1]),
        "tuned_units_r2_gt_0_1": int(tuned.sum()),
        "tuned_fraction": float(tuned.float().mean()),
        "median_positive_r2": float(unit_r2[unit_r2 > 0].median()) if (unit_r2 > 0).any() else 0.0,
        "preferred_direction_entropy": float(entropy),
        "largest_preferred_direction_gap_degrees": float(torch.rad2deg(largest_gap)),
        "preferred_direction_histogram": [float(value) for value in probabilities],
        "angle_bin_population_geometry": {
            "bins_present": len(means),
            "top_2_pc_variance": top2_variance,
            "radial_coefficient_of_variation": float(radius.std() / radius.mean().clamp_min(1e-8)),
            "ordered_winding_number": winding,
            "chord_distance_correlation": distance_correlation,
            "coordinates": [
                {
                    "angle_degrees": float(np.rad2deg(centers[index]) % 360),
                    "x": float(normalized_coordinates[index, 0]),
                    "y": float(normalized_coordinates[index, 1]),
                }
                for index in range(len(centers))
            ],
        },
    }


def _orthogonal_probe_sequence(
    fit_x: torch.Tensor,
    test_x: torch.Tensor,
    fit_y: torch.Tensor,
    test_y: torch.Tensor,
    *,
    max_probes: int,
    stop_r2: float,
) -> dict[str, Any]:
    mean = fit_x.mean(0, keepdim=True)
    scale = fit_x.std(0, keepdim=True).clamp_min(1e-5)
    train_source = ((fit_x - mean) / scale).double()
    test_source = ((test_x - mean) / scale).double()
    target_mean = fit_y.mean(0, keepdim=True).double()
    train_target = fit_y.double() - target_mean
    test_target = test_y.double()
    train = train_source
    test = test_source
    accumulated = torch.empty(train.shape[1], 0, dtype=train.dtype)
    history = []
    removed = 0
    for probe in range(max_probes):
        eye = torch.eye(train.shape[1], dtype=train.dtype)
        weight = torch.linalg.solve(train.T @ train + 1e-2 * eye, train.T @ train_target)
        prediction = test @ weight + target_mean
        score = _r2(prediction.float(), test_target.float())
        row: dict[str, float | int] = {"probe": probe + 1, "r2": score}
        if fit_y.shape[1] == 2:
            row["mae_degrees"] = _angular_mae(prediction.float(), test_target.float())
        history.append(row)
        if score < stop_r2:
            break
        if accumulated.shape[1]:
            weight = weight - accumulated @ (accumulated.T @ weight)
        basis, triangular = torch.linalg.qr(weight, mode="reduced")
        independent = triangular.diag().abs() > 1e-7
        basis = basis[:, independent]
        if not basis.shape[1]:
            break
        accumulated = torch.linalg.qr(torch.cat((accumulated, basis), dim=1), mode="reduced").Q
        train = train_source - (train_source @ accumulated) @ accumulated.T
        test = test_source - (test_source @ accumulated) @ accumulated.T
        removed = accumulated.shape[1]
    return {
        "removed_dimensions_before_threshold": removed,
        "threshold_reached": bool(history[-1]["r2"] < stop_r2),
        "sequence": history,
    }


@torch.no_grad()
def _causal_direction_circle(
    model,
    manifest_path: Path,
    seeds: list[int],
    device: torch.device,
    *,
    batch_size: int,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    causal = manifest["causal"]
    x_direction = torch.tensor(causal["xDirection"], device=device)
    y_direction = torch.tensor(causal["yDirection"], device=device)
    cosine = float(
        torch.dot(x_direction, y_direction)
        / (x_direction.norm() * y_direction.norm()).clamp_min(1e-8)
    )
    angles = torch.arange(8, device=device) * (2 * torch.pi / 8)
    directions = torch.stack(
        [torch.cos(angles[index]) * x_direction + torch.sin(angles[index]) * y_direction for index in range(8)]
    )
    directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-8)
    deltas = [[] for _ in range(8)]
    causal_batch_size = max(1, batch_size // 8)
    for start in range(0, len(seeds), causal_batch_size):
        classes = _context_batch(seeds[start : start + causal_batch_size], model, device)
        baseline = _rollout(model, classes, causal["rolloutFrames"])
        baseline_position = _visual_centroid(baseline, PLAYER_CLASSES)[:, -1]
        count = classes.shape[0]
        expanded_classes = classes[:, None].expand(-1, 8, -1, -1, -1).reshape(
            count * 8, *classes.shape[1:]
        )
        expanded_directions = directions[None].expand(count, -1, -1).reshape(count * 8, -1)
        steered = _rollout(
            model,
            expanded_classes,
            causal["rolloutFrames"],
            block_index=int(causal["block"]) - 1,
            direction=expanded_directions,
            strength=float(causal["strength"]),
            write_frames=int(causal["writeFrames"]),
        )
        steered_position = _visual_centroid(steered, PLAYER_CLASSES)[:, -1].reshape(count, 8, 2)
        delta = steered_position - baseline_position[:, None]
        for index in range(8):
            deltas[index].append(delta[:, index].cpu())

    rows = []
    angular_errors = []
    for index, values in enumerate(deltas):
        delta = torch.cat(values)
        mean = delta.mean(0)
        target = torch.tensor(
            (np.cos(float(angles[index])), np.sin(float(angles[index]))),
            dtype=mean.dtype,
        )
        mean_direction = mean / mean.norm().clamp_min(1e-8)
        error = float(
            torch.rad2deg(torch.acos(torch.dot(mean_direction, target).clamp(-1, 1)))
        )
        angular_errors.append(error)
        rows.append(
            {
                "target_degrees": index * 45,
                "mean_delta_x_pixels": float(mean[0]),
                "mean_delta_y_pixels": float(mean[1]),
                "mean_displacement_pixels": float(mean.norm()),
                "mean_direction_degrees": float(
                    torch.rad2deg(torch.atan2(mean[1], mean[0])) % 360
                ),
                "angular_error_degrees": error,
                "expected_half_plane_fraction": float((delta @ target > 0).float().mean()),
            }
        )
    return {
        "method": "normalized cos(theta) * x-write + sin(theta) * y-write",
        "x_y_direction_cosine": cosine,
        "samples": len(seeds),
        "mean_angular_error_degrees": float(np.mean(angular_errors)),
        "angles": rows,
    }


def run_ring_probe(
    checkpoint_path: Path,
    output_path: Path,
    *,
    fit_samples: int = 2048,
    test_samples: int = 1024,
    batch_size: int = 64,
    device_name: str = "auto",
    causal_manifest_path: Path | None = None,
    causal_samples: int = 256,
) -> dict[str, Any]:
    device = _device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_pixel_direct_from_checkpoint(checkpoint).to(device).eval().requires_grad_(False)
    fit_seeds = [12_000_011 + index * 9_973 for index in range(fit_samples)]
    test_seeds = [29_000_003 + index * 9_973 for index in range(test_samples)]

    def collect(seeds: list[int]):
        residual_batches = [[] for _ in range(len(model.blocks) + 1)]
        mlp_batches = [[] for _ in model.blocks]
        speeds = []
        directions = []
        for start in range(0, len(seeds), batch_size):
            classes = _context_batch(seeds[start : start + batch_size], model, device)
            residuals, mlps = _activations(model, classes)
            _, speed, direction = _targets(classes)
            valid = speed > 0.05
            speeds.append(speed[valid])
            directions.append(direction[valid])
            for index, values in enumerate(residuals):
                residual_batches[index].append(values.cpu()[valid])
            for index, values in enumerate(mlps):
                mlp_batches[index].append(values.cpu()[valid])
        return (
            [torch.cat(values) for values in residual_batches],
            [torch.cat(values) for values in mlp_batches],
            torch.cat(speeds),
            torch.cat(directions),
        )

    fit_residuals, fit_mlps, fit_speed, fit_direction = collect(fit_seeds)
    test_residuals, test_mlps, test_speed, test_direction = collect(test_seeds)
    layer_probes = _layer_probes(
        fit_residuals,
        test_residuals,
        fit_speed,
        test_speed,
        fit_direction,
        test_direction,
    )
    gains = [
        layer_probes[index + 1]["direction_r2"] - layer_probes[index]["direction_r2"]
        for index in range(len(model.blocks))
    ]
    emergence_block = int(np.argmax(gains)) + 1
    peak_block = max(range(1, len(layer_probes)), key=lambda i: layer_probes[i]["direction_r2"])

    mlp_tuning = [
        {"block": index + 1, **_unit_tuning(fit_x, test_x, fit_direction, test_direction)}
        for index, (fit_x, test_x) in enumerate(zip(fit_mlps, test_mlps, strict=True))
    ]
    direction_sequence = _orthogonal_probe_sequence(
        fit_residuals[peak_block],
        test_residuals[peak_block],
        fit_direction,
        test_direction,
        max_probes=min(model.config.hidden_size // 2, 48),
        stop_r2=0.1,
    )
    speed_sequence = _orthogonal_probe_sequence(
        fit_residuals[peak_block],
        test_residuals[peak_block],
        fit_speed[:, None],
        test_speed[:, None],
        max_probes=model.config.hidden_size,
        stop_r2=0.05,
    )

    result = {
        "version": 1,
        "model": "passive direct pixel transformer",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "label_source": "large-disc centroid displacement measured from rendered pixels",
        "fit_samples_requested": fit_samples,
        "test_samples_requested": test_samples,
        "fit_samples_after_motion_filter": int(fit_speed.shape[0]),
        "test_samples_after_motion_filter": int(test_speed.shape[0]),
        "layer_probes": layer_probes,
        "physics_emergence": {
            "criterion": "largest adjacent gain in held-out circular direction R2",
            "block": emergence_block,
            "gain": gains[emergence_block - 1],
            "peak_block": peak_block,
        },
        "mlp_direction_tuning": mlp_tuning,
        "orthogonal_probe_sequences_at_peak": {
            "block": peak_block,
            "direction": direction_sequence,
            "speed": speed_sequence,
        },
    }
    if causal_manifest_path is not None:
        causal_seeds = [41_000_009 + index * 9_973 for index in range(causal_samples)]
        result["causal_direction_circle"] = _causal_direction_circle(
            model,
            causal_manifest_path,
            causal_seeds,
            device,
            batch_size=batch_size,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure circular direction geometry by transformer depth")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fit-samples", type=int, default=2048)
    parser.add_argument("--test-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--causal-manifest", type=Path)
    parser.add_argument("--causal-samples", type=int, default=256)
    args = parser.parse_args()
    result = run_ring_probe(
        args.checkpoint,
        args.output,
        fit_samples=args.fit_samples,
        test_samples=args.test_samples,
        batch_size=args.batch_size,
        device_name=args.device,
        causal_manifest_path=args.causal_manifest,
        causal_samples=args.causal_samples,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
