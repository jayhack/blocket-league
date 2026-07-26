from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .env import BlocketLeagueEnv, WorldConfig
from .pixel_direct_model import build_pixel_direct_from_checkpoint
from .train_pixel_direct import frames_to_classes, palette_tensor


PLAYER_CLASSES = (5, 6)
PUCK_CLASSES = (7, 8)
TASKS = ("disc_impact", "wall_bounce", "goal")


@dataclass(frozen=True)
class EventExample:
    frames: np.ndarray
    label: int


def _passive_config(image_size: int) -> WorldConfig:
    return WorldConfig(
        image_size=image_size,
        player_acceleration=0.0,
        player_drag=0.12,
        puck_drag=0.12,
    )


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-8)


def _render_history(
    env: BlocketLeagueEnv,
    *,
    player_position: np.ndarray,
    player_velocity: np.ndarray,
    puck_position: np.ndarray,
    puck_velocity: np.ndarray,
    history_frames: int,
    score: int,
) -> np.ndarray:
    """Render a valid pre-event history ending at the supplied physical state.

    Over eight frames, passive drag changes velocity by less than five percent.
    Backward constant-velocity rendering keeps matched positive/negative scenes
    exactly aligned in their final frame while preserving the observable motion.
    """

    frames = []
    dt = env.config.dt
    for frame in range(history_frames):
        steps_back = history_frames - frame - 1
        env.state.player_position = (
            player_position - player_velocity * dt * steps_back
        ).astype(np.float32)
        env.state.player_velocity = player_velocity.astype(np.float32).copy()
        env.state.puck_position = (
            puck_position - puck_velocity * dt * steps_back
        ).astype(np.float32)
        env.state.puck_velocity = puck_velocity.astype(np.float32).copy()
        env.state.score = score
        env.state.reset_timer = 0
        env.state.last_event = "coast"
        frames.append(env.render())
    return np.stack(frames)


def _quiet_other_disc(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    position = rng.uniform((0.24, 0.22), (0.48, 0.78)).astype(np.float32)
    return position, np.zeros(2, dtype=np.float32)


def _wall_pair(
    seed: int,
    history_frames: int,
    image_size: int,
) -> tuple[EventExample, EventExample]:
    rng = np.random.default_rng(seed)
    config = _passive_config(image_size)
    use_player = bool(rng.integers(0, 2))
    axis = int(rng.integers(0, 2))
    high_side = bool(rng.integers(0, 2))
    radius = config.player_radius if use_player else config.puck_radius
    speed = float(rng.uniform(0.34, 0.72))
    normal = np.zeros(2, dtype=np.float32)
    normal[axis] = 1.0 if high_side else -1.0
    tangent = np.asarray((-normal[1], normal[0]), dtype=np.float32)
    boundary = 1.0 - config.wall - radius if high_side else config.wall + radius
    current = np.zeros(2, dtype=np.float32)
    current[axis] = boundary - normal[axis] * speed * config.dt * 0.35
    other_axis = 1 - axis
    if not use_player and axis == 0 and high_side:
        # Keep puck wall examples outside the goal mouth.
        current[other_axis] = float(rng.choice((rng.uniform(0.16, 0.29), rng.uniform(0.71, 0.84))))
    else:
        current[other_axis] = rng.uniform(0.18, 0.82)
    positive_velocity = normal * speed
    negative_velocity = tangent * speed * float(rng.choice((-1.0, 1.0)))
    other_position, other_velocity = _quiet_other_disc(rng)
    score = int(rng.integers(0, 5))

    def example(velocity: np.ndarray, expected: bool) -> EventExample:
        env = BlocketLeagueEnv(seed=seed, config=config)
        if use_player:
            player_position, player_velocity = current, velocity
            puck_position, puck_velocity = other_position, other_velocity
        else:
            player_position, player_velocity = other_position, other_velocity
            puck_position, puck_velocity = current, velocity
        frames = _render_history(
            env,
            player_position=player_position,
            player_velocity=player_velocity,
            puck_position=puck_position,
            puck_velocity=puck_velocity,
            history_frames=history_frames,
            score=score,
        )
        env.step(0)
        observed = env.state.last_event == "wall"
        if observed != expected:
            raise RuntimeError(f"Wall construction failed for seed {seed}: {env.state.last_event}")
        return EventExample(frames=frames, label=int(expected))

    return example(positive_velocity, True), example(negative_velocity, False)


def _goal_pair(
    seed: int,
    history_frames: int,
    image_size: int,
) -> tuple[EventExample, EventExample]:
    rng = np.random.default_rng(seed)
    config = _passive_config(image_size)
    speed = float(rng.uniform(0.48, 0.76))
    boundary = 1.0 - config.wall - config.puck_radius
    puck_position = np.asarray(
        (
            boundary - speed * config.dt * 0.35,
            rng.uniform(config.goal_low + 0.055, config.goal_high - 0.055),
        ),
        dtype=np.float32,
    )
    positive_velocity = np.asarray((speed, rng.uniform(-0.02, 0.02)), dtype=np.float32)
    negative_velocity = np.asarray((0.0, speed * float(rng.choice((-1.0, 1.0)))), dtype=np.float32)
    player_position, player_velocity = _quiet_other_disc(rng)
    score = int(rng.integers(0, 5))

    def example(velocity: np.ndarray, expected: bool) -> EventExample:
        env = BlocketLeagueEnv(seed=seed, config=config)
        frames = _render_history(
            env,
            player_position=player_position,
            player_velocity=player_velocity,
            puck_position=puck_position,
            puck_velocity=velocity,
            history_frames=history_frames,
            score=score,
        )
        env.step(0)
        observed = env.state.last_event == "goal"
        if observed != expected:
            raise RuntimeError(f"Goal construction failed for seed {seed}: {env.state.last_event}")
        return EventExample(frames=frames, label=int(expected))

    return example(positive_velocity, True), example(negative_velocity, False)


def _impact_pair(
    seed: int,
    history_frames: int,
    image_size: int,
) -> tuple[EventExample, EventExample]:
    rng = np.random.default_rng(seed)
    config = _passive_config(image_size)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    normal = np.asarray((np.cos(angle), np.sin(angle)), dtype=np.float32)
    tangent = np.asarray((-normal[1], normal[0]), dtype=np.float32)
    closing_speed = float(rng.uniform(0.48, 0.88))
    separation = (
        config.player_radius
        + config.puck_radius
        + closing_speed * config.dt * 0.30
    )
    midpoint = rng.uniform((0.34, 0.34), (0.66, 0.66)).astype(np.float32)
    player_position = midpoint - normal * separation * 0.5
    puck_position = midpoint + normal * separation * 0.5
    positive_player_velocity = normal * closing_speed * 0.5
    positive_puck_velocity = -normal * closing_speed * 0.5
    tangent_sign = float(rng.choice((-1.0, 1.0)))
    negative_player_velocity = tangent * closing_speed * 0.45 * tangent_sign
    negative_puck_velocity = negative_player_velocity.copy()
    score = int(rng.integers(0, 5))

    def example(
        player_velocity: np.ndarray,
        puck_velocity: np.ndarray,
        expected: bool,
    ) -> EventExample:
        env = BlocketLeagueEnv(seed=seed, config=config)
        frames = _render_history(
            env,
            player_position=player_position,
            player_velocity=player_velocity,
            puck_position=puck_position,
            puck_velocity=puck_velocity,
            history_frames=history_frames,
            score=score,
        )
        env.step(0)
        observed = env.state.last_event == "impact"
        if observed != expected:
            raise RuntimeError(f"Impact construction failed for seed {seed}: {env.state.last_event}")
        return EventExample(frames=frames, label=int(expected))

    return (
        example(positive_player_velocity, positive_puck_velocity, True),
        example(negative_player_velocity, negative_puck_velocity, False),
    )


def _make_examples(
    task: str,
    *,
    pairs: int,
    seed: int,
    history_frames: int,
    image_size: int,
) -> list[EventExample]:
    builders = {
        "disc_impact": _impact_pair,
        "wall_bounce": _wall_pair,
        "goal": _goal_pair,
    }
    builder = builders[task]
    examples: list[EventExample] = []
    attempts = 0
    while len(examples) < pairs * 2:
        pair_seed = seed + attempts * 9_973
        attempts += 1
        try:
            positive, negative = builder(pair_seed, history_frames, image_size)
        except RuntimeError:
            if attempts > pairs * 20:
                raise
            continue
        examples.extend((positive, negative))
    return examples


def _visual_centroids(classes: torch.Tensor, values: tuple[int, ...]) -> torch.Tensor:
    mask = torch.zeros_like(classes, dtype=torch.float32)
    for value in values:
        mask.add_(classes.eq(value))
    height, width = classes.shape[-2:]
    x_axis = torch.arange(width, device=classes.device, dtype=torch.float32) + 0.5
    y_axis = torch.arange(height, device=classes.device, dtype=torch.float32) + 0.5
    mass = mask.sum(dim=(-2, -1)).clamp_min(1e-6)
    x = (mask * x_axis).sum(dim=(-2, -1)) / mass
    y = (mask * y_axis[:, None]).sum(dim=(-2, -1)) / mass
    return torch.stack((x, y), dim=-1)


def _centroid_token(classes: torch.Tensor, values: tuple[int, ...], grid_size: int, patch_size: int) -> torch.Tensor:
    position = _visual_centroids(classes[:, -1:], values)[:, 0]
    x, y = position.unbind(dim=-1)
    patch_x = (x / patch_size).long().clamp(0, grid_size - 1)
    patch_y = (y / patch_size).long().clamp(0, grid_size - 1)
    return patch_y * grid_size + patch_x


@torch.no_grad()
def _layer_features(model, classes: torch.Tensor) -> list[torch.Tensor]:
    tokens = (
        model.patch_projection(model.patch_tokens(classes))
        + model.spatial_position
        + model.temporal_position[:, : classes.shape[1]]
    )
    states = [tokens]
    for block in model.blocks:
        tokens = block(tokens)
        states.append(tokens)

    player_token = _centroid_token(
        classes, PLAYER_CLASSES, model.config.grid_size, model.config.patch_size,
    )
    puck_token = _centroid_token(
        classes, PUCK_CLASSES, model.config.grid_size, model.config.patch_size,
    )
    batch = torch.arange(classes.shape[0], device=classes.device)
    features = []
    for state in states:
        last = state[:, -1].float()
        features.append(torch.cat((
            last.mean(dim=1),
            last[batch, player_token],
            last[batch, puck_token],
        ), dim=1))
    return features


def _fit_ridge_classifier(x: torch.Tensor, y: torch.Tensor, ridge: float = 1e-2):
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
    augmented = torch.cat((normalized, torch.ones(normalized.shape[0], 1)), dim=1)
    return augmented @ weight


def _roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    positive_scores = scores[labels.bool()]
    negative_scores = scores[~labels.bool()]
    comparisons = positive_scores[:, None] - negative_scores[None, :]
    return float((comparisons.gt(0).float() + 0.5 * comparisons.eq(0).float()).mean())


def _metrics(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    predictions = scores >= 0.5
    positives = labels.bool()
    negatives = ~positives
    true_positive_rate = (predictions[positives]).float().mean()
    true_negative_rate = (~predictions[negatives]).float().mean()
    paired_positive = scores[0::2]
    paired_negative = scores[1::2]
    pair_difference = paired_positive - paired_negative
    return {
        "roc_auc": _roc_auc(scores, labels),
        "balanced_accuracy": float((true_positive_rate + true_negative_rate) / 2),
        "matched_pair_accuracy": float(
            (pair_difference.gt(0).float() + 0.5 * pair_difference.eq(0).float()).mean()
        ),
    }


def _shuffle_summary(values: list[float]) -> dict[str, float | int]:
    tensor = torch.tensor(values)
    return {
        "repeats": len(values),
        "roc_auc_mean": float(tensor.mean()),
        "roc_auc_std": float(tensor.std(unbiased=False)),
    }


@torch.no_grad()
def _extract_dataset_features(
    model,
    examples: list[EventExample],
    device: torch.device,
    batch_size: int,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    by_layer: list[list[torch.Tensor]] = [[] for _ in range(len(model.blocks) + 1)]
    trajectory_features = []
    labels = []
    palette = palette_tensor(device)
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        videos = torch.from_numpy(np.stack([item.frames for item in batch])).permute(0, 1, 4, 2, 3)
        normalized = videos.to(device=device, dtype=torch.float32).div(127.5).sub(1.0)
        classes = frames_to_classes(normalized, palette)
        features = _layer_features(model, classes)
        trajectory_features.append(torch.cat((
            _visual_centroids(classes, PLAYER_CLASSES),
            _visual_centroids(classes, PUCK_CLASSES),
        ), dim=-1).flatten(1).cpu())
        for depth, values in enumerate(features):
            by_layer[depth].append(values.cpu())
        labels.extend(item.label for item in batch)
    return (
        [torch.cat(values) for values in by_layer],
        torch.cat(trajectory_features),
        torch.tensor(labels, dtype=torch.float32),
    )


def run_event_probes(
    checkpoint_path: Path,
    output_path: Path,
    *,
    fit_pairs: int = 1024,
    test_pairs: int = 512,
    batch_size: int = 64,
    seed: int = 17,
    device_name: str = "auto",
    shuffle_repeats: int = 8,
) -> dict[str, Any]:
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_pixel_direct_from_checkpoint(checkpoint).to(device).eval().requires_grad_(False)
    history_frames = model.config.history_frames
    results: dict[str, Any] = {}

    for task_index, task in enumerate(TASKS):
        fit_examples = _make_examples(
            task,
            pairs=fit_pairs,
            seed=seed + task_index * 10_000_019,
            history_frames=history_frames,
            image_size=model.config.image_size,
        )
        test_examples = _make_examples(
            task,
            pairs=test_pairs,
            seed=seed + 100_000_007 + task_index * 10_000_019,
            history_frames=history_frames,
            image_size=model.config.image_size,
        )
        fit_features, fit_trajectory, fit_labels = _extract_dataset_features(
            model, fit_examples, device, batch_size,
        )
        test_features, test_trajectory, test_labels = _extract_dataset_features(
            model, test_examples, device, batch_size,
        )
        generator = torch.Generator().manual_seed(seed + task_index)
        shuffled_label_sets = [
            fit_labels[torch.randperm(len(fit_labels), generator=generator)]
            for _ in range(shuffle_repeats)
        ]
        trajectory_fit = _fit_ridge_classifier(fit_trajectory, fit_labels)
        trajectory_scores = _predict(trajectory_fit, test_trajectory)
        trajectory_shuffle_aucs = []
        for shuffled_labels in shuffled_label_sets:
            trajectory_shuffle_fit = _fit_ridge_classifier(fit_trajectory, shuffled_labels)
            trajectory_shuffle_scores = _predict(trajectory_shuffle_fit, test_trajectory)
            trajectory_shuffle_aucs.append(_roc_auc(trajectory_shuffle_scores, test_labels))
        layers = []
        for depth, (fit_x, test_x) in enumerate(zip(fit_features, test_features, strict=True)):
            fit = _fit_ridge_classifier(fit_x, fit_labels)
            scores = _predict(fit, test_x)
            shuffle_aucs = []
            for shuffled_labels in shuffled_label_sets:
                shuffle_fit = _fit_ridge_classifier(fit_x, shuffled_labels)
                shuffle_scores = _predict(shuffle_fit, test_x)
                shuffle_aucs.append(_roc_auc(shuffle_scores, test_labels))
            layers.append({
                "stage": "patch embedding" if depth == 0 else f"block {depth}",
                "depth": depth,
                **_metrics(scores, test_labels),
                "shuffled_labels": _shuffle_summary(shuffle_aucs),
            })
        results[task] = {
            "fit_pairs": fit_pairs,
            "test_pairs": test_pairs,
            "positive_definition": f"{task} occurs on the immediately following simulator step",
            "negative_definition": "matched final-frame geometry with non-colliding tangential motion",
            "raw_trajectory_linear_baseline": {
                "features": "pixel-derived player and puck centroids across all eight input frames",
                **_metrics(trajectory_scores, test_labels),
                "shuffled_labels": _shuffle_summary(trajectory_shuffle_aucs),
            },
            "layers": layers,
        }

    payload = {
        "version": 1,
        "model": "passive direct pixel transformer",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "history_frames": history_frames,
        "probe_features": "last-frame spatial mean + pixel-located player token + pixel-located puck token",
        "protocol": (
            "Event-centered matched pairs have identical final-frame geometry and differ in "
            "pre-event motion. Linear ridge probes are fit on frozen activations and evaluated "
            "on disjoint seeds. Labels are never provided to the world model."
        ),
        "tasks": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe frozen pixel-transformer layers for imminent physics events")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fit-pairs", type=int, default=1024)
    parser.add_argument("--test-pairs", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = run_event_probes(
        args.checkpoint,
        args.output,
        fit_pairs=args.fit_pairs,
        test_pairs=args.test_pairs,
        batch_size=args.batch_size,
        seed=args.seed,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
