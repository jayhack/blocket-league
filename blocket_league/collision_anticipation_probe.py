from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .env import BlocketLeagueEnv
from .event_probe import (
    EventExample,
    _extract_dataset_features,
    _fit_ridge_classifier,
    _metrics,
    _passive_config,
    _predict,
    _render_history,
    _roc_auc,
    _shuffle_summary,
)
from .pixel_direct_model import (
    DirectPixelTransformer,
    PixelDirectConfig,
    build_pixel_direct_from_checkpoint,
)


DEFAULT_HORIZONS = (1, 2, 4, 6, 8)


def _effective_travel_time(env: BlocketLeagueEnv, steps: int) -> float:
    """Distance traveled per unit initial speed after ``steps`` simulator steps."""

    if steps <= 0:
        return 0.0
    config = env.config
    substep_dt = config.dt / config.substeps
    drag = float(config.player_drag)
    decay = np.exp(-drag * substep_dt)
    substeps = steps * config.substeps
    return float(substep_dt * sum(decay**index for index in range(1, substeps + 1)))


def _inside_history_bounds(
    env: BlocketLeagueEnv,
    position: np.ndarray,
    velocity: np.ndarray,
    radius: float,
    history_frames: int,
) -> bool:
    low = env.config.wall + radius + 0.01
    high = 1.0 - env.config.wall - radius - 0.01
    for steps_back in range(history_frames):
        previous = position - velocity * env.config.dt * steps_back
        if np.any(previous < low) or np.any(previous > high):
            return False
    return True


def _set_state(
    env: BlocketLeagueEnv,
    *,
    player_position: np.ndarray,
    player_velocity: np.ndarray,
    puck_position: np.ndarray,
    puck_velocity: np.ndarray,
    score: int,
) -> None:
    env.state.player_position = player_position.astype(np.float32).copy()
    env.state.player_velocity = player_velocity.astype(np.float32).copy()
    env.state.puck_position = puck_position.astype(np.float32).copy()
    env.state.puck_velocity = puck_velocity.astype(np.float32).copy()
    env.state.score = score
    env.state.reset_timer = 0
    env.state.last_event = "coast"


def _event_steps(
    env: BlocketLeagueEnv,
    *,
    player_position: np.ndarray,
    player_velocity: np.ndarray,
    puck_position: np.ndarray,
    puck_velocity: np.ndarray,
    score: int,
    steps: int,
) -> list[str]:
    _set_state(
        env,
        player_position=player_position,
        player_velocity=player_velocity,
        puck_position=puck_position,
        puck_velocity=puck_velocity,
        score=score,
    )
    events = []
    for _ in range(steps):
        env.step(0)
        events.append(env.state.last_event)
    return events


def _anticipation_pair(
    seed: int,
    history_frames: int,
    image_size: int,
    horizon: int,
) -> tuple[EventExample, EventExample]:
    """Build a matched collision / timed-miss pair.

    Both examples have identical positions at the final observed frame and identical
    per-object speed magnitudes. The positive pair collides for the first time exactly
    ``horizon`` frames later. The negative pair reaches closest approach at the same
    effective time but misses outside the sum of the disc radii.
    """

    rng = np.random.default_rng(seed)
    config = _passive_config(image_size)
    env = BlocketLeagueEnv(seed=seed, config=config)
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    normal = np.asarray((np.cos(angle), np.sin(angle)), dtype=np.float32)
    tangent = np.asarray((-normal[1], normal[0]), dtype=np.float32)
    closing_speed = float(rng.uniform(0.46, 0.72))
    minimum = config.player_radius + config.puck_radius
    travel_before = _effective_travel_time(env, horizon - 1)
    travel_through = _effective_travel_time(env, horizon)
    contact_time = 0.5 * (travel_before + travel_through)
    separation = minimum + closing_speed * contact_time

    # The miss is tangent to a circle just outside the contact boundary and reaches
    # closest approach at the same drag-adjusted time as the positive collision.
    miss_radius = minimum * 1.06
    cosine = float(np.sqrt(max(1.0 - (miss_radius / separation) ** 2, 1e-6)))
    sine = miss_radius / separation
    miss_relative_speed = separation * cosine / max(contact_time, 1e-6)
    miss_relative_velocity = miss_relative_speed * (-cosine * normal + sine * tangent)
    impact_relative_velocity = -closing_speed * normal

    # Add orthogonal common-mode motion so both examples have the same speed for
    # each object. This removes speed magnitude as a shortcut while leaving the
    # relative trajectories different.
    target_object_speed = max(closing_speed, miss_relative_speed) * 0.52 + 0.035
    if target_object_speed >= config.max_player_speed * 0.92:
        raise RuntimeError("Matched-speed construction exceeds the player speed budget")
    impact_common_speed = float(
        np.sqrt(max(target_object_speed**2 - (closing_speed * 0.5) ** 2, 0.0))
    )
    miss_common_speed = float(
        np.sqrt(max(target_object_speed**2 - (miss_relative_speed * 0.5) ** 2, 0.0))
    )
    common_sign = float(rng.choice((-1.0, 1.0)))
    impact_common = tangent * impact_common_speed * common_sign
    miss_direction = miss_relative_velocity / max(float(np.linalg.norm(miss_relative_velocity)), 1e-8)
    miss_perpendicular = np.asarray((-miss_direction[1], miss_direction[0]), dtype=np.float32)
    miss_common = miss_perpendicular * miss_common_speed * common_sign

    positive_player_velocity = impact_common - impact_relative_velocity * 0.5
    positive_puck_velocity = impact_common + impact_relative_velocity * 0.5
    negative_player_velocity = miss_common - miss_relative_velocity * 0.5
    negative_puck_velocity = miss_common + miss_relative_velocity * 0.5

    midpoint = rng.uniform((0.44, 0.44), (0.56, 0.56)).astype(np.float32)
    player_position = midpoint - normal * separation * 0.5
    puck_position = midpoint + normal * separation * 0.5
    score = int(rng.integers(0, 5))

    for position, velocity, radius in (
        (player_position, positive_player_velocity, config.player_radius),
        (puck_position, positive_puck_velocity, config.puck_radius),
        (player_position, negative_player_velocity, config.player_radius),
        (puck_position, negative_puck_velocity, config.puck_radius),
    ):
        if not _inside_history_bounds(env, position, velocity, radius, history_frames):
            raise RuntimeError("Rendered history leaves the arena")

    positive_events = _event_steps(
        env,
        player_position=player_position,
        player_velocity=positive_player_velocity,
        puck_position=puck_position,
        puck_velocity=positive_puck_velocity,
        score=score,
        steps=horizon,
    )
    if positive_events[-1] != "impact" or "impact" in positive_events[:-1]:
        raise RuntimeError(f"Positive event timing failed: {positive_events}")
    if any(event in {"wall", "goal"} for event in positive_events):
        raise RuntimeError(f"Positive example hit another boundary: {positive_events}")

    negative_events = _event_steps(
        env,
        player_position=player_position,
        player_velocity=negative_player_velocity,
        puck_position=puck_position,
        puck_velocity=negative_puck_velocity,
        score=score,
        steps=horizon,
    )
    if any(event in {"impact", "wall", "goal"} for event in negative_events):
        raise RuntimeError(f"Negative example contains an event: {negative_events}")

    def render(player_velocity: np.ndarray, puck_velocity: np.ndarray, label: int) -> EventExample:
        frames = _render_history(
            env,
            player_position=player_position,
            player_velocity=player_velocity,
            puck_position=puck_position,
            puck_velocity=puck_velocity,
            history_frames=history_frames,
            score=score,
        )
        return EventExample(frames=frames, label=label)

    return (
        render(positive_player_velocity, positive_puck_velocity, 1),
        render(negative_player_velocity, negative_puck_velocity, 0),
    )


def _make_anticipation_examples(
    *,
    pairs: int,
    seed: int,
    history_frames: int,
    image_size: int,
    horizon: int,
) -> list[EventExample]:
    examples: list[EventExample] = []
    attempts = 0
    while len(examples) < pairs * 2:
        pair_seed = seed + attempts * 9_973
        attempts += 1
        try:
            positive, negative = _anticipation_pair(
                pair_seed, history_frames, image_size, horizon,
            )
        except RuntimeError:
            if attempts > pairs * 100:
                raise
            continue
        examples.extend((positive, negative))
    return examples


def run_collision_anticipation_probes(
    checkpoint_path: Path,
    output_path: Path,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    fit_pairs: int = 1024,
    test_pairs: int = 512,
    batch_size: int = 64,
    seed: int = 31,
    device_name: str = "auto",
    shuffle_repeats: int = 8,
) -> dict[str, Any]:
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else
        "cpu" if device_name == "auto" else device_name
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_pixel_direct_from_checkpoint(checkpoint).to(device).eval().requires_grad_(False)
    history_frames = model.config.history_frames
    results: dict[str, Any] = {}

    for horizon_index, horizon in enumerate(horizons):
        fit_examples = _make_anticipation_examples(
            pairs=fit_pairs,
            seed=seed + horizon_index * 10_000_019,
            history_frames=history_frames,
            image_size=model.config.image_size,
            horizon=horizon,
        )
        test_examples = _make_anticipation_examples(
            pairs=test_pairs,
            seed=seed + 100_000_007 + horizon_index * 10_000_019,
            history_frames=history_frames,
            image_size=model.config.image_size,
            horizon=horizon,
        )
        fit_features, fit_trajectory, fit_labels = _extract_dataset_features(
            model, fit_examples, device, batch_size,
        )
        test_features, test_trajectory, test_labels = _extract_dataset_features(
            model, test_examples, device, batch_size,
        )
        generator = torch.Generator().manual_seed(seed + horizon_index)
        shuffled_label_sets = [
            fit_labels[torch.randperm(len(fit_labels), generator=generator)]
            for _ in range(shuffle_repeats)
        ]

        trajectory_fit = _fit_ridge_classifier(fit_trajectory, fit_labels)
        trajectory_scores = _predict(trajectory_fit, test_trajectory)
        trajectory_shuffle_aucs = []
        for shuffled_labels in shuffled_label_sets:
            shuffled_fit = _fit_ridge_classifier(fit_trajectory, shuffled_labels)
            trajectory_shuffle_aucs.append(
                _roc_auc(_predict(shuffled_fit, test_trajectory), test_labels)
            )

        layers = []
        for depth, (fit_x, test_x) in enumerate(zip(fit_features, test_features, strict=True)):
            fit = _fit_ridge_classifier(fit_x, fit_labels)
            scores = _predict(fit, test_x)
            shuffled_aucs = []
            for shuffled_labels in shuffled_label_sets:
                shuffled_fit = _fit_ridge_classifier(fit_x, shuffled_labels)
                shuffled_aucs.append(_roc_auc(_predict(shuffled_fit, test_x), test_labels))
            layers.append({
                "stage": "patch embedding" if depth == 0 else f"block {depth}",
                "depth": depth,
                **_metrics(scores, test_labels),
                "shuffled_labels": _shuffle_summary(shuffled_aucs),
            })

        results[str(horizon)] = {
            "horizon_frames": horizon,
            "fit_pairs": fit_pairs,
            "test_pairs": test_pairs,
            "positive_definition": f"first disc impact occurs exactly {horizon} frames after the observation",
            "negative_definition": "matched positions and per-object speeds; closest approach occurs at the same effective time but misses",
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
        "horizons": list(horizons),
        "probe_features": "last-frame spatial mean + pixel-located player token + pixel-located puck token",
        "protocol": (
            "At every horizon, positive and negative histories end in identical disc positions "
            "with matched per-object speed magnitudes. Positive motion produces its first impact "
            "at the requested future frame; negative motion produces a drag-adjusted timed miss. "
            "Linear ridge probes are trained on frozen activations and disjoint test seeds."
        ),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    return payload


def run_random_weight_control(
    checkpoint_path: Path,
    output_path: Path,
    *,
    horizons: tuple[int, ...] = (1, 4, 8),
    fit_pairs: int = 1024,
    test_pairs: int = 512,
    batch_size: int = 64,
    seed: int = 31,
    random_seeds: tuple[int, ...] = (101, 211, 307),
    device_name: str = "auto",
) -> dict[str, Any]:
    """Repeat the probe on untrained transformers with the same architecture."""

    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else
        "cpu" if device_name == "auto" else device_name
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = PixelDirectConfig(**checkpoint["model_config"])
    results: dict[str, Any] = {}

    for horizon_index, horizon in enumerate(horizons):
        fit_examples = _make_anticipation_examples(
            pairs=fit_pairs,
            seed=seed + horizon_index * 10_000_019,
            history_frames=config.history_frames,
            image_size=config.image_size,
            horizon=horizon,
        )
        test_examples = _make_anticipation_examples(
            pairs=test_pairs,
            seed=seed + 100_000_007 + horizon_index * 10_000_019,
            history_frames=config.history_frames,
            image_size=config.image_size,
            horizon=horizon,
        )
        seed_layers: list[list[dict[str, float]]] = []
        for random_seed in random_seeds:
            torch.manual_seed(random_seed)
            model = DirectPixelTransformer(config).to(device).eval().requires_grad_(False)
            fit_features, _, fit_labels = _extract_dataset_features(
                model, fit_examples, device, batch_size,
            )
            test_features, _, test_labels = _extract_dataset_features(
                model, test_examples, device, batch_size,
            )
            layers = []
            for depth, (fit_x, test_x) in enumerate(zip(fit_features, test_features, strict=True)):
                scores = _predict(_fit_ridge_classifier(fit_x, fit_labels), test_x)
                layers.append(_metrics(scores, test_labels))
            seed_layers.append(layers)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        layers = []
        for depth in range(config.depth + 1):
            aucs = torch.tensor([values[depth]["roc_auc"] for values in seed_layers])
            pair_accuracies = torch.tensor([
                values[depth]["matched_pair_accuracy"] for values in seed_layers
            ])
            layers.append({
                "stage": "patch embedding" if depth == 0 else f"block {depth}",
                "depth": depth,
                "roc_auc_mean": float(aucs.mean()),
                "roc_auc_std": float(aucs.std(unbiased=False)),
                "matched_pair_accuracy_mean": float(pair_accuracies.mean()),
            })
        results[str(horizon)] = {
            "horizon_frames": horizon,
            "fit_pairs": fit_pairs,
            "test_pairs": test_pairs,
            "random_initializations": list(random_seeds),
            "layers": layers,
        }

    payload = {
        "version": 1,
        "control": "untrained transformers with the checkpoint architecture",
        "device": str(device),
        "horizons": list(horizons),
        "random_initializations": list(random_seeds),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe frozen activations for future disc impacts")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--horizons", default=",".join(map(str, DEFAULT_HORIZONS)))
    parser.add_argument("--fit-pairs", type=int, default=1024)
    parser.add_argument("--test-pairs", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    horizons = tuple(int(value) for value in args.horizons.split(",") if value)
    result = run_collision_anticipation_probes(
        args.checkpoint,
        args.output,
        horizons=horizons,
        fit_pairs=args.fit_pairs,
        test_pairs=args.test_pairs,
        batch_size=args.batch_size,
        seed=args.seed,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
