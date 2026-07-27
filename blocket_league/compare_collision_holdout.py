from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _evaluation(payload: dict[str, Any]) -> dict[str, Any]:
    if "collision_quadrant_evaluation" in payload:
        return payload["collision_quadrant_evaluation"]
    return payload["evaluation"]


def build_comparison(
    control: dict[str, Any],
    holdout: dict[str, Any],
) -> dict[str, Any]:
    control_eval = _evaluation(control)
    holdout_eval = _evaluation(holdout)
    control_rows = {
        str(row["quadrant"]): row for row in control_eval["quadrants"]
    }
    holdout_rows = {
        str(row["quadrant"]): row for row in holdout_eval["quadrants"]
    }
    metric = "puck_position_error_px"
    rows = []
    for quadrant in ("upper-left", "upper-right", "lower-left", "lower-right"):
        control_row = control_rows[quadrant]
        holdout_row = holdout_rows[quadrant]
        rows.append(
            {
                "quadrant": quadrant,
                "held_out": bool(holdout_row["held_out"]),
                "control_puck_error_px": control_row[metric],
                "holdout_puck_error_px": holdout_row[metric],
                "error_ratio": holdout_row[metric]
                / max(control_row[metric], 1e-9),
            }
        )

    held_out_row = next(row for row in rows if row["held_out"])
    seen_rows = [row for row in rows if not row["held_out"]]
    held_out_error = held_out_row["holdout_puck_error_px"]
    seen_error = sum(row["holdout_puck_error_px"] for row in seen_rows) / len(
        seen_rows
    )
    control_held_out_error = held_out_row["control_puck_error_px"]
    config = holdout["config"]
    return {
        "experiment": "collision-location-support-holdout",
        "training": {
            "control_steps": config["steps"],
            "holdout_steps": config["steps"],
            "cache_samples_per_model": config["cache_samples"],
            "cache_frames": config["cache_frames"],
            "held_out_quadrant": config["excluded_collision_quadrant"],
            "seed": config["seed"],
        },
        "evaluation": {
            "samples_per_quadrant": holdout_eval["samples_per_quadrant"],
            "rollout_frames": holdout_eval["frames"],
            "metric": metric,
        },
        "summary": {
            "heldout_model_heldout_error_px": held_out_error,
            "heldout_model_seen_error_px": seen_error,
            "matched_control_heldout_error_px": control_held_out_error,
            "heldout_to_seen_ratio": held_out_error / max(seen_error, 1e-9),
            "heldout_to_control_ratio": held_out_error
            / max(control_held_out_error, 1e-9),
        },
        "quadrants": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("control_evaluation", type=Path)
    parser.add_argument("holdout_summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    comparison = build_comparison(
        _load(args.control_evaluation),
        _load(args.holdout_summary),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
