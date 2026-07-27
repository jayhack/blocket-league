from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bin_by_angle(summary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(row["center_degrees"]): row
        for row in summary["direction_evaluation"]["bins"]
    }


def build_comparison(control: dict[str, Any], holdout: dict[str, Any]) -> dict[str, Any]:
    control_bins = _bin_by_angle(control)
    holdout_bins = _bin_by_angle(holdout)
    metric = "puck_position_error_px"
    rows = []
    for angle in sorted(control_bins):
        control_row = control_bins[angle]
        holdout_row = holdout_bins[angle]
        rows.append({
            "center_degrees": angle,
            "held_out": bool(holdout_row["held_out"]),
            "control_puck_error_px": control_row[metric],
            "holdout_puck_error_px": holdout_row[metric],
            "error_ratio": holdout_row[metric] / max(control_row[metric], 1e-9),
        })

    held_out_rows = [row for row in rows if row["held_out"]]
    seen_rows = [row for row in rows if not row["held_out"]]
    held_out_error = sum(row["holdout_puck_error_px"] for row in held_out_rows) / len(held_out_rows)
    seen_error = sum(row["holdout_puck_error_px"] for row in seen_rows) / len(seen_rows)
    control_held_out_error = (
        sum(row["control_puck_error_px"] for row in held_out_rows) / len(held_out_rows)
    )
    return {
        "experiment": "puck-direction-support-holdout",
        "training": {
            "control_steps": control["config"]["steps"],
            "holdout_steps": holdout["config"]["steps"],
            "cache_samples_per_model": holdout["config"]["cache_samples"],
            "cache_frames": holdout["config"]["cache_frames"],
            "holdout_center_degrees": holdout["config"]["excluded_puck_angle_center_degrees"],
            "holdout_width_degrees": holdout["config"]["excluded_puck_angle_width_degrees"],
            "seed": holdout["config"]["seed"],
        },
        "evaluation": {
            "samples_per_bin": holdout["direction_evaluation"]["samples_per_bin"],
            "rollout_frames": holdout["direction_evaluation"]["frames"],
            "angle_sampling_width_degrees": holdout["direction_evaluation"][
                "angle_sampling_width_degrees"
            ],
            "metric": metric,
        },
        "summary": {
            "heldout_model_heldout_error_px": held_out_error,
            "heldout_model_seen_error_px": seen_error,
            "matched_control_heldout_error_px": control_held_out_error,
            "heldout_to_seen_ratio": held_out_error / max(seen_error, 1e-9),
            "heldout_to_control_ratio": held_out_error / max(control_held_out_error, 1e-9),
        },
        "bins": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("control_summary", type=Path)
    parser.add_argument("holdout_summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    comparison = build_comparison(_load(args.control_summary), _load(args.holdout_summary))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
