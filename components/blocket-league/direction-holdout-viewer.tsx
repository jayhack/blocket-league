import directionHoldout from "@/public/blocket-league/interpretability/direction-holdout.json";

import styles from "./blocket-league-lab.module.css";

const plot = {
  left: 76,
  right: 922,
  top: 58,
  bottom: 302,
};

const maxError = Math.ceil(
  Math.max(
    ...directionHoldout.bins.flatMap((row) => [
      row.control_puck_error_px,
      row.holdout_puck_error_px,
    ]),
  ) * 2,
) / 2;

function xFor(index: number) {
  return plot.left + (index / (directionHoldout.bins.length - 1)) * (plot.right - plot.left);
}

function yFor(error: number) {
  return plot.bottom - (error / maxError) * (plot.bottom - plot.top);
}

function pathFor(key: "control_puck_error_px" | "holdout_puck_error_px") {
  return directionHoldout.bins
    .map((row, index) => `${index === 0 ? "M" : "L"}${xFor(index)},${yFor(row[key])}`)
    .join(" ");
}

export function DirectionHoldoutViewer() {
  const summary = directionHoldout.summary;
  const penalty = (summary.heldout_to_control_ratio - 1) * 100;
  const seenDifference = (summary.heldout_to_seen_ratio - 1) * 100;

  return (
    <figure className={styles.directionHoldoutFigure}>
      <header className={styles.directionHoldoutHeader}>
        <p>
          Mean rendered-pixel puck error over 12 autoregressive frames · 32 unseen
          worlds per direction bin
        </p>
      </header>

      <div className={styles.directionHoldoutChart}>
        <div className={styles.directionHoldoutLegend} aria-hidden="true">
          <span data-series="control"><i /> All-angle control</span>
          <span data-series="holdout"><i /> 60° wedge held out</span>
        </div>
        <svg viewBox="0 0 960 366" role="img" aria-labelledby="direction-title direction-description">
          <title id="direction-title">Puck prediction error by initial motion direction</title>
          <desc id="direction-description">
            The direction-held-out model has similar error on due-east motion and the
            directions it saw, but is less accurate than the matched all-angle control
            on due-east motion.
          </desc>

          <rect
            className={styles.directionHeldoutBand}
            x={xFor(0) - 34}
            y={plot.top - 18}
            width="68"
            height={plot.bottom - plot.top + 36}
          />
          <text className={styles.directionHeldoutLabel} x={xFor(0)} y={plot.top - 27} textAnchor="middle">
            NEVER SEEN
          </text>

          {[0, maxError / 2, maxError].map((tick) => (
            <g key={tick} className={styles.directionGridline}>
              <line x1={plot.left} x2={plot.right} y1={yFor(tick)} y2={yFor(tick)} />
              <text x={plot.left - 18} y={yFor(tick) + 5} textAnchor="end">
                {tick.toFixed(1)}
              </text>
            </g>
          ))}

          <path className={styles.directionControlLine} d={pathFor("control_puck_error_px")} />
          <path className={styles.directionHoldoutLine} d={pathFor("holdout_puck_error_px")} />

          {directionHoldout.bins.map((row, index) => (
            <g key={row.center_degrees}>
              <circle
                className={styles.directionControlPoint}
                cx={xFor(index)}
                cy={yFor(row.control_puck_error_px)}
                r="5"
              />
              <circle
                className={styles.directionHoldoutPoint}
                cx={xFor(index)}
                cy={yFor(row.holdout_puck_error_px)}
                r="5"
              />
              <text className={styles.directionAxisLabel} x={xFor(index)} y="338" textAnchor="middle">
                {row.center_degrees}°
              </text>
            </g>
          ))}

          <text
            className={styles.directionYLabel}
            x="19"
            y={(plot.top + plot.bottom) / 2}
            textAnchor="middle"
            transform={`rotate(-90 19 ${(plot.top + plot.bottom) / 2})`}
          >
            puck position error (px)
          </text>
        </svg>
      </div>

      <figcaption className={styles.directionHoldoutCaption}>
        <div>
          <span>Unseen due east</span>
          <strong>{summary.heldout_model_heldout_error_px.toFixed(2)} px</strong>
          <small>{Math.abs(seenDifference).toFixed(1)}% lower than its seen-direction average</small>
        </div>
        <div>
          <span>Seen-direction average</span>
          <strong>{summary.heldout_model_seen_error_px.toFixed(2)} px</strong>
          <small>same held-out model</small>
        </div>
        <div>
          <span>Matched control · east</span>
          <strong>{summary.matched_control_heldout_error_px.toFixed(2)} px</strong>
          <small>{penalty.toFixed(0)}% holdout penalty</small>
        </div>
      </figcaption>
    </figure>
  );
}
