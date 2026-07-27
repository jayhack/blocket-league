import collisionHoldout from "@/public/blocket-league/interpretability/collision-holdout.json";

import styles from "./blocket-league-lab.module.css";

const maxError =
  Math.ceil(
    Math.max(
      ...collisionHoldout.quadrants.flatMap((row) => [
        row.control_puck_error_px,
        row.holdout_puck_error_px,
      ]),
    ) * 4,
  ) / 4;

function labelFor(quadrant: string) {
  return quadrant.replace("-", " ");
}

export function CollisionHoldoutViewer() {
  return (
    <figure className={styles.collisionHoldoutFigure}>
      <header className={styles.collisionHoldoutHeader}>
        <p>
          Mean puck-position error over 12 autoregressive frames · 32 collision
          worlds per quadrant
        </p>
        <div className={styles.collisionHoldoutLegend} aria-hidden="true">
          <span data-series="control"><i /> All-location control</span>
          <span data-series="holdout"><i /> Upper-right held out</span>
        </div>
      </header>

      <div className={styles.collisionQuadrantGrid}>
        {collisionHoldout.quadrants.map((row) => (
          <div
            className={styles.collisionQuadrant}
            data-held-out={row.held_out || undefined}
            key={row.quadrant}
          >
            <div className={styles.collisionQuadrantTitle}>
              <strong>{labelFor(row.quadrant)}</strong>
              {row.held_out ? <span>NEVER SEEN</span> : null}
            </div>
            <div className={styles.collisionBars}>
              <div>
                <span
                  data-series="control"
                  style={{ width: `${(row.control_puck_error_px / maxError) * 100}%` }}
                />
                <small>{row.control_puck_error_px.toFixed(2)} px</small>
              </div>
              <div>
                <span
                  data-series="holdout"
                  style={{ width: `${(row.holdout_puck_error_px / maxError) * 100}%` }}
                />
                <small>{row.holdout_puck_error_px.toFixed(2)} px</small>
              </div>
            </div>
          </div>
        ))}
      </div>
    </figure>
  );
}
