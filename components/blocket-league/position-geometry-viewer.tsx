import collisionProbe from "@/public/blocket-league/interpretability/collision-anticipation-probes.json";
import positionGeometry from "@/public/blocket-league/interpretability/position-geometry.json";
import positionWrite from "@/public/blocket-league/interpretability/position-write.json";
import ringProbe from "@/public/blocket-league/interpretability/ring-probe.json";

import styles from "./blocket-league-lab.module.css";

const plot = { left: 72, right: 900, top: 54, bottom: 286 };

function x(depth: number) {
  return plot.left + (depth / 6) * (plot.right - plot.left);
}

function y(score: number) {
  return plot.bottom - Math.max(0, Math.min(1, score)) * (plot.bottom - plot.top);
}

function line(rows: Array<{ r2: number }>) {
  return rows.map((row, index) => `${index ? "L" : "M"}${x(index)},${y(row.r2)}`).join(" ");
}

function signedCurve(axis: "x_plus" | "x_minus" | "y_plus" | "y_minus") {
  const values =
    positionWrite.rollout.effects.puck_token_jacobian[axis].expectedAxisMeanDeltaByFrame;
  const sign = axis.endsWith("minus") ? -1 : 1;
  return values.map((value) => value * sign);
}

function meanCurve() {
  const axes = (["x_plus", "x_minus", "y_plus", "y_minus"] as const).map(signedCurve);
  return axes[0].map((_, index) =>
    axes.reduce((sum, values) => sum + values[index], 0) / axes.length
  );
}

const causalCurve = meanCurve();

function causalX(frame: number) {
  return plot.left + (frame / (causalCurve.length - 1)) * (plot.right - plot.left);
}

function causalY(value: number) {
  return plot.bottom - (value / 1.4) * (plot.bottom - plot.top);
}

export function PositionGeometryViewer() {
  const player = positionGeometry.trained.player.readouts.fixed_bottom_right;
  const puck = positionGeometry.trained.puck.readouts.fixed_bottom_right;
  const playerStandard = player.ordinaryHeldOutTrajectories;
  const puckStandard = puck.ordinaryHeldOutTrajectories;
  const playerQuadrant = player.upperRightNeverSeenByProbe;
  const puckQuadrant = puck.upperRightNeverSeenByProbe;
  const motion = ringProbe.layer_probes[5];
  const collision = collisionProbe.results["8"].layers[5];
  const causalPath = causalCurve
    .map((value, index) => `${index ? "L" : "M"}${causalX(index)},${causalY(value)}`)
    .join(" ");

  return (
    <div className={styles.positionGeometryStory}>
      <div className={styles.stateVariableSummary}>
        <div>
          <span>WHERE</span>
          <strong>0.98 R²</strong>
          <p>Player and puck x/y from one fixed token.</p>
        </div>
        <div>
          <span>HOW IT MOVES</span>
          <strong>{motion.direction_r2.toFixed(2)} R²</strong>
          <p>Velocity direction, with speed at {motion.speed_r2.toFixed(2)} R².</p>
        </div>
        <div>
          <span>WHAT HAPPENS NEXT</span>
          <strong>{collision.roc_auc.toFixed(2)} AUROC</strong>
          <p>Collision prediction eight frames before contact.</p>
        </div>
      </div>

      <figure className={styles.positionDepthFigure}>
        <header>
          <p>
            Linear x/y decoding from the literal bottom-right token after the final
            observed frame. No entity token is selected.
          </p>
        </header>
        <div className={styles.positionDepthLegend} aria-hidden="true">
          <span data-series="player"><i /> Player · ordinary</span>
          <span data-series="puck"><i /> Puck · ordinary</span>
          <span data-series="quadrant"><i /> Upper-right excluded from probe fit</span>
        </div>
        <svg viewBox="0 0 960 350" role="img" aria-labelledby="position-depth-title position-depth-desc">
          <title id="position-depth-title">Cartesian position decodability across transformer depth</title>
          <desc id="position-depth-desc">
            Position is absent from the fixed token at the input embedding, becomes
            readable after spatial attention, and peaks at block five. Decoders also
            transfer to an upper-right quadrant excluded during probe fitting.
          </desc>
          {[0, 0.5, 1].map((tick) => (
            <g className={styles.positionGridline} key={tick}>
              <line x1={plot.left} x2={plot.right} y1={y(tick)} y2={y(tick)} />
              <text x={plot.left - 17} y={y(tick) + 5} textAnchor="end">{tick.toFixed(1)}</text>
            </g>
          ))}
          <path className={styles.positionPlayerLine} d={line(playerStandard)} />
          <path className={styles.positionPuckLine} d={line(puckStandard)} />
          <path className={styles.positionQuadrantLine} d={line(playerQuadrant)} />
          <path className={styles.positionQuadrantLine} d={line(puckQuadrant)} />
          {playerStandard.map((row, index) => (
            <g key={row.stage}>
              <circle className={styles.positionPlayerPoint} cx={x(index)} cy={y(row.r2)} r="5" />
              <circle className={styles.positionPuckPoint} cx={x(index)} cy={y(puckStandard[index].r2)} r="4" />
              <text className={styles.positionAxisLabel} x={x(index)} y="326" textAnchor="middle">
                {index === 0 ? "Embedding" : `Block ${index}`}
              </text>
            </g>
          ))}
        </svg>
        <figcaption>
          <strong>One layer below the output, R² reaches 0.967 for the player and 0.980 for the puck.</strong>
          {" "}When the decoder has never seen its target in the upper-right quadrant, it
          still reaches 0.668 and 0.719. A matched random transformer fails this test.
        </figcaption>
      </figure>

      <div className={styles.positionWriteHeading}>
        <h3>Readable is not automatically writable.</h3>
        <p>
          Adding the linear probe&apos;s own x/y vector barely changes the rendered puck.
          A probe identifies correlation; it does not identify the downstream route that
          controls pixels. Reorient the write using the Jacobian of the rendered puck
          centroid, however, and all four directions become causal.
        </p>
      </div>

      <figure className={styles.positionCausalFigure}>
        <div className={styles.positionCausalStats}>
          <div>
            <span>LINEAR-PROBE WRITE</span>
            <strong>≈0 px</strong>
            <small>inconsistent immediate displacement</small>
          </div>
          <div>
            <span>JACOBIAN-ORIENTED WRITE</span>
            <strong>0.40 px</strong>
            <small>mean immediate intended displacement</small>
          </div>
          <div>
            <span>ONE WRITE, 12 FRAMES LATER</span>
            <strong>1.16 px</strong>
            <small>mean intended separation from baseline</small>
          </div>
        </div>
        <svg viewBox="0 0 960 350" role="img" aria-labelledby="position-causal-title position-causal-desc">
          <title id="position-causal-title">Effect of one puck-position Jacobian write over twelve generated frames</title>
          <desc id="position-causal-desc">
            Averaged across positive and negative horizontal and vertical interventions,
            the intended displacement begins around 0.4 pixels and grows beyond one pixel
            after the write has stopped.
          </desc>
          {[0, 0.5, 1, 1.4].map((tick) => (
            <g className={styles.positionGridline} key={tick}>
              <line x1={plot.left} x2={plot.right} y1={causalY(tick)} y2={causalY(tick)} />
              <text x={plot.left - 17} y={causalY(tick) + 5} textAnchor="end">{tick.toFixed(1)}</text>
            </g>
          ))}
          <path className={styles.positionCausalLine} d={causalPath} />
          {causalCurve.map((value, index) => (
            <circle
              className={styles.positionCausalPoint}
              cx={causalX(index)}
              cy={causalY(value)}
              key={index}
              r="5"
            />
          ))}
          {causalCurve.map((_, index) => (
            <text className={styles.positionAxisLabel} x={causalX(index)} y="326" textAnchor="middle" key={index}>
              {index + 1}
            </text>
          ))}
          <text className={styles.positionCausalWriteLabel} x={causalX(0) + 10} y={causalY(causalCurve[0]) - 15}>
            write once
          </text>
        </svg>
        <figcaption>
          Strength-8 block-5 write at the puck&apos;s spatial token; 128 unseen worlds.
          The puck remains present in every final frame. Immediate player displacement is
          only 0.01–0.05 px, so the edit is initially entity-specific; later differences
          include the altered world&apos;s downstream physics.
        </figcaption>
      </figure>
    </div>
  );
}
