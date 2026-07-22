import ringProbe from "@/public/blocket-league/interpretability/ring-probe.json";

import styles from "./blocket-league-lab.module.css";


const plot = {
  left: 34,
  right: 310,
  top: 24,
  bottom: 178,
};

function depthX(depth: number) {
  return plot.left + (depth / 6) * (plot.right - plot.left);
}

function scoreY(score: number) {
  return plot.bottom - Math.max(0, Math.min(1, score)) * (plot.bottom - plot.top);
}

function pathFor(key: "direction_r2" | "speed_r2") {
  return ringProbe.layer_probes
    .map((row, index) => `${index ? "L" : "M"}${depthX(row.depth)},${scoreY(row[key])}`)
    .join(" ");
}

export function PhysicsEmergenceViewer() {
  const blockFive = ringProbe.mlp_direction_tuning.find((row) => row.block === 5);
  if (!blockFive) return null;
  const coordinates = blockFive.angle_bin_population_geometry.coordinates;
  const ringPoints = coordinates
    .map((point) => `${120 + point.x * 72},${105 - point.y * 72}`)
    .join(" ");
  const causal = ringProbe.causal_direction_circle;
  const maxDisplacement = Math.max(...causal.angles.map((row) => row.mean_displacement_pixels));

  return (
    <figure className={styles.emergenceFigure}>
      <div className={styles.emergencePanels}>
        <div className={styles.emergencePanel}>
          <div className={styles.emergencePanelHeader}>
            <span>01 / ACCESS</span>
            <strong>Direction becomes readable</strong>
            <small>held-out linear probes · rendered-pixel labels</small>
          </div>
          <svg viewBox="0 0 340 220" role="img" aria-label="Direction and speed probe R squared by transformer depth. Direction rises from zero at the patch embedding to 0.63 at block one and peaks at 0.88 at block five.">
            <rect className={styles.emergenceZone} x={depthX(3.65)} y="12" width={depthX(5.35) - depthX(3.65)} height="178" />
            {[0, 0.5, 1].map((tick) => (
              <g key={tick}>
                <line className={styles.emergenceGrid} x1={plot.left} x2={plot.right} y1={scoreY(tick)} y2={scoreY(tick)} />
                <text className={styles.emergenceAxisText} x="4" y={scoreY(tick) + 3}>{tick.toFixed(1)}</text>
              </g>
            ))}
            <path className={styles.emergenceDirectionLine} d={pathFor("direction_r2")} />
            <path className={styles.emergenceSpeedLine} d={pathFor("speed_r2")} />
            {ringProbe.layer_probes.map((row) => (
              <g key={row.depth}>
                <circle className={styles.emergenceDirectionPoint} cx={depthX(row.depth)} cy={scoreY(row.direction_r2)} r="3" />
                <circle className={styles.emergenceSpeedPoint} cx={depthX(row.depth)} cy={scoreY(row.speed_r2)} r="2.5" />
                <text className={styles.emergenceAxisText} x={depthX(row.depth)} y="205" textAnchor="middle">
                  {row.depth === 0 ? "embed" : `B${row.depth}`}
                </text>
              </g>
            ))}
            <text className={styles.emergenceZoneLabel} x={depthX(4.5)} y="22" textAnchor="middle">RING ORGANIZES</text>
            <g className={styles.emergenceLegend}>
              <line className={styles.emergenceDirectionLine} x1="44" x2="60" y1="16" y2="16" />
              <text x="65" y="19">direction</text>
              <line className={styles.emergenceSpeedLine} x1="128" x2="144" y1="16" y2="16" />
              <text x="149" y="19">speed</text>
            </g>
          </svg>
        </div>

        <div className={styles.emergencePanel}>
          <div className={styles.emergencePanelHeader}>
            <span>02 / ORGANIZE</span>
            <strong>The population closes into a ring</strong>
            <small>block 5 MLP · means across 24 direction bins</small>
          </div>
          <svg viewBox="0 0 240 220" role="img" aria-label="The block five MLP population traces a closed loop as the observed motion direction rotates through 360 degrees.">
            <circle className={styles.populationGuide} cx="120" cy="105" r="72" />
            <polygon className={styles.populationRing} points={ringPoints} />
            {coordinates.map((point) => (
              <circle
                className={styles.populationPoint}
                key={point.angle_degrees}
                cx={120 + point.x * 72}
                cy={105 - point.y * 72}
                r="3"
              />
            ))}
            <text className={styles.populationValue} x="120" y="101" textAnchor="middle">305 / 768</text>
            <text className={styles.populationCaption} x="120" y="117" textAnchor="middle">direction-tuned units</text>
            <text className={styles.populationMetric} x="120" y="206" textAnchor="middle">circular distance r = 0.976 · winding = 1</text>
          </svg>
        </div>

        <div className={styles.emergencePanel}>
          <div className={styles.emergencePanelHeader}>
            <span>03 / CONTROL</span>
            <strong>Two writes chart the whole circle</strong>
            <small>256 unseen 12-frame hallucinations</small>
          </div>
          <svg viewBox="0 0 240 220" role="img" aria-label="Eight interpolated activation writes produce generated displacement directions with 5.1 degrees of mean angular error.">
            <circle className={styles.populationGuide} cx="120" cy="105" r="72" />
            <line className={styles.compassAxis} x1="38" x2="202" y1="105" y2="105" />
            <line className={styles.compassAxis} x1="120" x2="120" y1="23" y2="187" />
            {causal.angles.map((row) => {
              const target = row.target_degrees * Math.PI / 180;
              const observed = row.mean_direction_degrees * Math.PI / 180;
              const radius = 72 * row.mean_displacement_pixels / maxDisplacement;
              return (
                <g key={row.target_degrees}>
                  <line
                    className={styles.compassTarget}
                    x1="120"
                    y1="105"
                    x2={120 + Math.cos(target) * 72}
                    y2={105 + Math.sin(target) * 72}
                  />
                  <circle
                    className={styles.compassObserved}
                    cx={120 + Math.cos(observed) * radius}
                    cy={105 + Math.sin(observed) * radius}
                    r="4"
                  />
                </g>
              );
            })}
            <text className={styles.populationValue} x="120" y="101" textAnchor="middle">5.1°</text>
            <text className={styles.populationCaption} x="120" y="117" textAnchor="middle">mean steering error</text>
            <text className={styles.populationMetric} x="120" y="206" textAnchor="middle">v(θ) = cos θ · vₓ + sin θ · vᵧ</text>
          </svg>
        </div>
      </div>
      <figcaption className={styles.emergenceCaption}>
        <span><strong>Readable is not organized.</strong> The first temporal-attention block exposes direction, but sinusoidally tuned MLP units only form a clean circular population at blocks 4–5.</span>
        <span><strong>Distributed is not uncontrollable.</strong> Direction occupies 74 residual dimensions, yet downstream averaging supplies a two-vector causal chart over the code.</span>
      </figcaption>
    </figure>
  );
}
