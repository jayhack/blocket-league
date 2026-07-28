import ringProbe from "@/public/blocket-league/interpretability/ring-probe.json";
import collisionAnticipation from "@/public/blocket-league/interpretability/collision-anticipation-probes.json";
import collisionRandomControl from "@/public/blocket-league/interpretability/collision-anticipation-random-control.json";

import styles from "./blocket-league-lab.module.css";

function directionColor(angle: number) {
  return `hsl(${angle} 72% 62%)`;
}

const eventPlot = {
  left: 82,
  right: 872,
  top: 52,
  bottom: 282,
};

function eventX(depth: number) {
  return eventPlot.left + (depth / 6) * (eventPlot.right - eventPlot.left);
}

function eventY(auc: number) {
  return eventPlot.bottom - ((auc - 0.5) / 0.5) * (eventPlot.bottom - eventPlot.top);
}

function eventPath(layers: Array<{ depth: number; roc_auc: number }>) {
  return layers
    .map((layer, index) => `${index ? "L" : "M"}${eventX(layer.depth)},${eventY(layer.roc_auc)}`)
    .join(" ");
}

const collisionSeries = [
  { horizon: 1, color: "#f4f4f0", result: collisionAnticipation.results["1"] },
  { horizon: 2, color: "#8bd5f2", result: collisionAnticipation.results["2"] },
  { horizon: 4, color: "#2196d2", result: collisionAnticipation.results["4"] },
  { horizon: 6, color: "#f4a261", result: collisionAnticipation.results["6"] },
  { horizon: 8, color: "#e85d2a", result: collisionAnticipation.results["8"] },
] as const;

export function CollisionAnticipationViewer() {
  const eightFrameResult = collisionAnticipation.results["8"];
  const eightFrameRandom = collisionRandomControl.results["8"].layers.find(
    (layer) => layer.depth === 5,
  );

  return (
    <div className={styles.emergenceStory}>
      <figure className={styles.eventFigure}>
        <header className={styles.eventFigureHeader}>
          <p>Held-out AUROC · 1,024 matched fit pairs and 512 disjoint test pairs per horizon</p>
        </header>

        <div className={styles.eventChart}>
          <div className={styles.eventLegend} aria-hidden="true">
            {collisionSeries.map((series) => (
              <span key={series.horizon}>
                <i style={{ background: series.color }} />
                {series.horizon} {series.horizon === 1 ? "frame" : "frames"} ahead
              </span>
            ))}
          </div>
          <svg viewBox="0 0 960 350" role="img" aria-labelledby="event-title event-description">
            <title id="event-title">Future collision decodability across transformer depth and anticipation horizon</title>
            <desc id="event-description">
              Linear probes become better at predicting future disc collisions through progressively
              deeper blocks. Block five remains strongly predictive even eight frames before contact.
            </desc>

            {[0.5, 0.75, 1].map((tick) => (
              <g key={tick} className={styles.eventGridline}>
                <line x1={eventPlot.left} x2={eventPlot.right} y1={eventY(tick)} y2={eventY(tick)} />
                <text x={eventPlot.left - 19} y={eventY(tick) + 6} textAnchor="end">{tick.toFixed(2)}</text>
              </g>
            ))}
            <text
              className={styles.eventYLabel}
              x="22"
              y={(eventPlot.top + eventPlot.bottom) / 2}
              textAnchor="middle"
              transform={`rotate(-90 22 ${(eventPlot.top + eventPlot.bottom) / 2})`}
            >
              Held-out AUROC
            </text>
            <line
              className={styles.eventChanceLine}
              x1={eventPlot.left}
              x2={eventPlot.right}
              y1={eventY(0.5)}
              y2={eventY(0.5)}
            />

            {collisionSeries.map((series) => (
              <g key={series.horizon}>
                <path
                  className={styles.eventHorizonLine}
                  d={eventPath(series.result.layers)}
                  style={{ stroke: series.color }}
                />
                {series.result.layers.map((layer) => (
                  <circle
                    key={layer.depth}
                    className={styles.eventPoint}
                    cx={eventX(layer.depth)}
                    cy={eventY(layer.roc_auc)}
                    r="5"
                    style={{ stroke: series.color }}
                  />
                ))}
              </g>
            ))}

            {eightFrameResult.layers.map((layer) => (
              <text key={layer.depth} className={styles.eventAxisLabel} x={eventX(layer.depth)} y="326" textAnchor="middle">
                {layer.depth === 0 ? "Embedding" : `Block ${layer.depth}`}
              </text>
            ))}
          </svg>
        </div>

        <figcaption className={styles.eventCaption}>
          <p>
            <strong>At block 5, collisions remain predictable eight frames, or 400 ms, ahead.</strong>{" "}
            Held-out AUROC reaches {eightFrameResult.layers[5].roc_auc.toFixed(3)} at that horizon,
            compared with {eightFrameResult.raw_trajectory_linear_baseline.roc_auc.toFixed(3)} for a
            linear readout of the explicit pixel trajectory.
          </p>
          <p>
            An untrained transformer with the same architecture reaches only{" "}
            {eightFrameRandom?.roc_auc_mean.toFixed(3) ?? "0.567"} at block 5, averaged across three
            random initializations. Shuffled-label controls remain near chance. Training therefore
            organizes a substantially more predictive collision representation than either raw
            linear pixels or random transformer features.
          </p>
        </figcaption>
      </figure>
    </div>
  );
}

export function MotionRingViewer() {
  const blockFive = ringProbe.mlp_direction_tuning.find((row) => row.block === 5);
  if (!blockFive) return null;

  const coordinates = blockFive.angle_bin_population_geometry.coordinates;
  const pcaCenter = { x: 480, y: 292 };
  const pcaScale = 222;
  const pcaPoints = coordinates
    .map((point) => `${pcaCenter.x + point.x * pcaScale},${pcaCenter.y - point.y * pcaScale}`)
    .join(" ");

  return (
    <div className={styles.emergenceStory}>

      <figure className={styles.pcaFigure}>
        <header className={styles.pcaFigureHeader}>
          <p>
            24 observed motion-direction bins projected from 768 activation dimensions to two principal components.
          </p>
        </header>

        <div className={styles.pcaChart}>
          <svg viewBox="0 0 960 610" role="img" aria-labelledby="pca-title pca-description">
            <title id="pca-title">Principal component projection of direction-conditioned hidden activations</title>
            <desc id="pca-description">
              Twenty-four mean block-five MLP activation vectors projected from 768 dimensions
              onto their first two principal components form a single closed ring. Point color
              represents observed physical motion direction.
            </desc>

            <line className={styles.pcaAxis} x1="130" x2="830" y1={pcaCenter.y} y2={pcaCenter.y} />
            <line className={styles.pcaAxis} x1={pcaCenter.x} x2={pcaCenter.x} y1="42" y2="542" />
            <text className={styles.pcaAxisLabel} x="820" y={pcaCenter.y - 16} textAnchor="end">PC1</text>
            <text className={styles.pcaAxisLabel} x={pcaCenter.x + 18} y="62">PC2</text>

            <polygon className={styles.pcaRingLine} points={pcaPoints} />

            {coordinates.map((point) => {
              const labelPoint = [7.5, 97.5, 187.5, 277.5].some(
                (angle) => Math.abs(point.angle_degrees - angle) < 0.2,
              );
              const x = pcaCenter.x + point.x * pcaScale;
              const y = pcaCenter.y - point.y * pcaScale;
              return (
                <g key={point.angle_degrees}>
                  <circle
                    className={styles.pcaPoint}
                    cx={x}
                    cy={y}
                    r="9"
                    style={{ fill: directionColor(point.angle_degrees) }}
                  />
                  {labelPoint ? (
                    <text
                      className={styles.pcaDirectionLabel}
                      x={x + (point.x >= 0 ? 16 : -16)}
                      y={y + (point.y >= 0 ? -13 : 24)}
                      textAnchor={point.x >= 0 ? "start" : "end"}
                    >
                      {Math.round(point.angle_degrees)}°
                    </text>
                  ) : null}
                </g>
              );
            })}

            <g transform="translate(270 574)">
              {[0, 45, 90, 135, 180, 225, 270, 315].map((angle, index) => (
                <rect key={angle} x={index * 52} y="0" width="52" height="8" style={{ fill: directionColor(angle) }} />
              ))}
              <text className={styles.pcaLegendLabel} x="-18" y="10" textAnchor="end">0°</text>
              <text className={styles.pcaLegendLabel} x="434" y="10">360°</text>
              <text className={styles.pcaLegendTitle} x="208" y="32" textAnchor="middle">observed motion direction</text>
            </g>
          </svg>
        </div>

      </figure>
    </div>
  );
}
