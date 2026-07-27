"use client";

import { useEffect, useId, useMemo, useState } from "react";

import styles from "./experiment.module.css";

type LossPoint = {
  step: number;
  loss: number;
  lossEma50: number;
};

type LossArtifact = {
  points: LossPoint[];
};

const WIDTH = 960;
const HEIGHT = 360;
const MARGIN = { top: 20, right: 22, bottom: 42, left: 58 };
const Y_MIN = 0.02;
const Y_MAX = 5;
const Y_TICKS = [3, 1, 0.3, 0.1, 0.03];

function formatLoss(value: number) {
  return value < 0.1 ? value.toFixed(3) : value.toFixed(2);
}

function pathFor(
  points: LossPoint[],
  value: (point: LossPoint) => number,
  x: (step: number) => number,
  y: (loss: number) => number,
) {
  return points
    .map((point, index) => {
      const command = index === 0 ? "M" : "L";
      return `${command}${x(point.step).toFixed(2)},${y(value(point)).toFixed(2)}`;
    })
    .join(" ");
}

export function LossViewer({ lossUrl }: { lossUrl: string }) {
  const gradientId = useId().replaceAll(":", "");
  const [points, setPoints] = useState<LossPoint[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [loadError, setLoadError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch(lossUrl)
      .then((response) => {
        if (!response.ok) throw new Error("loss artifact unavailable");
        return response.json() as Promise<LossArtifact>;
      })
      .then((artifact) => {
        if (!cancelled) setPoints(artifact.points);
      })
      .catch(() => {
        if (!cancelled) setLoadError(true);
      });
    return () => {
      cancelled = true;
    };
  }, [lossUrl]);

  const plotWidth = WIDTH - MARGIN.left - MARGIN.right;
  const plotHeight = HEIGHT - MARGIN.top - MARGIN.bottom;
  const maxStep = points.at(-1)?.step ?? 1;
  const x = (step: number) => MARGIN.left + (step / maxStep) * plotWidth;
  const y = (loss: number) => {
    const bounded = Math.max(Y_MIN, Math.min(Y_MAX, loss));
    const ratio =
      (Math.log10(Y_MAX) - Math.log10(bounded)) /
      (Math.log10(Y_MAX) - Math.log10(Y_MIN));
    return MARGIN.top + ratio * plotHeight;
  };

  const paths = useMemo(
    () => ({
      raw: pathFor(points, (point) => point.loss, x, y),
      ema: pathFor(points, (point) => point.lossEma50, x, y),
    }),
    // x and y are deterministic functions of points.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [points],
  );

  if (loadError) return null;
  if (points.length === 0) {
    return <div className={styles.lossLoading}>Loading training loss…</div>;
  }

  const activeIndex = selectedIndex ?? points.length - 1;
  const active = points[activeIndex];
  const xTicks = Array.from({ length: 7 }, (_, index) => (maxStep / 6) * index);

  return (
    <section className={styles.lossViewer} aria-labelledby="loss-viewer-title">
      <div className={styles.lossHeader}>
        <div>
          <span>OPTIMIZATION TRACE</span>
          <h2 id="loss-viewer-title">Training loss</h2>
        </div>
        <div className={styles.lossSelection} aria-live="polite">
          <span>STEP {active.step.toLocaleString()}</span>
          <strong>{formatLoss(active.lossEma50)}</strong>
          <small>50-step EMA</small>
        </div>
      </div>

      <div className={styles.lossChart}>
        <svg
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          role="img"
          aria-label="Training loss from step 1 through 30,000 on a logarithmic scale"
          onPointerLeave={() => setSelectedIndex(null)}
          onPointerMove={(event) => {
            const bounds = event.currentTarget.getBoundingClientRect();
            const svgX = ((event.clientX - bounds.left) / bounds.width) * WIDTH;
            const ratio = Math.max(
              0,
              Math.min(1, (svgX - MARGIN.left) / plotWidth),
            );
            const targetStep = ratio * maxStep;
            let nearest = 0;
            for (let index = 1; index < points.length; index += 1) {
              if (
                Math.abs(points[index].step - targetStep) <
                Math.abs(points[nearest].step - targetStep)
              ) {
                nearest = index;
              }
            }
            setSelectedIndex(nearest);
          }}
        >
          <defs>
            <linearGradient
              id={gradientId}
              x1="0"
              x2="1"
              y1="0"
              y2="0"
            >
              <stop offset="0%" stopColor="var(--gold)" />
              <stop offset="42%" stopColor="var(--mint)" />
              <stop offset="100%" stopColor="var(--mint)" />
            </linearGradient>
          </defs>

          {Y_TICKS.map((tick) => (
            <g key={tick}>
              <line
                className={styles.lossGrid}
                x1={MARGIN.left}
                x2={WIDTH - MARGIN.right}
                y1={y(tick)}
                y2={y(tick)}
              />
              <text
                className={styles.lossAxisLabel}
                x={MARGIN.left - 12}
                y={y(tick)}
                textAnchor="end"
                dominantBaseline="middle"
              >
                {tick}
              </text>
            </g>
          ))}

          {xTicks.map((tick) => (
            <text
              className={styles.lossAxisLabel}
              key={tick}
              x={x(tick)}
              y={HEIGHT - 13}
              textAnchor={
                tick === 0 ? "start" : tick === maxStep ? "end" : "middle"
              }
            >
              {tick === 0 ? "0" : `${Math.round(tick / 1_000)}k`}
            </text>
          ))}

          <path className={styles.lossRawLine} d={paths.raw} />
          <path
            className={styles.lossEmaLine}
            d={paths.ema}
            stroke={`url(#${gradientId})`}
          />

          <line
            className={styles.lossCursor}
            x1={x(active.step)}
            x2={x(active.step)}
            y1={MARGIN.top}
            y2={HEIGHT - MARGIN.bottom}
          />
          <circle
            className={styles.lossRawPoint}
            cx={x(active.step)}
            cy={y(active.loss)}
            r="3.5"
          />
          <circle
            className={styles.lossEmaPoint}
            cx={x(active.step)}
            cy={y(active.lossEma50)}
            r="5"
          />
        </svg>
      </div>

      <div className={styles.lossLegend}>
        <span><i className={styles.lossEmaSwatch} />50-step EMA</span>
        <span><i className={styles.lossRawSwatch} />sampled minibatch loss</span>
        <small>log scale · hover to inspect</small>
      </div>
    </section>
  );
}
