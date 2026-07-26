"use client";

import { useEffect, useState } from "react";

import styles from "./blocket-league-lab.module.css";

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

type Effect = {
  release_x_delta_px?: number;
  final_x_delta_px?: number;
  post_release_growth_px: number;
  expected_sign_fraction?: number;
};

type PixelManifest = {
  testSamples: number;
  causal: {
    writeFrames: number;
    rolloutFrames: number;
    effects: Record<string, Effect>;
  };
};

export function PixelInterpretabilityViewer() {
  const [manifest, setManifest] = useState<PixelManifest | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    fetch(`${BASE_PATH}/blocket-league/interpretability/passive-pixel-manifest.json`)
      .then((response) => {
        if (!response.ok) throw new Error("The passive-pixel study is unavailable");
        return response.json() as Promise<PixelManifest>;
      })
      .then((value) => { if (!cancelled) setManifest(value); })
      .catch((reason: unknown) => {
        if (!cancelled) setError(reason instanceof Error ? reason.message : String(reason));
      });
    return () => { cancelled = true; };
  }, []);

  if (error) return <p className={styles.interpretabilityLoading}>Study unavailable: {error}</p>;
  if (!manifest) return <p className={styles.interpretabilityLoading}>Loading passive-pixel study…</p>;

  const plus = manifest.causal.effects.x_plus;
  const minus = manifest.causal.effects.x_minus;
  const random = manifest.causal.effects.random;
  const value = (number: number | undefined) => number ?? 0;

  const plotTop = 84;
  const plotBottom = 326;
  const minEffect = -2.6;
  const maxEffect = 4;
  const y = (effect: number) => plotBottom
    - ((effect - minEffect) / (maxEffect - minEffect)) * (plotBottom - plotTop);

  const startX = 76;
  const releaseX = 374;
  const finalX = 844;
  const plusFinal = value(plus.final_x_delta_px);
  const minusFinal = value(minus.final_x_delta_px);
  const randomFinal = value(random.final_x_delta_px);
  const intendedFraction = (plus.expected_sign_fraction ?? 0) * 100;

  return (
    <figure className={styles.causalSimple}>
      <header className={styles.causalSimpleHeader}>
        <div>
          <p>
            We write one recovered motion direction into the hidden state for four frames,
            then let go. The model keeps carrying that motion forward.
          </p>
        </div>
        <div className={styles.causalSimpleMetric}>
          <strong>+{plus.post_release_growth_px.toFixed(2)} px</strong>
          <span>farther after the write stops</span>
        </div>
      </header>

      <div className={styles.causalSimpleChart}>
        <div className={styles.causalSimpleLegend} aria-hidden="true">
          <span data-series="positive"><i /> Push right</span>
          <span data-series="negative"><i /> Push left</span>
          <span data-series="random"><i /> Random direction</span>
        </div>

        <svg viewBox="0 0 960 390" role="img" aria-labelledby="causal-chart-title causal-chart-description">
          <title id="causal-chart-title">A brief hidden-state write causes persistent horizontal motion</title>
          <desc id="causal-chart-description">
            Horizontal displacement from frame zero to frame twelve for rightward, leftward,
            and random hidden-state interventions. The intervention ends at frame four, but
            the rightward and leftward effects continue to grow.
          </desc>

          <rect className={styles.causalWriteRegion} x={startX} y={plotTop} width={releaseX - startX} height={plotBottom - plotTop} />

          {[-2, 0, 2, 4].map((tick) => (
            <g key={tick} className={styles.causalGridline}>
              <line x1={startX} x2={finalX} y1={y(tick)} y2={y(tick)} />
              <text x={startX - 18} y={y(tick) + 6} textAnchor="end">{tick > 0 ? `+${tick}` : tick} px</text>
            </g>
          ))}

          <line className={styles.causalReleaseLine} x1={releaseX} x2={releaseX} y1={plotTop} y2={plotBottom} />
          <text className={styles.causalWriteLabel} x={(startX + releaseX) / 2} y={plotTop + 28} textAnchor="middle">WRITE ACTIVE</text>
          <text className={styles.causalReleaseLabel} x={releaseX + 16} y={plotTop + 28}>WRITE STOPS</text>

          <polyline
            className={styles.causalPositiveLine}
            points={`${startX},${y(0)} ${releaseX},${y(value(plus.release_x_delta_px))} ${finalX},${y(plusFinal)}`}
          />
          <polyline
            className={styles.causalNegativeLine}
            points={`${startX},${y(0)} ${releaseX},${y(value(minus.release_x_delta_px))} ${finalX},${y(minusFinal)}`}
          />
          <polyline
            className={styles.causalRandomLine}
            points={`${startX},${y(0)} ${releaseX},${y(value(random.release_x_delta_px))} ${finalX},${y(randomFinal)}`}
          />

          {[
            [releaseX, value(plus.release_x_delta_px), "positive"],
            [finalX, plusFinal, "positive"],
            [releaseX, value(minus.release_x_delta_px), "negative"],
            [finalX, minusFinal, "negative"],
            [releaseX, value(random.release_x_delta_px), "random"],
            [finalX, randomFinal, "random"],
          ].map(([cx, effect, series], index) => (
            <circle key={`${series}-${index}`} data-series={series} cx={Number(cx)} cy={y(Number(effect))} r="6" />
          ))}

          <text className={styles.causalEndpointPositive} x={finalX + 18} y={y(plusFinal) + 6}>+{plusFinal.toFixed(2)} px</text>
          <text className={styles.causalEndpointNegative} x={finalX + 18} y={y(minusFinal) + 6}>{minusFinal.toFixed(2)} px</text>
          <text className={styles.causalEndpointRandom} x={finalX + 18} y={y(randomFinal) + 6}>{randomFinal.toFixed(2)} px</text>

          <text className={styles.causalAxisLabel} x={startX} y="366" textAnchor="middle">Start</text>
          <text className={styles.causalAxisLabel} x={releaseX} y="366" textAnchor="middle">Frame {manifest.causal.writeFrames}</text>
          <text className={styles.causalAxisLabel} x={finalX} y="366" textAnchor="middle">Frame {manifest.causal.rolloutFrames}</text>
        </svg>
      </div>

      <figcaption className={styles.causalSimpleCaption}>
        After the intervention ends, displacement keeps growing instead of flattening out.
        Across {manifest.testSamples} unseen worlds, {intendedFraction.toFixed(1)}% move in the intended direction.
      </figcaption>
    </figure>
  );
}
