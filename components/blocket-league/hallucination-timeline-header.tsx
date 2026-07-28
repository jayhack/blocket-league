"use client";

import { type CSSProperties } from "react";
import { Pause, Play } from "lucide-react";

import styles from "./blocket-league-lab.module.css";

type HallucinationTimelineHeaderProps = {
  frame: number;
  inputFrames: number;
  totalFrames: number;
  playing: boolean;
  onFrameChange: (frame: number) => void;
  onPlayingChange: (playing: boolean) => void;
};

export function HallucinationTimelineHeader({
  frame,
  inputFrames,
  totalFrames,
  playing,
  onFrameChange,
  onPlayingChange,
}: HallucinationTimelineHeaderProps) {
  const hallucinating = frame >= inputFrames;
  const inputShare = inputFrames / totalFrames;
  const trackInset = 20;
  const timelineStyle = {
    "--input-share": `${inputShare * 100}%`,
    "--timeline-boundary": `calc(${inputShare * 100}% + ${
      trackInset * (1 - 2 * inputShare)
    }px)`,
  } as CSSProperties;

  return (
    <div className={styles.hallucinationVideoHeader}>
      <div className={styles.hallucinationTimeline} style={timelineStyle}>
        <div className={styles.hallucinationPhases}>
          <div className={!hallucinating ? styles.hallucinationPhaseActive : undefined}>
            <strong>Input</strong>
          </div>
          <div
            className={
              hallucinating
                ? `${styles.hallucinationPhaseActive} ${styles.hallucinationPhasePredictedActive}`
                : undefined
            }
          >
            <strong>Hallucination</strong>
          </div>
        </div>
        <div className={styles.hallucinationTransport}>
          <input
            type="range"
            min={0}
            max={totalFrames - 1}
            value={frame}
            aria-label="Scrub through observed and hallucinated frames"
            onChange={(event) => onFrameChange(Number(event.target.value))}
          />
        </div>
      </div>
      <button type="button" onClick={() => onPlayingChange(!playing)}>
        {playing ? <Pause aria-hidden="true" /> : <Play aria-hidden="true" />}
        {playing ? "Pause" : "Play"}
      </button>
    </div>
  );
}
