"use client";

import { useEffect, useRef, useState, type CSSProperties } from "react";
import { Pause, Play } from "lucide-react";

import styles from "./blocket-league-lab.module.css";

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

type Scenario = {
  id: string;
  title: string;
  description: string;
  atlas: string;
  atlasRow: number;
  event: string;
  meanEntityErrorPixels: number;
};

type Manifest = {
  frameSize: number;
  inputFrames: number;
  hallucinationFrames: number;
  playbackFps?: number;
  scenarios: Scenario[];
};

type SourceManifest = {
  frameSize: number;
  inputFrames?: number;
  hallucinationFrames?: number;
  contextFrames?: number;
  futureFrames?: number;
  playbackFps?: number;
  scenarios: {
    id: string;
    title: string;
    description: string;
    atlas: string;
    event?: string;
    meanEntityErrorPixels?: number;
    lanes?: { kind: "truth" | "sample" }[];
  }[];
};

function resolveAsset(manifestUrl: string, asset: string) {
  if (asset.startsWith("/")) return `${BASE_PATH}${asset}`;
  return `${manifestUrl.slice(0, manifestUrl.lastIndexOf("/") + 1)}${asset}`;
}

function normalizeManifest(
  source: SourceManifest,
  manifestUrl: string,
): Manifest {
  const inputFrames = source.inputFrames ?? source.contextFrames;
  const hallucinationFrames =
    source.hallucinationFrames ?? source.futureFrames;
  if (inputFrames === undefined || hallucinationFrames === undefined) {
    throw new Error("Unsupported hallucination manifest");
  }

  return {
    frameSize: source.frameSize,
    inputFrames,
    hallucinationFrames,
    playbackFps: source.playbackFps,
    scenarios: source.scenarios.map((scenario) => ({
      id: scenario.id,
      title: scenario.title,
      description: scenario.description,
      atlas: resolveAsset(manifestUrl, scenario.atlas),
      atlasRow: scenario.lanes
        ? Math.max(
            0,
            scenario.lanes.findIndex((lane) => lane.kind === "sample"),
          )
        : 0,
      event: scenario.event ?? "",
      meanEntityErrorPixels: scenario.meanEntityErrorPixels ?? 0,
    })),
  };
}

function FilmFrame({
  image,
  index,
  row,
  size,
  label,
}: {
  image: HTMLImageElement | null;
  index: number;
  row: number;
  size: number;
  label: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !image) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.imageSmoothingEnabled = false;
    context.drawImage(
      image,
      index * size,
      row * size,
      size,
      size,
      0,
      0,
      size,
      size,
    );
  }, [image, index, row, size]);

  return (
    <canvas
      ref={canvasRef}
      className={styles.hallucinationVideo}
      width={size}
      height={size}
      role="img"
      aria-label={`${label}, animated frame ${index + 1}`}
    />
  );
}

export function HallucinationFilmstrip({
  manifestUrl = `${BASE_PATH}/blocket-league/hallucinations/manifest.json`,
  compact = false,
}: {
  manifestUrl?: string;
  compact?: boolean;
}) {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [scenarioIndex, setScenarioIndex] = useState(0);
  const [atlas, setAtlas] = useState<HTMLImageElement | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch(manifestUrl)
      .then((response) => {
        if (!response.ok) throw new Error("Hallucination manifest unavailable");
        return response.json() as Promise<SourceManifest>;
      })
      .then((value) => {
        if (!cancelled) setManifest(normalizeManifest(value, manifestUrl));
      })
      .catch(() => { if (!cancelled) setError(true); });
    return () => { cancelled = true; };
  }, [manifestUrl]);

  const scenario = manifest?.scenarios[scenarioIndex];
  const totalFrames = manifest ? manifest.inputFrames + manifest.hallucinationFrames : 0;

  useEffect(() => {
    if (!scenario) return;
    let cancelled = false;
    const image = new Image();
    image.onload = () => { if (!cancelled) setAtlas(image); };
    image.src = scenario.atlas;
    return () => { cancelled = true; };
  }, [scenario]);

  useEffect(() => {
    if (!atlas || !manifest || !playing || totalFrames === 0) return;
    const timer = window.setInterval(() => {
      setFrameIndex((current) => (current + 1) % totalFrames);
    }, manifest.playbackFps ? 1_000 / manifest.playbackFps : 140);
    return () => window.clearInterval(timer);
  }, [atlas, manifest, playing, totalFrames]);

  if (error) return <p className={styles.trajectoryLoading}>Hallucination samples could not be loaded.</p>;
  if (!manifest || !scenario) return <p className={styles.trajectoryLoading}>Loading hallucinations…</p>;

  const hallucinating = frameIndex >= manifest.inputFrames;
  const inputShare = manifest.inputFrames / totalFrames;
  const trackInset = 20;
  const timelineStyle = {
    "--input-share": `${inputShare * 100}%`,
    "--timeline-boundary": `calc(${inputShare * 100}% + ${
      trackInset * (1 - 2 * inputShare)
    }px)`,
  } as CSSProperties;

  return (
    <div
      className={`${styles.hallucinationViewer} ${
        compact ? styles.hallucinationViewerCompact : ""
      }`}
    >
      <div className={styles.hallucinationTabs} role="group" aria-label="Held-out physical scenario">
        <div className={styles.hallucinationTabsHeader}>Samples</div>
        {manifest.scenarios.map((item, index) => (
          <button
            key={item.id}
            type="button"
            aria-pressed={scenarioIndex === index}
            className={scenarioIndex === index ? styles.hallucinationTabActive : undefined}
            onClick={() => {
              setAtlas(null);
              setFrameIndex(0);
              setPlaying(true);
              setScenarioIndex(index);
            }}
          >
            <span className={styles.hallucinationSampleTitle}>{item.title}</span>
            {scenarioIndex === index ? (
              <span className={styles.hallucinationSampleDescription}>
                {item.description}
              </span>
            ) : null}
          </button>
        ))}
      </div>
      <div className={styles.hallucinationStage}>
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
                value={frameIndex}
                aria-label="Scrub through observed and hallucinated frames"
                onChange={(event) => {
                  setFrameIndex(Number(event.target.value));
                  setPlaying(false);
                }}
              />
            </div>
          </div>
          <button type="button" onClick={() => setPlaying((value) => !value)}>
            {playing ? <Pause aria-hidden="true" /> : <Play aria-hidden="true" />}
            {playing ? "Pause" : "Play"}
          </button>
        </div>
        <FilmFrame
          image={atlas}
          index={frameIndex}
          row={scenario.atlasRow}
          size={manifest.frameSize}
          label={scenario.title}
        />
      </div>
    </div>
  );
}
