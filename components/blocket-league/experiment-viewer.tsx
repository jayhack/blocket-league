"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Pause, Play, RotateCcw } from "lucide-react";

import styles from "./experiment.module.css";

type ExperimentLane = {
  id: string;
  label: string;
  kind: "truth" | "sample";
  playerErrorPx: number;
  puckErrorPx: number;
  shortPlayerErrorPx?: number;
  shortPuckErrorPx?: number;
};

type ExperimentScenario = {
  id: string;
  title: string;
  description: string;
  atlas: string;
  events: string[];
  lanes: ExperimentLane[];
};

type ExperimentManifest = {
  frameSize: number;
  contextFrames: number;
  futureFrames: number;
  playbackFps: number;
  checkpointStep: number;
  generationLabel: string;
  metricBoundary: number;
  scenarios: ExperimentScenario[];
};

function resolveAsset(manifestUrl: string, asset: string) {
  if (asset.startsWith("/")) return asset;
  return `${manifestUrl.slice(0, manifestUrl.lastIndexOf("/") + 1)}${asset}`;
}

export function ExperimentViewer({ manifestUrl }: { manifestUrl: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [manifest, setManifest] = useState<ExperimentManifest | null>(null);
  const [atlas, setAtlas] = useState<HTMLImageElement | null>(null);
  const [scenarioIndex, setScenarioIndex] = useState(0);
  const [laneIndex, setLaneIndex] = useState(1);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [loadError, setLoadError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch(manifestUrl)
      .then((response) => {
        if (!response.ok) throw new Error("experiment manifest unavailable");
        return response.json() as Promise<ExperimentManifest>;
      })
      .then((value) => {
        if (!cancelled) setManifest(value);
      })
      .catch(() => {
        if (!cancelled) setLoadError(true);
      });
    return () => {
      cancelled = true;
    };
  }, [manifestUrl]);

  const scenario = manifest?.scenarios[scenarioIndex] ?? null;
  const lane = scenario?.lanes[laneIndex] ?? null;
  const totalFrames = manifest
    ? manifest.contextFrames + manifest.futureFrames
    : 1;

  useEffect(() => {
    if (!scenario) return;
    let cancelled = false;
    const image = new Image();
    image.onload = () => {
      if (!cancelled) setAtlas(image);
    };
    image.src = resolveAsset(manifestUrl, scenario.atlas);
    return () => {
      cancelled = true;
    };
  }, [manifestUrl, scenario]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !atlas || !manifest || !lane) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.imageSmoothingEnabled = false;
    context.clearRect(0, 0, manifest.frameSize, manifest.frameSize);
    context.drawImage(
      atlas,
      frame * manifest.frameSize,
      laneIndex * manifest.frameSize,
      manifest.frameSize,
      manifest.frameSize,
      0,
      0,
      manifest.frameSize,
      manifest.frameSize,
    );
  }, [atlas, frame, lane, laneIndex, manifest]);

  useEffect(() => {
    if (!manifest || !playing) return;
    const timer = window.setTimeout(() => {
      setFrame((current) => {
        if (current >= totalFrames - 1) return 0;
        return current + 1;
      });
    }, 1_000 / manifest.playbackFps);
    return () => window.clearTimeout(timer);
  }, [frame, manifest, playing, totalFrames]);

  const futureIndex = manifest ? frame - manifest.contextFrames : -1;
  const isObserved = futureIndex < 0;
  const event = useMemo(() => {
    if (!scenario || futureIndex < 0) return "shared observation";
    return scenario.events[futureIndex] ?? "coast";
  }, [futureIndex, scenario]);

  const selectScenario = (index: number) => {
    setScenarioIndex(index);
    setLaneIndex(1);
    setAtlas(null);
    setFrame(0);
    setPlaying(true);
  };

  if (loadError) {
    return (
      <div className={styles.loading}>Experiment samples could not be loaded.</div>
    );
  }
  if (!manifest || !scenario || !lane) {
    return <div className={styles.loading}>Loading checkpoint samples…</div>;
  }

  const useShortMetrics = futureIndex < manifest.metricBoundary;
  const playerError =
    useShortMetrics && lane.shortPlayerErrorPx !== undefined
      ? lane.shortPlayerErrorPx
      : lane.playerErrorPx;
  const puckError =
    useShortMetrics && lane.shortPuckErrorPx !== undefined
      ? lane.shortPuckErrorPx
      : lane.puckErrorPx;

  return (
    <div className={styles.viewer}>
      <div
        className={styles.scenarioPicker}
        role="group"
        aria-label="Held-out scenario"
      >
        {manifest.scenarios.map((option, index) => (
          <button
            key={option.id}
            type="button"
            className={index === scenarioIndex ? styles.active : undefined}
            onClick={() => selectScenario(index)}
            aria-pressed={index === scenarioIndex}
          >
            <span>0{index + 1}</span>
            <strong>{option.title}</strong>
          </button>
        ))}
      </div>

      <div className={styles.viewerHeader}>
        <div>
          <strong>{isObserved ? "Input" : lane.label}</strong>
          <span>
            {isObserved
              ? `observed ${frame + 1} / ${manifest.contextFrames}`
              : `predicted ${futureIndex + 1} / ${manifest.futureFrames} · ${event}`}
          </span>
        </div>
        <button type="button" onClick={() => setPlaying((value) => !value)}>
          {playing ? <Pause aria-hidden="true" /> : <Play aria-hidden="true" />}
          {playing ? "Pause" : "Play"}
        </button>
      </div>

      <div className={styles.stage}>
        <canvas
          ref={canvasRef}
          width={manifest.frameSize}
          height={manifest.frameSize}
          role="img"
          aria-label={`${scenario.title}, ${lane.label}, frame ${frame + 1}`}
        />
      </div>

      <div className={styles.transport}>
        <input
          type="range"
          min={0}
          max={totalFrames - 1}
          value={frame}
          onChange={(event) => {
            setFrame(Number(event.target.value));
            setPlaying(false);
          }}
          aria-label="Experiment frame"
        />
        <button
          type="button"
          onClick={() => {
            setFrame(0);
            setPlaying(true);
          }}
        >
          <RotateCcw aria-hidden="true" />
          Restart
        </button>
      </div>

      <div className={styles.footer}>
        <div className={styles.lanePicker} role="group" aria-label="Sample lane">
          {scenario.lanes.map((option, index) => (
            <button
              type="button"
              key={option.id}
              className={index === laneIndex ? styles.laneActive : undefined}
              onClick={() => setLaneIndex(index)}
              aria-pressed={index === laneIndex}
            >
              <strong>{option.label}</strong>
              <span>{option.kind === "truth" ? "reference" : manifest.generationLabel}</span>
            </button>
          ))}
        </div>
        <div className={styles.readout}>
          <span>{isObserved ? "CONTEXT" : "AUTOREGRESSIVE"}</span>
          <strong>
            {lane.kind === "truth" || isObserved
              ? "reference frame"
              : `P ${playerError.toFixed(1)} · K ${puckError.toFixed(1)} px`}
          </strong>
          <small>checkpoint step {manifest.checkpointStep.toLocaleString()}</small>
        </div>
      </div>

      <p className={styles.description}>{scenario.description}</p>
    </div>
  );
}

