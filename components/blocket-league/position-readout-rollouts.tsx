"use client";

import { useEffect, useRef, useState } from "react";

import styles from "./blocket-league-lab.module.css";
import { HallucinationTimelineHeader } from "./hallucination-timeline-header";

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
const MANIFEST_URL = `${BASE_PATH}/blocket-league/position-rollouts/manifest.json`;

type DecodedPosition = [number, number, number, number] | null;

type Scenario = {
  id: string;
  title: string;
  description: string;
  atlas: string;
  decodedPositions: DecodedPosition[];
  meanPlayerErrorPx: number;
  meanPuckErrorPx: number;
  meanEntityErrorPx: number;
};

type Manifest = {
  frameSize: number;
  inputFrames: number;
  hallucinationFrames: number;
  playbackFps: number;
  fitSamples: number;
  block: number;
  scenarios: Scenario[];
};

function resolveAsset(asset: string) {
  return asset.startsWith("/") ? `${BASE_PATH}${asset}` : asset;
}

function ReadoutFrame({
  atlas,
  frame,
  frameSize,
  decoded,
  label,
}: {
  atlas: HTMLImageElement | null;
  frame: number;
  frameSize: number;
  decoded: DecodedPosition;
  label: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !atlas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.imageSmoothingEnabled = false;
    context.clearRect(0, 0, frameSize, frameSize);
    context.drawImage(
      atlas,
      frame * frameSize,
      0,
      frameSize,
      frameSize,
      0,
      0,
      frameSize,
      frameSize,
    );
  }, [atlas, frame, frameSize]);

  return (
    <div className={styles.positionReadoutFrame}>
      <canvas
        ref={canvasRef}
        className={styles.positionReadoutCanvas}
        width={frameSize}
        height={frameSize}
        role="img"
        aria-label={`${label}, frame ${frame + 1}, with linearly decoded player and puck positions`}
      />
      {decoded ? (
        <>
          <span
            aria-hidden="true"
            className={styles.positionMarker}
            style={{
              left: `${(decoded[0] / frameSize) * 100}%`,
              top: `${(decoded[1] / frameSize) * 100}%`,
            }}
          />
          <span
            aria-hidden="true"
            className={`${styles.positionMarker} ${styles.positionMarkerPuck}`}
            style={{
              left: `${(decoded[2] / frameSize) * 100}%`,
              top: `${(decoded[3] / frameSize) * 100}%`,
            }}
          />
        </>
      ) : null}
    </div>
  );
}

export function PositionReadoutRollouts() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [scenarioIndex, setScenarioIndex] = useState(0);
  const [atlas, setAtlas] = useState<HTMLImageElement | null>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch(MANIFEST_URL)
      .then((response) => {
        if (!response.ok) throw new Error("Position rollout manifest unavailable");
        return response.json() as Promise<Manifest>;
      })
      .then((value) => {
        if (cancelled) return;
        setManifest(value);
        setFrame(value.inputFrames);
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const scenario = manifest?.scenarios[scenarioIndex];
  const totalFrames = manifest
    ? manifest.inputFrames + manifest.hallucinationFrames
    : 0;

  useEffect(() => {
    if (!scenario) return;
    let cancelled = false;
    const image = new Image();
    image.onload = () => {
      if (!cancelled) setAtlas(image);
    };
    image.src = resolveAsset(scenario.atlas);
    return () => {
      cancelled = true;
    };
  }, [scenario]);

  useEffect(() => {
    if (!manifest || !atlas || !playing || totalFrames === 0) return;
    const timer = window.setInterval(() => {
      setFrame((current) =>
        current + 1 >= totalFrames ? manifest.inputFrames : current + 1
      );
    }, 1_000 / manifest.playbackFps);
    return () => window.clearInterval(timer);
  }, [atlas, manifest, playing, totalFrames]);

  if (failed) {
    return <p className={styles.trajectoryLoading}>Position-readout rollouts could not be loaded.</p>;
  }
  if (!manifest || !scenario) {
    return <p className={styles.trajectoryLoading}>Loading coordinate readouts…</p>;
  }

  const decoded = scenario.decodedPositions[frame] ?? null;

  return (
    <div className={styles.positionReadoutViewer}>
      <div className={styles.positionReadoutSamples}>
        <div className={styles.positionReadoutSamplesHeader}>Samples</div>
        {manifest.scenarios.map((item, index) => (
          <button
            type="button"
            aria-pressed={scenarioIndex === index}
            className={scenarioIndex === index ? styles.positionReadoutSampleActive : undefined}
            key={item.id}
            onClick={() => {
              setScenarioIndex(index);
              setAtlas(null);
              setFrame(manifest.inputFrames);
              setPlaying(true);
            }}
          >
            <span>{item.title}</span>
            {scenarioIndex === index ? <small>{item.description}</small> : null}
          </button>
        ))}
        <div className={styles.positionReadoutLegend}>
          <span data-entity="player"><i /> decoded player x/y</span>
          <span data-entity="puck"><i /> decoded puck x/y</span>
          <small>Crosshairs come from one fixed hidden token—not from the displayed pixels.</small>
        </div>
      </div>

      <div className={styles.positionReadoutStage}>
        <HallucinationTimelineHeader
          frame={frame}
          inputFrames={manifest.inputFrames}
          totalFrames={totalFrames}
          playing={playing}
          onFrameChange={(nextFrame) => {
            setFrame(nextFrame);
            setPlaying(false);
          }}
          onPlayingChange={setPlaying}
        />
        <ReadoutFrame
          atlas={atlas}
          frame={frame}
          frameSize={manifest.frameSize}
          decoded={decoded}
          label={scenario.title}
        />
      </div>
    </div>
  );
}
