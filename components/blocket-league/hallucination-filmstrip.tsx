"use client";

import { useEffect, useRef, useState } from "react";

import styles from "./blocket-league-lab.module.css";

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

type Scenario = {
  id: string;
  title: string;
  description: string;
  atlas: string;
  event: string;
  meanEntityErrorPixels: number;
};

type Manifest = {
  frameSize: number;
  inputFrames: number;
  hallucinationFrames: number;
  scenarios: Scenario[];
};

function FilmFrame({
  image,
  index,
  size,
  label,
}: {
  image: HTMLImageElement | null;
  index: number;
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
    context.drawImage(image, index * size, 0, size, size, 0, 0, size, size);
  }, [image, index, size]);

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

export function HallucinationFilmstrip() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [scenarioIndex, setScenarioIndex] = useState(0);
  const [atlas, setAtlas] = useState<HTMLImageElement | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch(`${BASE_PATH}/blocket-league/hallucinations/manifest.json`)
      .then((response) => {
        if (!response.ok) throw new Error("Hallucination manifest unavailable");
        return response.json() as Promise<Manifest>;
      })
      .then((value) => { if (!cancelled) setManifest(value); })
      .catch(() => { if (!cancelled) setError(true); });
    return () => { cancelled = true; };
  }, []);

  const scenario = manifest?.scenarios[scenarioIndex];
  const totalFrames = manifest ? manifest.inputFrames + manifest.hallucinationFrames : 0;

  useEffect(() => {
    if (!scenario) return;
    let cancelled = false;
    const image = new Image();
    image.onload = () => { if (!cancelled) setAtlas(image); };
    image.src = `${BASE_PATH}${scenario.atlas}`;
    return () => { cancelled = true; };
  }, [scenario]);

  useEffect(() => {
    if (!atlas || !playing || totalFrames === 0) return;
    const timer = window.setInterval(() => {
      setFrameIndex((current) => (current + 1) % totalFrames);
    }, 140);
    return () => window.clearInterval(timer);
  }, [atlas, playing, totalFrames]);

  if (error) return <p className={styles.trajectoryLoading}>Hallucination samples could not be loaded.</p>;
  if (!manifest || !scenario) return <p className={styles.trajectoryLoading}>Loading hallucinations…</p>;

  const hallucinating = frameIndex >= manifest.inputFrames;
  const phaseFrame = hallucinating
    ? frameIndex - manifest.inputFrames + 1
    : frameIndex + 1;
  const phaseTotal = hallucinating ? manifest.hallucinationFrames : manifest.inputFrames;

  return (
    <div className={styles.hallucinationViewer}>
      <div className={styles.hallucinationTabs} role="group" aria-label="Held-out physical scenario">
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
            {item.title}
          </button>
        ))}
        <p>{scenario.description}</p>
      </div>
      <div className={styles.hallucinationStage}>
        <div className={styles.hallucinationVideoHeader}>
          <div>
            <strong>{hallucinating ? "Hallucination" : "Input"}</strong>
            <span>{phaseFrame} / {phaseTotal}</span>
          </div>
          <button type="button" onClick={() => setPlaying((value) => !value)}>
            {playing ? "Pause" : "Play"}
          </button>
        </div>
        <FilmFrame
          image={atlas}
          index={frameIndex}
          size={manifest.frameSize}
          label={scenario.title}
        />
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
        <div className={styles.hallucinationPhases}>
          <div className={!hallucinating ? styles.hallucinationPhaseActive : undefined}>
            <strong>Input</strong><span>{manifest.inputFrames} observed frames</span>
          </div>
          <div className={hallucinating ? styles.hallucinationPhaseActive : undefined}>
            <strong>Hallucination</strong><span>{manifest.hallucinationFrames} predicted frames</span>
          </div>
        </div>
      </div>
    </div>
  );
}
