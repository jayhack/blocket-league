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
  imagined,
  label,
}: {
  image: HTMLImageElement | null;
  index: number;
  size: number;
  imagined: boolean;
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
      className={`${styles.hallucinationFrame} ${imagined ? styles.hallucinationFrameImagined : ""}`}
      width={size}
      height={size}
      role="img"
      aria-label={`${label}, frame ${index + 1}, ${imagined ? "model hallucination" : "input"}`}
    />
  );
}

export function HallucinationFilmstrip() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [scenarioIndex, setScenarioIndex] = useState(0);
  const [atlas, setAtlas] = useState<HTMLImageElement | null>(null);
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

  useEffect(() => {
    if (!scenario) return;
    let cancelled = false;
    const image = new Image();
    image.onload = () => { if (!cancelled) setAtlas(image); };
    image.src = `${BASE_PATH}${scenario.atlas}`;
    return () => { cancelled = true; };
  }, [scenario]);

  if (error) return <p className={styles.trajectoryLoading}>Hallucination samples could not be loaded.</p>;
  if (!manifest || !scenario) return <p className={styles.trajectoryLoading}>Loading hallucinations…</p>;

  const totalFrames = manifest.inputFrames + manifest.hallucinationFrames;
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
              setScenarioIndex(index);
            }}
          >
            {item.title}
          </button>
        ))}
      </div>
      <div className={styles.hallucinationScenarioCopy}>
        <strong>{scenario.title}</strong>
        <span>{scenario.description}</span>
      </div>
      <div className={styles.hallucinationGrid}>
        {Array.from({ length: totalFrames }, (_, index) => (
          <FilmFrame
            key={index}
            image={atlas}
            index={index}
            size={manifest.frameSize}
            imagined={index >= manifest.inputFrames}
            label={scenario.title}
          />
        ))}
      </div>
      <div className={styles.hallucinationPhases}>
        <div><strong>Input</strong><span>{manifest.inputFrames} observed frames</span></div>
        <div><strong>Hallucination</strong><span>{manifest.hallucinationFrames} autoregressive frames</span></div>
      </div>
    </div>
  );
}
