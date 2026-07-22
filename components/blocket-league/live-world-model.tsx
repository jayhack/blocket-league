"use client";

import { useCallback, useEffect, useId, useRef, useState } from "react";
import { Cpu, RotateCcw } from "lucide-react";

import { ACTION_NAMES, ACTION_VECTORS, keyboardAction } from "@/lib/blocket-league/sim";

import styles from "./blocket-league-lab.module.css";

const PAD_ACTIONS = [
  { action: 1, label: "↑" },
  { action: 7, label: "←" },
  { action: 3, label: "→" },
  { action: 5, label: "↓" },
] as const;
const MOVEMENT_KEYS = new Set(["ArrowUp", "ArrowRight", "ArrowDown", "ArrowLeft", "w", "a", "s", "d"]);
const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

type OrtRuntime = typeof import("onnxruntime-web");
type OrtSession = import("onnxruntime-web").InferenceSession;

type LiveManifest = {
  modelKind: "passive-direct-pixel-autoregressive";
  checkpointStep: number;
  sourceFps: number;
  frameSize: number;
  patchSize: number;
  gridSize: number;
  historyFrames: number;
  hiddenSize: number;
  interventionBlock: number;
  interventionStrength: number;
  modelParameters: number;
  modelBytes: number;
  palette: number[][];
  directions: { x: number[]; y: number[] };
  assets: { dynamics: string; starterContext: string; starterFrame: string };
};

type EngineState = {
  runtime: OrtRuntime;
  dynamics: OrtSession;
  provider: "webgpu" | "wasm";
  manifest: LiveManifest;
  starterContext: Float32Array;
  history: Float32Array;
  lastGreenSpatialMask: Float32Array;
};

type DreamFrame = { image: ImageData; action: number };
type PlayerStatus = "idle" | "loading" | "ready" | "running" | "paused" | "error";

let activeLivePlayerId: string | null = null;

function assetUrl(path: string) {
  return `${BASE_PATH}${path}`;
}

function normalizeMovementKey(key: string) {
  return key.length === 1 ? key.toLowerCase() : key;
}

async function fetchBytes(url: string, onProgress?: (loaded: number) => void) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url} (${response.status})`);
  if (!response.body) return new Uint8Array(await response.arrayBuffer());
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress?.(loaded);
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

function classesToImage(classes: Float32Array, manifest: LiveManifest) {
  const pixels = manifest.frameSize * manifest.frameSize;
  const rgba = new Uint8ClampedArray(pixels * 4);
  for (let pixel = 0; pixel < pixels; pixel += 1) {
    const color = manifest.palette[Number(classes[pixel])] ?? manifest.palette[0];
    const output = pixel * 4;
    rgba[output] = color[0];
    rgba[output + 1] = color[1];
    rgba[output + 2] = color[2];
    rgba[output + 3] = 255;
  }
  return new ImageData(rgba, manifest.frameSize, manifest.frameSize);
}

function logitsToClasses(logits: Float32Array, manifest: LiveManifest) {
  const pixels = manifest.frameSize * manifest.frameSize;
  const output = new Float32Array(pixels);
  for (let pixel = 0; pixel < pixels; pixel += 1) {
    let bestClass = 0;
    let bestValue = -Infinity;
    for (let classIndex = 0; classIndex < manifest.palette.length; classIndex += 1) {
      const value = logits[classIndex * pixels + pixel];
      if (value > bestValue) {
        bestValue = value;
        bestClass = classIndex;
      }
    }
    output[pixel] = bestClass;
  }
  return output;
}

export function greenTokenMask(
  history: Float32Array,
  manifest: LiveManifest,
  fallback?: Float32Array,
) {
  const pixels = manifest.frameSize * manifest.frameSize;
  const offset = history.length - pixels;
  const spatial = new Float32Array(manifest.gridSize * manifest.gridSize);
  let peakMass = 0;
  for (let pixel = 0; pixel < pixels; pixel += 1) {
    const value = Number(history[offset + pixel]);
    if (value !== 5 && value !== 6) continue;
    const patchX = Math.floor((pixel % manifest.frameSize) / manifest.patchSize);
    const patchY = Math.floor(Math.floor(pixel / manifest.frameSize) / manifest.patchSize);
    const patch = patchY * manifest.gridSize + patchX;
    spatial[patch] += 1;
    peakMass = Math.max(peakMass, spatial[patch]);
  }
  const mask = new Float32Array(manifest.historyFrames * manifest.gridSize * manifest.gridSize);
  const routed = peakMass > 0 ? spatial : fallback;
  if (!routed?.some((value) => value > 0)) return { mask, spatial };
  const scale = peakMass > 0 ? peakMass : Math.max(...routed);
  const timeOffset = (manifest.historyFrames - 1) * manifest.gridSize * manifest.gridSize;
  for (let patch = 0; patch < routed.length; patch += 1) {
    mask[timeOffset + patch] = routed[patch] / Math.max(scale, 1);
  }
  return { mask, spatial: peakMass > 0 ? spatial : routed.slice() };
}

function steeringVector(action: number, manifest: LiveManifest) {
  const vector = ACTION_VECTORS[action] ?? ACTION_VECTORS[0];
  const length = Math.hypot(vector.x, vector.y) || 1;
  const x = vector.x / length;
  const y = vector.y / length;
  const output = new Float32Array(manifest.hiddenSize);
  for (let index = 0; index < output.length; index += 1) {
    output[index] = manifest.interventionStrength * (
      x * manifest.directions.x[index] + y * manifest.directions.y[index]
    );
  }
  return output;
}

async function generateFrame(engine: EngineState, action: number) {
  const { manifest, runtime } = engine;
  const greenRouting = greenTokenMask(engine.history, manifest, engine.lastGreenSpatialMask);
  const result = await engine.dynamics.run({
    pixel_history: new runtime.Tensor(
      "float32",
      engine.history,
      [1, manifest.historyFrames, manifest.frameSize, manifest.frameSize],
    ),
    intervention: new runtime.Tensor(
      "float32",
      steeringVector(action, manifest),
      [1, manifest.hiddenSize],
    ),
    intervention_mask: new runtime.Tensor(
      "float32",
      greenRouting.mask,
      [1, manifest.historyFrames, manifest.gridSize * manifest.gridSize],
    ),
  });
  const logitsTensor = result.next_logits as import("onnxruntime-web").Tensor;
  const next = logitsToClasses(logitsTensor.data as Float32Array, manifest);
  logitsTensor.dispose();
  const pixels = manifest.frameSize * manifest.frameSize;
  engine.history.copyWithin(0, pixels);
  engine.history.set(next, engine.history.length - pixels);
  engine.lastGreenSpatialMask = greenRouting.spatial;
  return classesToImage(next, manifest);
}

export function LiveWorldModel() {
  const instanceId = useId();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<EngineState | null>(null);
  const keysRef = useRef(new Set<string>());
  const manualActionRef = useRef<number | null>(null);
  const startPlaybackRef = useRef<() => void>(() => {});
  const runningRef = useRef(false);
  const loopIdRef = useRef(0);
  const playbackTimerRef = useRef<number | null>(null);
  const queueRef = useRef<DreamFrame[]>([]);
  const [status, setStatus] = useState<PlayerStatus>("idle");
  const [loadProgress, setLoadProgress] = useState(0);
  const [error, setError] = useState("");
  const [inputAction, setInputAction] = useState(0);

  const drawStarter = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const image = new Image();
    image.onload = () => {
      const context = canvas.getContext("2d");
      if (!context) return;
      context.imageSmoothingEnabled = false;
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
    };
    image.src = assetUrl("/blocket-league/live/starter-frame.png");
  }, []);

  useEffect(() => { drawStarter(); }, [drawStarter]);

  const stopPlayback = useCallback(() => {
    runningRef.current = false;
    loopIdRef.current += 1;
    if (playbackTimerRef.current !== null) {
      window.clearInterval(playbackTimerRef.current);
      playbackTimerRef.current = null;
    }
  }, []);

  useEffect(() => () => {
    stopPlayback();
    const engine = engineRef.current;
    if (engine) void engine.dynamics.release();
  }, [stopPlayback]);

  const loadModel = async () => {
    if (status === "loading" || engineRef.current) return;
    setStatus("loading");
    setError("");
    try {
      setLoadProgress(0.02);
      const response = await fetch(assetUrl("/blocket-league/live/manifest.json"));
      if (!response.ok) throw new Error("The browser model manifest is missing.");
      const manifest = await response.json() as LiveManifest;
      if (manifest.modelKind !== "passive-direct-pixel-autoregressive") {
        throw new Error("The loaded checkpoint is not the passive pixel model.");
      }
      let loaded = 0;
      const [modelBytes, starterBytes] = await Promise.all([
        fetchBytes(assetUrl(manifest.assets.dynamics), (value) => {
          loaded = value;
          setLoadProgress(0.05 + 0.7 * loaded / manifest.modelBytes);
        }),
        fetchBytes(assetUrl(manifest.assets.starterContext)),
      ]);
      setLoadProgress(0.78);
      const runtime = await import("onnxruntime-web/webgpu");
      runtime.env.logLevel = "warning";
      runtime.env.wasm.numThreads = 1;
      runtime.env.wasm.proxy = false;
      let selectedProvider: "webgpu" | "wasm" = "wasm";
      let dynamics: OrtSession;
      if ("gpu" in navigator) {
        try {
          dynamics = await runtime.InferenceSession.create(modelBytes, {
            executionProviders: ["webgpu"], graphOptimizationLevel: "all",
          });
          selectedProvider = "webgpu";
        } catch (gpuError) {
          console.warn("WebGPU initialization failed; falling back to WASM.", gpuError);
          dynamics = await runtime.InferenceSession.create(modelBytes, {
            executionProviders: ["wasm"], graphOptimizationLevel: "all",
          });
        }
      } else {
        dynamics = await runtime.InferenceSession.create(modelBytes, {
          executionProviders: ["wasm"], graphOptimizationLevel: "all",
        });
      }
      const starterContext = new Float32Array(starterBytes.slice().buffer);
      const expected = manifest.historyFrames * manifest.frameSize * manifest.frameSize;
      if (starterContext.length !== expected) {
        await dynamics.release();
        throw new Error(`Starter context has ${starterContext.length} pixels; expected ${expected}.`);
      }
      engineRef.current = {
        runtime: runtime as OrtRuntime,
        dynamics,
        provider: selectedProvider,
        manifest,
        starterContext: starterContext.slice(),
        history: starterContext.slice(),
        lastGreenSpatialMask: greenTokenMask(starterContext, manifest).spatial,
      };
      setLoadProgress(1);
      setStatus("ready");
      window.setTimeout(() => startPlaybackRef.current(), 0);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : String(loadError));
      setStatus("error");
    }
  };

  const startPlayback = () => {
    const engine = engineRef.current;
    if (!engine || runningRef.current) return;
    runningRef.current = true;
    const loopId = loopIdRef.current + 1;
    loopIdRef.current = loopId;
    setStatus("running");
    playbackTimerRef.current = window.setInterval(() => {
      const next = queueRef.current.shift();
      if (!next) return;
      const context = canvasRef.current?.getContext("2d");
      if (context) {
        context.imageSmoothingEnabled = false;
        context.putImageData(next.image, 0, 0);
      }
    }, 1_000 / engine.manifest.sourceFps);
    const inferenceLoop = async () => {
      while (runningRef.current && loopIdRef.current === loopId) {
        if (queueRef.current.length >= 2) {
          await new Promise((resolve) => window.setTimeout(resolve, 8));
          continue;
        }
        const action = manualActionRef.current ?? keyboardAction(keysRef.current);
        try {
          const image = await generateFrame(engine, action);
          if (!runningRef.current || loopIdRef.current !== loopId) return;
          queueRef.current.push({ image, action });
        } catch (inferenceError) {
          stopPlayback();
          setError(inferenceError instanceof Error ? inferenceError.message : String(inferenceError));
          setStatus("error");
          return;
        }
      }
    };
    void inferenceLoop();
  };

  useEffect(() => { startPlaybackRef.current = startPlayback; });

  useEffect(() => {
    if (activeLivePlayerId === null) activeLivePlayerId = instanceId;
    const publish = () => setInputAction(manualActionRef.current ?? keyboardAction(keysRef.current));
    const keyDown = (event: KeyboardEvent) => {
      if (activeLivePlayerId !== instanceId) return;
      const key = normalizeMovementKey(event.key);
      if (!MOVEMENT_KEYS.has(key)) return;
      event.preventDefault();
      keysRef.current.add(key);
      publish();
      if (engineRef.current && !runningRef.current) startPlaybackRef.current();
    };
    const keyUp = (event: KeyboardEvent) => {
      if (activeLivePlayerId !== instanceId) return;
      const key = normalizeMovementKey(event.key);
      if (!MOVEMENT_KEYS.has(key)) return;
      event.preventDefault();
      keysRef.current.delete(key);
      publish();
    };
    const clear = () => { keysRef.current.clear(); manualActionRef.current = null; setInputAction(0); };
    window.addEventListener("keydown", keyDown);
    window.addEventListener("keyup", keyUp);
    window.addEventListener("blur", clear);
    return () => {
      window.removeEventListener("keydown", keyDown);
      window.removeEventListener("keyup", keyUp);
      window.removeEventListener("blur", clear);
      if (activeLivePlayerId === instanceId) activeLivePlayerId = null;
    };
  }, [instanceId]);

  const resetDream = () => {
    stopPlayback();
    queueRef.current = [];
    const engine = engineRef.current;
    if (engine) {
      engine.history = engine.starterContext.slice();
      engine.lastGreenSpatialMask = greenTokenMask(engine.history, engine.manifest).spatial;
    }
    keysRef.current.clear();
    manualActionRef.current = null;
    setInputAction(0);
    setError("");
    drawStarter();
    setStatus(engine ? "ready" : "idle");
    if (engine) window.setTimeout(() => startPlaybackRef.current(), 0);
  };

  const beginManualAction = (action: number) => {
    manualActionRef.current = action;
    setInputAction(action);
    if (engineRef.current && !runningRef.current) startPlaybackRef.current();
  };
  const endManualAction = () => {
    manualActionRef.current = null;
    setInputAction(keyboardAction(keysRef.current));
  };

  return (
    <div
      className={styles.livePlayer}
      onFocusCapture={() => { activeLivePlayerId = instanceId; }}
      onPointerEnter={() => { activeLivePlayerId = instanceId; }}
    >
      <div className={styles.livePlayerGrid}>
        <div className={styles.liveDreamColumn}>
          <div className={styles.liveCanvasWrap}>
            <canvas ref={canvasRef} className={styles.liveCanvas} width={64} height={64} role="img" aria-label="Live frames imagined by the passive Blocket League pixel transformer." />
            {(status === "idle" || status === "loading" || status === "error") && (
              <div className={styles.liveCanvasOverlay}>
                {status === "idle" && <button type="button" onClick={loadModel}><Cpu aria-hidden="true" /> Load local model</button>}
                {status === "loading" && <><div className={styles.liveLoadTrack} aria-label={`${Math.round(loadProgress * 100)} percent loaded`}><span style={{ width: `${loadProgress * 100}%` }} /></div><strong>{Math.round(loadProgress * 100)}%</strong></>}
                {status === "error" && <button type="button" onClick={loadModel}><RotateCcw aria-hidden="true" /> Try again</button>}
              </div>
            )}
          </div>
        </div>
        <aside className={styles.liveControls} aria-label="Activation steering controls">
          <div className={styles.pad} aria-label="Hidden-state direction pad">
            {PAD_ACTIONS.map(({ action, label }) => (
              <button key={action} type="button" className={inputAction === action ? styles.padActive : undefined} aria-label={`${ACTION_NAMES[action]} activation write`} aria-pressed={inputAction === action} data-pad-action={action} onPointerDown={() => beginManualAction(action)} onPointerUp={endManualAction} onPointerCancel={endManualAction} onPointerLeave={endManualAction}>{label}</button>
            ))}
          </div>
          <button className={styles.liveReset} type="button" onClick={resetDream} disabled={status === "idle" || status === "loading"}><RotateCcw aria-hidden="true" /> Reset</button>
          {error && <p className={styles.liveError}>{error}</p>}
        </aside>
      </div>
    </div>
  );
}
