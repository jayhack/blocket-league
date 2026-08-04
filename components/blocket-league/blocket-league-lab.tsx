"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  RotateCcw,
} from "lucide-react";

import {
  createPassiveWorld,
  resetPassiveRound,
  snapshotWorld,
  stepWorld,
  WORLD,
  type Vec2,
  type WorldState,
} from "@/lib/blocket-league/sim";
import type { BlocketLeagueCopy } from "@/lib/blocket-league/content-types";
import { experiments } from "@/lib/blocket-league/experiments";

import styles from "./blocket-league-lab.module.css";
import { CollisionHoldoutViewer } from "./collision-holdout-viewer";
import { EditableMarkdown } from "./editable-markdown";
import { DirectionHoldoutViewer } from "./direction-holdout-viewer";
import { HallucinationFilmstrip } from "./hallucination-filmstrip";
import { LiveWorldModel } from "./live-world-model";
import {
  CollisionAnticipationViewer,
  MotionRingViewer,
} from "./physics-emergence-viewer";
import { PixelInterpretabilityViewer } from "./pixel-interpretability-viewer";
import { PositionGeometryViewer } from "./position-geometry-viewer";
import { PositionReadoutRollouts } from "./position-readout-rollouts";

const TABLE_OF_CONTENTS = [
  { id: "top", label: "Introduction" },
  { id: "play", label: "Play the game" },
  { id: "training", label: "Train a toy model" },
  { id: "generalization", label: "Generalization" },
  { id: "linear-position", label: "Decode the representation" },
  { id: "direction-ring", label: "Direction ring" },
  { id: "lens", label: "Jacobian lens" },
  { id: "intervention", label: "Causal intervention" },
  { id: "game-surgery", label: "Brain surgery" },
  { id: "conclusion", label: "Conclusion" },
  { id: "experiments", label: "Appendix A · Model scale" },
  { id: "direction-holdout", label: "Appendix B · Direction holdout" },
  { id: "collision-holdout", label: "Appendix C · Spatial holdout" },
  { id: "position-geometry", label: "Appendix D · Cartesian position" },
  { id: "collision-representation", label: "Appendix E · Collision probe" },
  { id: "experiment-index", label: "Appendix F · Experiments" },
] as const;

const MODEL_HISTORY = [
  { label: "t−7", player: [24, 68], puck: [72, 28] },
  { label: "t−6", player: [28, 63], puck: [69, 31] },
  { label: "t−5", player: [33, 58], puck: [65, 35] },
  { label: "t−4", player: [38, 53], puck: [61, 39] },
  { label: "t−3", player: [43, 48], puck: [57, 43] },
  { label: "t−2", player: [48, 43], puck: [53, 47] },
  { label: "t−1", player: [53, 38], puck: [49, 51] },
  { label: "t", player: [58, 33], puck: [45, 55] },
] as const;

const NANO_EXPERIMENT = experiments.find(
  (experiment) => experiment.slug === "nano-1p5mb",
)!;

function GitHubMark() {
  return (
    <svg aria-hidden="true" viewBox="0 0 16 16">
      <path
        fill="currentColor"
        d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.6-.18-3.28-.8-3.28-3.56 0-.79.28-1.43.74-1.93-.07-.18-.32-.91.07-1.9 0 0 .6-.19 1.97.74A6.9 6.9 0 0 1 8 4.8a6.9 6.9 0 0 1 1.8.24c1.37-.93 1.97-.74 1.97-.74.39.99.14 1.72.07 1.9.46.5.74 1.14.74 1.93 0 2.77-1.69 3.38-3.3 3.56.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"
      />
    </svg>
  );
}

function TableOfContents() {
  const [activeId, setActiveId] = useState<string>(TABLE_OF_CONTENTS[0].id);

  useEffect(() => {
    const sections = TABLE_OF_CONTENTS.map(({ id }) => document.getElementById(id)).filter(
      (section): section is HTMLElement => section !== null,
    );

    const updateActiveSection = () => {
      const readingLine = window.innerHeight * 0.3;
      let nextId = sections[0]?.id ?? TABLE_OF_CONTENTS[0].id;

      for (const section of sections) {
        if (section.getBoundingClientRect().top <= readingLine) nextId = section.id;
      }

      setActiveId(nextId);
    };

    updateActiveSection();
    window.addEventListener("scroll", updateActiveSection, { passive: true });
    window.addEventListener("resize", updateActiveSection);

    return () => {
      window.removeEventListener("scroll", updateActiveSection);
      window.removeEventListener("resize", updateActiveSection);
    };
  }, []);

  return (
    <nav className={styles.tableOfContents} aria-label="Table of contents">
      <span className={styles.tableOfContentsTitle}>Contents</span>
      <ol>
        {TABLE_OF_CONTENTS.map(({ id, label }) => (
          <li key={id}>
            <a
              className={activeId === id ? styles.tableOfContentsActive : undefined}
              href={`#${id}`}
              aria-current={activeId === id ? "location" : undefined}
            >
              {label}
            </a>
          </li>
        ))}
      </ol>
    </nav>
  );
}

function DiagramFrame({
  label,
  player,
  puck,
  predicted = false,
}: {
  label: string;
  player: readonly [number, number];
  puck: readonly [number, number];
  predicted?: boolean;
}) {
  return (
    <div className={`${styles.diagramFrame} ${predicted ? styles.diagramFramePredicted : ""}`}>
      <span className={styles.diagramGoal} />
      <span
        className={styles.diagramPlayer}
        style={{ left: `${player[0]}%`, top: `${player[1]}%` }}
      />
      <span
        className={styles.diagramPuck}
        style={{ left: `${puck[0]}%`, top: `${puck[1]}%` }}
      />
      <small>{label}</small>
    </div>
  );
}

function drawCelestialDisc(
  context: CanvasRenderingContext2D,
  position: Vec2,
  radius: number,
  colors: {
    glow: string;
    outer: string;
    middle: string;
    inner: string;
    core: string;
  },
) {
  const halo = context.createRadialGradient(
    position.x,
    position.y,
    radius * 0.25,
    position.x,
    position.y,
    radius * 1.75,
  );
  halo.addColorStop(0, colors.glow);
  halo.addColorStop(0.55, colors.glow.replace(/[\d.]+\)$/, "0.12)"));
  halo.addColorStop(1, "rgba(0, 0, 0, 0)");
  context.fillStyle = halo;
  context.beginPath();
  context.arc(position.x, position.y, radius * 1.75, 0, Math.PI * 2);
  context.fill();

  const surface = context.createRadialGradient(
    position.x - radius * 0.18,
    position.y - radius * 0.18,
    radius * 0.04,
    position.x,
    position.y,
    radius,
  );
  surface.addColorStop(0, colors.core);
  surface.addColorStop(0.28, colors.inner);
  surface.addColorStop(0.64, colors.middle);
  surface.addColorStop(1, colors.outer);
  context.fillStyle = surface;
  context.beginPath();
  context.arc(position.x, position.y, radius, 0, Math.PI * 2);
  context.fill();

  context.strokeStyle = "rgba(235, 242, 244, 0.42)";
  context.lineWidth = radius * 0.045;
  context.stroke();
}

function drawWorld(
  canvas: HTMLCanvasElement,
  state: WorldState,
) {
  const bounds = canvas.getBoundingClientRect();
  if (bounds.width <= 0) return;
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  const side = bounds.width;
  const width = Math.round(side * pixelRatio);
  if (canvas.width !== width || canvas.height !== width) {
    canvas.width = width;
    canvas.height = width;
  }
  const context = canvas.getContext("2d");
  if (!context) return;
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, side, side);
  context.fillStyle = "#010106";
  context.fillRect(0, 0, side, side);

  context.save();
  context.scale(side, side);
  const wall = WORLD.wall;
  context.fillStyle = "#020610";
  context.fillRect(wall, wall, 1 - wall * 2, 1 - wall * 2);

  context.save();
  context.beginPath();
  context.rect(wall, wall, 1 - wall * 2, 1 - wall * 2);
  context.clip();
  const wash = context.createRadialGradient(0.48, 0.44, 0.03, 0.5, 0.5, 0.72);
  wash.addColorStop(0, "rgba(4, 93, 173, 0.28)");
  wash.addColorStop(0.48, "rgba(0, 43, 111, 0.18)");
  wash.addColorStop(1, "rgba(1, 2, 10, 0)");
  context.fillStyle = wash;
  context.fillRect(wall, wall, 1 - wall * 2, 1 - wall * 2);
  context.restore();

  context.strokeStyle = "rgba(91, 151, 183, 0.3)";
  context.lineWidth = 0.004;
  context.beginPath();
  context.moveTo(0.5, wall);
  context.lineTo(0.5, 1 - wall);
  context.stroke();
  context.strokeStyle = "rgba(113, 177, 200, 0.38)";
  context.beginPath();
  context.arc(0.5, 0.5, 0.14, 0, Math.PI * 2);
  context.stroke();

  const goalGlow = context.createLinearGradient(
    1 - wall - 0.06,
    0,
    1 - wall + 0.01,
    0,
  );
  goalGlow.addColorStop(0, "rgba(227, 35, 20, 0)");
  goalGlow.addColorStop(0.55, "rgba(239, 53, 20, 0.34)");
  goalGlow.addColorStop(1, "rgba(255, 194, 55, 0.64)");
  context.fillStyle = goalGlow;
  context.fillRect(1 - wall - 0.045, WORLD.goalLow, 0.045, WORLD.goalHigh - WORLD.goalLow);
  context.strokeStyle = "#073f9c";
  context.lineWidth = 0.018;
  context.lineCap = "round";
  context.beginPath();
  context.moveTo(wall, wall);
  context.lineTo(1 - wall, wall);
  context.moveTo(wall, 1 - wall);
  context.lineTo(1 - wall, 1 - wall);
  context.moveTo(wall, wall);
  context.lineTo(wall, 1 - wall);
  context.moveTo(1 - wall, wall);
  context.lineTo(1 - wall, WORLD.goalLow);
  context.moveTo(1 - wall, WORLD.goalHigh);
  context.lineTo(1 - wall, 1 - wall);
  context.stroke();

  context.strokeStyle = "#15bce4";
  context.lineWidth = 0.006;
  context.stroke();

  context.strokeStyle = "#ffba32";
  context.lineWidth = 0.014;
  context.beginPath();
  context.moveTo(1 - wall, WORLD.goalLow);
  context.lineTo(1 - wall, WORLD.goalHigh);
  context.stroke();

  drawCelestialDisc(context, state.playerPosition, WORLD.playerRadius, {
    glow: "rgba(255, 72, 18, 0.4)",
    outer: "#e32613",
    middle: "#ff5b1a",
    inner: "#ffb52f",
    core: "#ffe173",
  });
  drawCelestialDisc(context, state.puckPosition, WORLD.puckRadius, {
    glow: "rgba(20, 157, 255, 0.38)",
    outer: "#123fa8",
    middle: "#087ac7",
    inner: "#19c4e8",
    core: "#b7f4ff",
  });

  if (state.resetTimer > 0) {
    context.fillStyle = "rgba(1, 2, 8, 0.82)";
    context.fillRect(0.3, 0.43, 0.4, 0.14);
    context.strokeStyle = "#0b8dd0";
    context.lineWidth = 0.003;
    context.strokeRect(0.3, 0.43, 0.4, 0.14);
    context.fillStyle = "#ffc53d";
    context.font = "500 0.055px system-ui, sans-serif";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText("GOAL", 0.5, 0.5);
  }
  context.restore();
}

export function BlocketLeagueLab({
  copy,
  editable = false,
}: {
  copy: BlocketLeagueCopy;
  editable?: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [initialWorld] = useState(() => createPassiveWorld(17));
  const worldRef = useRef<WorldState>(initialWorld);
  const [snapshot, setSnapshot] = useState(() => snapshotWorld(initialWorld));

  useEffect(() => {
    let animationFrame = 0;
    let previous = performance.now();
    let accumulator = 0;
    const stepDuration = 1_000 / WORLD.fps;

    const animate = (now: number) => {
      const world = worldRef.current;
      accumulator = Math.min(accumulator + now - previous, 250);
      previous = now;
      let advanced = false;
      while (accumulator >= stepDuration) {
        stepWorld(world, 0, true);
        accumulator -= stepDuration;
        advanced = true;
      }
      if (advanced) setSnapshot(snapshotWorld(world));
      if (canvasRef.current) drawWorld(canvasRef.current, world);
      animationFrame = requestAnimationFrame(animate);
    };
    animationFrame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationFrame);
  }, []);

  const reset = () => {
    const world = worldRef.current;
    resetPassiveRound(world, true);
    setSnapshot(snapshotWorld(world));
  };

  return (
    <main className={styles.root}>
      <TableOfContents />
      <header className={styles.header}>
        <a className={styles.backLink} href="https://www.jay.ai/writing">
          <ArrowLeft aria-hidden="true" />
          <span>All writing</span>
        </a>
        <a className={styles.githubLink} href="https://github.com/jayhack/blocket-league" target="_blank" rel="noreferrer">
          <GitHubMark />
          <span>GitHub</span>
        </a>
      </header>

      <section className={styles.hero} id="top">
        <h1>
          J-Lens for video models:
          <br />
          uncovering steerable physical dynamics.
        </h1>
        <div className={styles.heroMeta}>
          <span>by <a href="https://jay.ai" target="_blank" rel="noreferrer">Jay Hack</a></span>
          <span aria-hidden="true">·</span>
          <a href="https://github.com/jayhack/blocket-league" target="_blank" rel="noreferrer">View the code ↗</a>
        </div>
        <EditableMarkdown blockId="hero-intro" markdown={copy["hero-intro"]} editable={editable} className={styles.heroCopy} />
        <EditableMarkdown blockId="hero-sources" markdown={copy["hero-sources"]} editable={editable} className={styles.heroCopy} />
        <div className={styles.heroGameIntro} id="play">
          <EditableMarkdown
            blockId="play-title"
            markdown={copy["play-title"]}
            editable={editable}
            headingId="play-title"
          />
          <EditableMarkdown blockId="play-intro" markdown={copy["play-intro"]} editable={editable} className={styles.heroGameCopy} />
        </div>
        <div className={styles.heroGame}>
          <LiveWorldModel />
        </div>
        <div className={styles.heroGameIntro}>
          <EditableMarkdown blockId="play-takeaway" markdown={copy["play-takeaway"]} editable={editable} className={styles.heroGameCopy} />
        </div>
      </section>

      <section className={styles.modelSection} id="training" aria-labelledby="training-title">
        <div className={styles.sectionHeading}>
          <div>
            <EditableMarkdown blockId="model-title" markdown={copy["model-title"]} editable={editable} headingId="training-title" />
          </div>
          <EditableMarkdown blockId="model" markdown={copy.model} editable={editable} className={styles.sectionCopy} />
        </div>

        <figure className={styles.simpleArchitectureFigure}>
          <div
            className={styles.simpleArchitecture}
            role="img"
            aria-label="Three raw frames enter a five-layer pixel transformer, which predicts the next frame."
          >
            <div className={styles.simpleArchitectureFrames} aria-hidden="true">
              {MODEL_HISTORY.slice(-3).map((frame) => (
                <DiagramFrame key={frame.label} {...frame} />
              ))}
            </div>
            <ArrowRight className={styles.simpleArchitectureArrow} aria-hidden="true" />
            <div className={styles.simpleTransformerBlock}>
              <strong>Transformer</strong>
              <span>5 layers</span>
            </div>
            <ArrowRight className={styles.simpleArchitectureArrow} aria-hidden="true" />
            <div className={styles.simpleArchitectureOutput} aria-hidden="true">
              <DiagramFrame
                label=""
                player={[63, 29]}
                puck={[41, 59]}
                predicted
              />
            </div>
          </div>
          <figcaption className={styles.architectureSubtitle}>
            A 5-layer pixel transformer on raw frame data of physics rollouts.{" "}
            <a
              href="https://github.com/jayhack/blocket-league/blob/main/blocket_league/pixel_direct_model.py"
              target="_blank"
              rel="noreferrer"
            >
              View the code on GitHub ↗
            </a>
          </figcaption>
        </figure>

        <EditableMarkdown
          blockId="dataset"
          markdown={copy.dataset}
          editable={editable}
          className={styles.trainingStepCopy}
        />

        <div className={`${styles.simulatorShell} ${styles.trainingSimulator}`}>
          <div className={styles.canvasColumn}>
            <div className={styles.canvasHeader}>
              <div className={styles.score} aria-label={`${snapshot.score} goals scored`}>
                <strong>{snapshot.score.toString().padStart(2, "0")}</strong>
                <span>{snapshot.score === 1 ? "goal" : "goals"}</span>
              </div>
              <button className={styles.simulatorReset} type="button" onClick={reset}>
                <RotateCcw aria-hidden="true" /> Reset world
              </button>
            </div>
            <canvas
              ref={canvasRef}
              className={styles.canvas}
              role="img"
              aria-label="A galactic square physics arena with a warm solar disc, a blue satellite disc, and a gold goal on the right."
            />
          </div>
          <p className={styles.simulatorClaim}>
            Blocket League
          </p>
        </div>
        <EditableMarkdown blockId="model-results" markdown={copy["model-results"]} editable={editable} className={styles.modelResultsCopy} />
        <HallucinationFilmstrip />
      </section>

      <section
        className={styles.generalizationSection}
        id="generalization"
        aria-labelledby="generalization-title"
      >
        <div className={styles.sectionHeading}>
          <div>
            <EditableMarkdown
              blockId="generalization-title"
              markdown={copy["generalization-title"]}
              editable={editable}
              headingId="generalization-title"
            />
          </div>
          <EditableMarkdown
            blockId="generalization-intro"
            markdown={copy["generalization-intro"]}
            editable={editable}
            className={styles.sectionCopy}
          />
        </div>
        <div className={styles.generalizationSampler}>
          <HallucinationFilmstrip
            compact
            manifestUrl="/experiments/collision-holdout-upper-right/manifest.json"
            scenarioPrefix="upper-right"
          />
        </div>
        <div className={styles.positionReadoutConclusion}>
          <strong>Collision physics transfers into the missing quadrant.</strong>
          <p>
            Against a matched control trained everywhere, the penalty in the unseen
            quadrant is only 0.8%.
          </p>
        </div>
      </section>

      <section
        className={styles.positionReadoutSection}
        id="linear-position"
        aria-labelledby="linear-position-title"
      >
        <div className={styles.sectionHeading}>
          <div>
            <EditableMarkdown
              blockId="linear-position-title"
              markdown={copy["linear-position-title"]}
              editable={editable}
              headingId="linear-position-title"
            />
          </div>
          <EditableMarkdown
            blockId="linear-position-intro"
            markdown={copy["linear-position-intro"]}
            editable={editable}
            className={styles.sectionCopy}
          />
        </div>
        <PositionReadoutRollouts />
        <div className={styles.positionReadoutConclusion}>
          <strong>The crosshairs follow the model&apos;s own hallucinations.</strong>
          <p>
            The same fixed-token decoder keeps tracking both objects even when the
            model is feeding its own predicted pixels back as input.
          </p>
          <Link href="#position-geometry">
            See the full layer-by-layer and random-weight controls in Appendix D ↓
          </Link>
        </div>
      </section>

      <section className={styles.emergenceSection} id="direction-ring" aria-labelledby="direction-ring-title">
        <div className={styles.sectionHeading}>
          <div>
            <EditableMarkdown blockId="representation-title" markdown={copy["representation-title"]} editable={editable} headingId="direction-ring-title" />
          </div>
          <EditableMarkdown blockId="representation-depth" markdown={copy["representation-depth"]} editable={editable} className={styles.sectionCopy} />
        </div>
        <MotionRingViewer />
      </section>

      <section className={styles.lensSection} id="lens" aria-labelledby="lens-title">
        <div className={styles.sectionHeading}>
          <div>
            <EditableMarkdown blockId="jacobian-title" markdown={copy["jacobian-title"]} editable={editable} headingId="lens-title" />
          </div>
          <EditableMarkdown blockId="jacobian-lens" markdown={copy["jacobian-lens"]} editable={editable} className={styles.sectionCopy} />
        </div>

        <div
          className={styles.lensDiagram}
          role="img"
          aria-label="Across 512 rendered trajectories, measure how block five activations affect the player disc's next-frame position, then average those effects into reusable horizontal and vertical motion directions."
        >
          <div className={styles.lensFlow}>
            <div className={`${styles.lensStage} ${styles.lensContexts}`}>
              <div className={styles.diagramStageHeader}>
                <strong>Sample 512 worlds</strong>
              </div>
              <div className={styles.lensStageVisual}>
                <div className={styles.contextFan} aria-hidden="true">
                  {MODEL_HISTORY.slice(2, 5).map((frame, index) => (
                    <DiagramFrame key={frame.label} {...frame} label={`world ${index + 1}`} />
                  ))}
                </div>
              </div>
              <p className={styles.lensStageCaption}>
                Run the frozen model on many randomized motion histories.
              </p>
            </div>

            <div className={`${styles.lensStage} ${styles.lensTrace}`}>
              <div className={styles.diagramStageHeader}>
                <strong>Trace downstream motion</strong>
              </div>
              <div className={`${styles.lensStageVisual} ${styles.lensTraceVisual}`} aria-hidden="true">
                <div className={styles.activationGrid}>
                  {Array.from({ length: 25 }, (_, index) => (
                    <span key={index} className={index === 17 ? styles.activationCellActive : undefined} />
                  ))}
                </div>
                <ArrowRight />
                <div className={styles.centroidBoard}>
                  <span className={styles.centroidDisc} />
                  <span className={styles.centroidCrossX} />
                  <span className={styles.centroidCrossY} />
                </div>
              </div>
              <p className={styles.lensStageCaption}>
                Backpropagate the player disc&apos;s next-frame x/y position to its block-5 hidden state.
              </p>
            </div>

            <div className={`${styles.lensStage} ${styles.lensDirections}`}>
              <div className={styles.diagramStageHeader}>
                <strong>Average one direction</strong>
              </div>
              <div className={styles.lensStageVisual}>
                <div className={styles.directionAxes} aria-hidden="true">
                  <div><span>v<sub>x</sub></span><i>→</i></div>
                  <div><span>v<sub>y</sub></span><i>↓</i></div>
                </div>
              </div>
              <p className={styles.lensStageCaption}>
                Average across worlds to get a reusable vector we can write into the residual-stream activations.
              </p>
            </div>
          </div>

        </div>
      </section>

      <section className={styles.interpretabilitySection} id="intervention" aria-labelledby="interpretability-title">
        <div className={styles.sectionHeading}>
          <div>
            <EditableMarkdown blockId="causal-title" markdown={copy["causal-title"]} editable={editable} headingId="interpretability-title" />
          </div>
          <EditableMarkdown blockId="causal-intervention" markdown={copy["causal-intervention"]} editable={editable} className={styles.sectionCopy} />
        </div>
        <PixelInterpretabilityViewer />
      </section>

      <section className={styles.liveSection} id="game-surgery" aria-labelledby="live-title">
        <div className={styles.sectionHeading}>
          <div>
            <EditableMarkdown blockId="brain-surgery-title" markdown={copy["brain-surgery-title"]} editable={editable} headingId="live-title" />
          </div>
          <EditableMarkdown blockId="brain-surgery" markdown={copy["brain-surgery"]} editable={editable} className={styles.sectionCopy} />
        </div>
        <LiveWorldModel />
      </section>

      <section
        className={styles.conclusionSection}
        id="conclusion"
        aria-labelledby="conclusion-title"
      >
        <div className={styles.sectionHeading}>
          <div>
            <EditableMarkdown
              blockId="conclusion-title"
              markdown={copy["conclusion-title"]}
              editable={editable}
              headingId="conclusion-title"
            />
          </div>
          <EditableMarkdown
            blockId="conclusion"
            markdown={copy.conclusion}
            editable={editable}
            className={`${styles.sectionCopy} ${styles.conclusionCopy}`}
          />
        </div>
      </section>

      <section
        className={styles.experimentSection}
        id="experiments"
        aria-labelledby="experiments-title"
      >
        <div className={styles.sectionHeading}>
          <div>
            <EditableMarkdown blockId="model-scale-title" markdown={copy["model-scale-title"]} editable={editable} headingId="experiments-title" />
          </div>
          <p className={styles.sectionCopy}>
            We compressed the pixel transformer from 3.67 million parameters to
            377,136—about one tenth the size—and retrained it for 12,000 steps.
            The 1.54 MB model learns the arena and coarse motion, but its entities
            fragment during long autoregressive rollouts: 64-frame position error
            rises from 6.53 px to 19.10 px. For this recipe, 377,136 parameters is
            below the useful capacity floor.
          </p>
        </div>

        <Link className={styles.experimentCard} href={`/${NANO_EXPERIMENT.slug}`}>
          <div className={styles.experimentCardBody}>
            <div>
              <h3>{NANO_EXPERIMENT.title}</h3>
              <p>{NANO_EXPERIMENT.verdict}</p>
            </div>
            <dl>
              {NANO_EXPERIMENT.metrics.slice(0, 3).map((metric) => (
                <div key={metric.label}>
                  <dt>{metric.label}</dt>
                  <dd>{metric.value}</dd>
                </div>
              ))}
            </dl>
          </div>
          <div className={styles.experimentCardAction}>
            View the full experiment, metrics, and checkpoint samples →
          </div>
        </Link>
      </section>

      <section className={styles.directionHoldoutSection} id="direction-holdout" aria-labelledby="direction-holdout-title">
        <div className={styles.sectionHeading}>
          <div>
            <EditableMarkdown
              blockId="direction-holdout-title"
              markdown={copy["direction-holdout-title"]}
              editable={editable}
              headingId="direction-holdout-title"
            />
          </div>
          <p className={styles.sectionCopy}>
            We trained two new transformers from scratch. For one, we rejected every
            24-frame training world in which the puck ever moved within ±30° of due east;
            the matched control saw all angles. Then we forced both models to begin in
            eight controlled direction bins and rolled each one forward from pixels.
          </p>
        </div>
        <DirectionHoldoutViewer />
        <div className={styles.directionHoldoutConclusion}>
          <strong>It generalizes, but not for free.</strong>
          <p>
            The held-out model reaches 0.95 px error due east—almost identical to its
            0.99 px average on directions present in training. There is no catastrophic
            hole at the missing angle. Yet the all-angle control reaches 0.56 px on the
            same due-east worlds, so removing that 60° wedge costs 71% in accuracy.
            The transformer has learned a direction-general transition rule, while direct
            experience still sharpens it.
          </p>
          <div className={styles.directionHoldoutLinks}>
            <Link href="/direction-holdout-east-60/">
              View the held-out direction samples →
            </Link>
            <a href="https://github.com/jayhack/blocket-league/blob/main/docs/direction-holdout-experiment.md" target="_blank" rel="noreferrer">
              Read the protocol and reproduction commands ↗
            </a>
          </div>
        </div>
      </section>

      <section
        className={styles.collisionHoldoutSection}
        id="collision-holdout"
        aria-labelledby="collision-holdout-title"
      >
        <div className={styles.sectionHeading}>
          <div>
            <EditableMarkdown
              blockId="collision-holdout-title"
              markdown={copy["collision-holdout-title"]}
              editable={editable}
              headingId="collision-holdout-title"
            />
          </div>
          <p className={styles.sectionCopy}>
            We rejected every 24-frame training world containing a player–puck
            collision in the upper-right quadrant, then tested collisions centered
            in all four quadrants. The model could still observe free motion in the
            missing region; only collision outcomes were removed.
          </p>
        </div>
        <CollisionHoldoutViewer />
        <div className={styles.directionHoldoutConclusion}>
          <strong>Yes—almost perfectly.</strong>
          <p>
            On 32 upper-right collision worlds, the held-out model reaches 1.257 px
            puck error versus 1.247 px for the matched all-location control: just a
            0.8% penalty. Its error in the unseen quadrant is only 7.5% above its own
            three-quadrant average. In contrast to the direction holdout, the
            transformer appears to have learned collision dynamics as a
            location-independent rule.
          </p>
          <div className={styles.directionHoldoutLinks}>
            <Link href="/collision-holdout-upper-right/">
              View the held-out collision samples →
            </Link>
            <a href="https://github.com/jayhack/blocket-league/blob/main/docs/spatial-collision-holdout-experiment.md" target="_blank" rel="noreferrer">
              Read the protocol and reproduction commands ↗
            </a>
          </div>
        </div>
      </section>

      <section
        className={styles.positionGeometrySection}
        id="position-geometry"
        aria-labelledby="position-geometry-title"
      >
        <div className={styles.sectionHeading}>
          <div>
            <EditableMarkdown
              blockId="position-geometry-title"
              markdown={copy["position-geometry-title"]}
              editable={editable}
              headingId="position-geometry-title"
            />
          </div>
          <p className={styles.sectionCopy}>
            Earlier position probes selected the token containing the rendered
            entity, so they could partly recover absolute patch identity. Here the
            decoder receives only one fixed hidden state: the bottom-right token
            after the eighth observed frame. We ask it to report the player and
            puck&apos;s x/y coordinates, then exclude the upper-right quadrant from
            probe fitting to test whether horizontal and vertical position recombine.
          </p>
        </div>
        <PositionGeometryViewer />
        <div className={styles.directionHoldoutConclusion}>
          <strong>The map is readable; the causal handle is downstream.</strong>
          <p>
            Spatial attention broadcasts both objects&apos; locations into a fixed
            token, and a linear transform recovers separate x/y axes. But adding
            that regression vector back into the residual stream does not teleport
            the puck. Linear decodability alone is correlational. When we instead
            orient the write with the downstream Jacobian of the rendered puck
            centroid, the same position variable becomes steerable: one brief write
            moves the puck in the intended direction in 84–90% of worlds immediately,
            and the altered trajectory continues separating after the write stops.
          </p>
          <p>
            This still does not prove that a collision circuit literally consumes
            this particular coordinate readout. The next decisive test is geometric
            mediation: decompose a collide-versus-miss activation patch into the
            x/y-and-velocity subspace and its orthogonal remainder, then ask which
            component causes the model to apply a collision-like change in
            post-contact velocity.
          </p>
          <div className={styles.directionHoldoutLinks}>
            <a href="https://github.com/jayhack/blocket-league/blob/main/docs/position-geometry-experiment.md" target="_blank" rel="noreferrer">
              Read the protocol and reproduction commands ↗
            </a>
          </div>
        </div>
      </section>

      <section
        className={styles.emergenceSection}
        id="collision-representation"
        aria-labelledby="collision-representation-title"
      >
        <div className={styles.sectionHeading}>
          <div>
            <EditableMarkdown
              blockId="collision-representation-title"
              markdown={copy["collision-representation-title"]}
              editable={editable}
              headingId="collision-representation-title"
            />
          </div>
          <EditableMarkdown
            blockId="collision-representation"
            markdown={copy["collision-representation"]}
            editable={editable}
            className={styles.sectionCopy}
          />
        </div>
        <CollisionAnticipationViewer />
      </section>

      <section
        className={styles.experimentIndexSection}
        id="experiment-index"
        aria-labelledby="experiment-index-title"
      >
        <div className={styles.sectionHeading}>
          <div>
            <EditableMarkdown
              blockId="experiment-index-title"
              markdown={copy["experiment-index-title"]}
              editable={editable}
              headingId="experiment-index-title"
            />
          </div>
          <p className={styles.sectionCopy}>
            Each run links to its checkpoint metadata, metrics, and held-out
            sample rollouts.
          </p>
        </div>

        <div className={styles.experimentIndex}>
          {experiments.map((experiment) => (
            <Link
              className={styles.experimentIndexRow}
              href={`/${experiment.slug}`}
              key={experiment.slug}
            >
              <div className={styles.experimentIndexSummary}>
                <strong>{experiment.title}</strong>
                <span>{experiment.description}</span>
              </div>
              <div className={styles.experimentIndexMetric}>
                <span>{experiment.metrics[0].label}</span>
                <strong>{experiment.metrics[0].value}</strong>
              </div>
              <span className={styles.experimentIndexArrow} aria-hidden="true">→</span>
            </Link>
          ))}
        </div>
      </section>

      <footer className={styles.footer}>
        <span>BLOCKET LEAGUE</span>
        <a
          href="https://github.com/jayhack/blocket-league"
          target="_blank"
          rel="noreferrer"
        >
          View the code on GitHub ↗
        </a>
      </footer>
    </main>
  );
}
