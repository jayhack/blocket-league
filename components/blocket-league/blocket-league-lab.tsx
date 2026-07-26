"use client";

import { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  CircleDot,
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

import styles from "./blocket-league-lab.module.css";
import { EditableMarkdown } from "./editable-markdown";
import { HallucinationFilmstrip } from "./hallucination-filmstrip";
import { LiveWorldModel } from "./live-world-model";
import { PhysicsEmergenceViewer } from "./physics-emergence-viewer";
import { PixelInterpretabilityViewer } from "./pixel-interpretability-viewer";

const TABLE_OF_CONTENTS = [
  { id: "top", label: "Introduction" },
  { id: "play", label: "Play the game" },
  { id: "world", label: "Dataset" },
  { id: "model", label: "Pixel transformer" },
  { id: "lens", label: "Jacobian lens" },
  { id: "intervention", label: "Causal intervention" },
  { id: "game-surgery", label: "Brain surgery" },
  { id: "representations", label: "Representation geometry" },
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
        <a className={styles.wordmark} href="#top" aria-label="Blocket League home">
          <span className={styles.mark}><CircleDot aria-hidden="true" /></span>
          <span>BLOCKET LEAGUE</span>
        </a>
        <a className={styles.githubLink} href="https://github.com/jayhack/blocket-league" target="_blank" rel="noreferrer">
          GitHub ↗
        </a>
      </header>

      <section className={styles.hero} id="top">
        <h1>J-Lens for video models: uncovering steerable, interpretable physical dynamics.</h1>
        <div className={styles.heroMeta}>
          <span>by <a href="https://jay.ai" target="_blank" rel="noreferrer">Jay Hack</a></span>
          <span aria-hidden="true">·</span>
          <a href="https://github.com/jayhack/blocket-league" target="_blank" rel="noreferrer">View the code ↗</a>
        </div>
        <EditableMarkdown blockId="hero-intro" markdown={copy["hero-intro"]} editable={editable} className={styles.heroCopy} />
        <EditableMarkdown blockId="hero-sources" markdown={copy["hero-sources"]} editable={editable} className={styles.heroCopy} />
        <div className={styles.heroGameIntro} id="play">
          <h2>Play the game: &quot;Blocket League&quot;</h2>
          <EditableMarkdown blockId="play-intro" markdown={copy["play-intro"]} editable={editable} className={styles.heroGameCopy} />
        </div>
        <div className={styles.heroGame}>
          <LiveWorldModel />
        </div>
        <div className={styles.heroGameIntro}>
          <EditableMarkdown blockId="play-takeaway" markdown={copy["play-takeaway"]} editable={editable} className={styles.heroGameCopy} />
        </div>
      </section>

      <section className={styles.labSection} id="world" aria-labelledby="world-title">
        <div className={styles.sectionHeading}>
          <div>
            <h2 id="world-title">The dataset: raw physics rollouts from &quot;Blocket League&quot;.</h2>
          </div>
          <EditableMarkdown blockId="dataset" markdown={copy.dataset} editable={editable} className={styles.sectionCopy} />
        </div>

        <div className={styles.simulatorShell}>
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
        </div>
      </section>

      <section className={styles.modelSection} id="model" aria-labelledby="model-title">
        <div className={styles.sectionHeading}>
          <div>
            <h2 id="model-title">The model: pixel transformers for unsupervised prediction.</h2>
          </div>
          <EditableMarkdown blockId="model" markdown={copy.model} editable={editable} className={styles.sectionCopy} />
        </div>

        <div
          className={styles.simpleArchitecture}
          role="img"
          aria-label="Three previous rendered frames enter a causal video transformer, which predicts the next rendered frame."
        >
          <div className={styles.simpleArchitectureFrames} aria-hidden="true">
            {MODEL_HISTORY.slice(-3).map((frame) => (
              <DiagramFrame key={frame.label} {...frame} />
            ))}
          </div>
          <ArrowRight className={styles.simpleArchitectureArrow} aria-hidden="true" />
          <div className={styles.simpleTransformerBlock}>
            <strong>Transformer</strong>
            <span>6 causal blocks</span>
          </div>
          <ArrowRight className={styles.simpleArchitectureArrow} aria-hidden="true" />
          <div className={styles.simpleArchitectureOutput} aria-hidden="true">
            <DiagramFrame
              label="next frame"
              player={[63, 29]}
              puck={[41, 59]}
              predicted
            />
          </div>
          <div className={styles.simpleArchitectureLabels}>
            <span>Previous frames</span>
            <span>Predict the next frame</span>
          </div>
        </div>
        <EditableMarkdown blockId="model-results" markdown={copy["model-results"]} editable={editable} className={styles.modelResultsCopy} />
        <HallucinationFilmstrip />
      </section>

      <section className={styles.lensSection} id="lens" aria-labelledby="lens-title">
        <div className={styles.sectionHeading}>
          <div>
            <h2 id="lens-title">Identifying latent activations for physical phenomena</h2>
          </div>
          <EditableMarkdown blockId="jacobian-lens" markdown={copy["jacobian-lens"]} editable={editable} className={styles.sectionCopy} />
        </div>

        <div
          className={styles.lensDiagram}
          role="img"
          aria-label="Across 512 rendered trajectories, select the block five activation at the green puck's spatial token, run the frozen downstream model to the next rendered frame, backpropagate the green puck centroid's x and y coordinates, and average those gradients into reusable x and y velocity directions."
        >
          <div className={styles.lensFlow}>
            <div className={`${styles.lensStage} ${styles.lensContexts}`}>
              <div className={styles.diagramStageHeader}>
                <span>SAMPLE</span>
                <strong>512 worlds</strong>
                <small>separate fit contexts</small>
              </div>
              <div className={styles.contextFan} aria-hidden="true">
                {MODEL_HISTORY.slice(2, 5).map((frame, index) => (
                  <DiagramFrame key={frame.label} {...frame} label={`world ${index + 1}`} />
                ))}
              </div>
            </div>

            <ArrowRight className={styles.lensArrow} aria-hidden="true" />

            <div className={`${styles.lensStage} ${styles.lensActivation}`}>
              <div className={styles.diagramStageHeader}>
                <span>LOCATE</span>
                <strong>h<sub>ℓ,p</sub> at block 5</strong>
                <small>p = green-puck spatial token</small>
              </div>
              <div className={styles.activationGrid} aria-hidden="true">
                {Array.from({ length: 25 }, (_, index) => (
                  <span key={index} className={index === 17 ? styles.activationCellActive : undefined} />
                ))}
              </div>
              <div className={styles.activationVector}>192D activation</div>
            </div>

            <div className={styles.jacobianBridge}>
              <div className={styles.forwardRail}><span>FROZEN FORWARD</span><ArrowRight aria-hidden="true" /></div>
              <div className={styles.bridgeBlocks}>
                <span>B6</span><span>NORM</span><span>PIXEL HEAD</span>
              </div>
              <div className={styles.backwardRail}><ArrowRight aria-hidden="true" /><span>BACKPROP ∂(x̂, ŷ) / ∂h</span></div>
            </div>

            <div className={`${styles.lensStage} ${styles.lensReadout}`}>
              <div className={styles.diagramStageHeader}>
                <span>MEASURE</span>
                <strong>Green-puck centroid</strong>
                <small>next-frame x/y readout from green logits</small>
              </div>
              <div className={styles.centroidBoard} aria-hidden="true">
                <span className={styles.centroidDisc} />
                <span className={styles.centroidCrossX} />
                <span className={styles.centroidCrossY} />
              </div>
              <div className={styles.centroidCoordinates}>ŷ = (x̂, ŷ)</div>
            </div>

            <ArrowRight className={styles.lensArrow} aria-hidden="true" />

            <div className={`${styles.lensStage} ${styles.lensDirections}`}>
              <div className={styles.diagramStageHeader}>
                <span>AVERAGE</span>
                <strong>Velocity directions</strong>
                <small>emphasize downstream +x / +y movement</small>
              </div>
              <div className={styles.directionAxes} aria-hidden="true">
                <div><span>v<sub>x</sub></span><i>→</i></div>
                <div><span>v<sub>y</sub></span><i>↓</i></div>
              </div>
              <div className={styles.directionWrite}>h ← h + αv</div>
            </div>
          </div>

        </div>
      </section>

      <section className={styles.interpretabilitySection} id="intervention" aria-labelledby="interpretability-title">
        <div className={styles.sectionHeading}>
          <div>
            <h2 id="interpretability-title">These variables are causal. Write to them and the hallucination changes.</h2>
          </div>
          <EditableMarkdown blockId="causal-intervention" markdown={copy["causal-intervention"]} editable={editable} className={styles.sectionCopy} />
        </div>
        <PixelInterpretabilityViewer />
      </section>

      <section className={styles.liveSection} id="game-surgery" aria-labelledby="live-title">
        <div className={styles.sectionHeading}>
          <div>
            <h2 id="live-title">This is a video game. You play it through brain surgery.</h2>
          </div>
          <EditableMarkdown blockId="brain-surgery" markdown={copy["brain-surgery"]} editable={editable} className={styles.sectionCopy} />
        </div>
        <LiveWorldModel />
      </section>

      <section className={styles.emergenceSection} id="representations" aria-labelledby="emergence-title">
        <div className={styles.sectionHeading}>
          <div>
            <h2 id="emergence-title">A linear probe shows where motion becomes readable.</h2>
          </div>
          <EditableMarkdown blockId="representation-depth" markdown={copy["representation-depth"]} editable={editable} className={styles.sectionCopy} />
        </div>
        <PhysicsEmergenceViewer />
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
