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

import styles from "./blocket-league-lab.module.css";
import { HallucinationFilmstrip } from "./hallucination-filmstrip";
import { LiveWorldModel } from "./live-world-model";
import { PhysicsEmergenceViewer } from "./physics-emergence-viewer";
import { PixelInterpretabilityViewer } from "./pixel-interpretability-viewer";

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

function drawDisc(
  context: CanvasRenderingContext2D,
  position: Vec2,
  radius: number,
  fill: string,
  core: string,
) {
  context.shadowColor = "rgba(0, 0, 0, 0.38)";
  context.shadowBlur = 0.022;
  context.shadowOffsetY = 0.012;
  context.fillStyle = fill;
  context.beginPath();
  context.arc(position.x, position.y, radius, 0, Math.PI * 2);
  context.fill();
  context.shadowColor = "transparent";
  context.fillStyle = core;
  context.beginPath();
  context.arc(position.x, position.y, radius * 0.31, 0, Math.PI * 2);
  context.fill();
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
  context.fillStyle = "#070b10";
  context.fillRect(0, 0, side, side);

  context.save();
  context.scale(side, side);
  const wall = WORLD.wall;
  context.fillStyle = "#0c2022";
  context.fillRect(wall, wall, 1 - wall * 2, 1 - wall * 2);
  const wash = context.createRadialGradient(0.5, 0.5, 0.05, 0.5, 0.5, 0.7);
  wash.addColorStop(0, "rgba(29, 78, 77, 0.16)");
  wash.addColorStop(1, "rgba(3, 9, 12, 0.08)");
  context.fillStyle = wash;
  context.fillRect(wall, wall, 1 - wall * 2, 1 - wall * 2);

  context.strokeStyle = "rgba(78, 132, 126, 0.32)";
  context.lineWidth = 0.006;
  context.beginPath();
  context.moveTo(0.5, wall);
  context.lineTo(0.5, 1 - wall);
  context.stroke();
  context.beginPath();
  context.arc(0.5, 0.5, 0.14, 0, Math.PI * 2);
  context.stroke();

  context.fillStyle = "rgba(238, 181, 62, 0.18)";
  context.fillRect(1 - wall - 0.045, WORLD.goalLow, 0.045, WORLD.goalHigh - WORLD.goalLow);
  context.strokeStyle = "#526769";
  context.lineWidth = 0.013;
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
  context.strokeStyle = "#eeb53e";
  context.lineWidth = 0.012;
  context.beginPath();
  context.moveTo(1 - wall, WORLD.goalLow);
  context.lineTo(1 - wall, WORLD.goalHigh);
  context.stroke();

  drawDisc(context, state.playerPosition, WORLD.playerRadius, "#32d5ad", "#0b3934");
  drawDisc(context, state.puckPosition, WORLD.puckRadius, "#eff2e9", "#96a199");

  if (state.resetTimer > 0) {
    context.fillStyle = "rgba(7, 11, 16, 0.72)";
    context.fillRect(0.3, 0.43, 0.4, 0.14);
    context.fillStyle = "#eeb53e";
    context.font = "500 0.055px system-ui, sans-serif";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText("GOAL", 0.5, 0.5);
  }
  context.restore();
}

export function BlocketLeagueLab() {
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
        <h1>A pixel transformer learns physics. We turn its hidden state into a game.</h1>
        <p className={styles.heroCopy}>
          In this lab, we construct a pixel-to-pixel transformer that models a toy world with
          collisions, goals, and resets. Using methods inspired by <a href="https://arxiv.org/abs/2602.07050" target="_blank" rel="noreferrer"><em>Interpreting Physics in Video World Models</em></a> and Anthropic&apos;s <a href="https://www.anthropic.com/research/global-workspace" target="_blank" rel="noreferrer">J-space work</a>, we show that its hidden activations contain a compact model of high-level physical phenomena—including velocity and collisions—and causal directions for motion.
        </p>
        <p className={styles.heroCopy}>
          We then manipulate those directions directly and change the model&apos;s predicted future.
          The result is a steerable video game built from pure video observations: every frame is a
          transformer hallucination, and your keyboard edits the physics inside it.
        </p>
        <div className={styles.heroGame}>
          <LiveWorldModel />
        </div>
      </section>

      <section className={styles.labSection} id="world" aria-labelledby="world-title">
        <div className={styles.sectionHeading}>
          <div>
            <h2 id="world-title">Train a video model on simple physics simulations.</h2>
          </div>
          <p>
            Every clip begins with randomized momentum. After that, the two circles simply coast,
            collide, bounce, and score. Goal-centered clips teach the pause and a moving kickoff
            selected by the visible score. The training set contains pixels—not actions or state vectors.
          </p>
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
              aria-label="A playable square arena with a teal player disc, a white puck, and a gold goal on the right."
            />
          </div>
        </div>
      </section>

      <section className={styles.modelSection} aria-labelledby="model-title">
        <div className={styles.sectionHeading}>
          <div>
            <h2 id="model-title">Pixel transformers work surprisingly well.</h2>
          </div>
          <p>
            The transformer is trained like a visual next-token predictor: previous rendered frames
            go in, and it predicts the pixels of the next. There are no actions or coordinates in the
            input. During a rollout, each predicted image becomes part of the history, so position,
            motion, collisions, and resets must survive through the model&apos;s own pixels.
          </p>
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
      </section>

      <section className={styles.trajectorySection} aria-labelledby="prediction-title">
        <div className={styles.sectionHeading}>
          <div>
            <h2 id="prediction-title">It predicts collisions, bounces, and goals.</h2>
          </div>
          <p>
            Each held-out film begins with 12 rendered input frames. The next 36 are generated one
            frame at a time, with every prediction fed back into the model. Across contact, wall,
            and goal-reset scenarios, the futures remain structured and event-specific rather than
            collapsing into random pixels.
          </p>
        </div>
        <HallucinationFilmstrip />
      </section>

      <section className={styles.lensSection} aria-labelledby="lens-title">
        <div className={styles.sectionHeading}>
          <div>
            <h2 id="lens-title">Hidden-state activation directions predict downstream physics.</h2>
          </div>
          <p>
            These directions are vectors in the transformer&apos;s 192-dimensional hidden state—not
            directions on the game board. Linear probes recover position at 0.99 R² and velocity at
            0.90 R² on unseen clips. We then adapt the <a href="https://transformer-circuits.pub/2026/workspace/index.html#the-jacobian-lens" target="_blank" rel="noreferrer">J-space paper&apos;s Jacobian lens</a> to extract
            reusable activation-space vectors by averaging how each hidden activation changes the next
            generated frame. Collision outcomes appear in the video, but we have not yet isolated a
            single collision vector.
          </p>
        </div>

        <div
          className={styles.lensDiagram}
          role="img"
          aria-label="Across 512 rendered contexts, select the block five activation at the green disc's spatial token, run the frozen downstream model to the next rendered frame, backpropagate the green centroid's x and y coordinates, and average those gradients into global x and y steering directions."
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
                <small>p = green spatial token</small>
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
              <div className={styles.backwardRail}><ArrowRight aria-hidden="true" /><span>BACKPROP ∂ŷ / ∂h</span></div>
            </div>

            <div className={`${styles.lensStage} ${styles.lensReadout}`}>
              <div className={styles.diagramStageHeader}>
                <span>MEASURE</span>
                <strong>Next-frame centroid</strong>
                <small>soft readout from green logits</small>
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
                <strong>Global directions</strong>
                <small>reusable across unseen rollouts</small>
              </div>
              <div className={styles.directionAxes} aria-hidden="true">
                <div><span>v<sub>x</sub></span><i>→</i></div>
                <div><span>v<sub>y</sub></span><i>↓</i></div>
              </div>
              <div className={styles.directionWrite}>h ← h + αv</div>
            </div>
          </div>

          <div className={styles.lensEquation}>
            <span>PHYSICS J-LENS</span>
            <strong>v<sub>x</sub> = normalize [ 1/K · Σ<sub>i</sub> ∂x̂<sub>i,t+1</sub> / ∂h<sub>i,ℓ,p</sub> ]</strong>
            <small>Repeat for y. Average across contexts to remove rollout-specific accidents and retain a reusable downstream effect.</small>
          </div>

          <div className={styles.lensComparison}>
            <div>
              <span>ANTHROPIC J-LENS</span>
              <strong>activation → future final residuals → vocabulary logits</strong>
              <small>Average across prompts, source positions, and future positions.</small>
            </div>
            <div>
              <span>OUR MOTION LENS</span>
              <strong>player token → next-frame pixels → rendered centroid</strong>
              <small>Average across worlds; then write the resulting x/y direction back into the same token.</small>
            </div>
          </div>
        </div>
      </section>

      <section className={styles.interpretabilitySection} aria-labelledby="interpretability-title">
        <div className={styles.sectionHeading}>
          <div>
            <h2 id="interpretability-title">These variables are causal. Write to them and the hallucination changes.</h2>
          </div>
          <p>
            Write the recovered +x direction for four frames, then stop. By frame 12, the green circle
            is 3.51 pixels farther right on average across 256 unseen worlds, and 85.9% move in the
            intended direction. A random activation direction has almost no effect.
          </p>
        </div>
        <PixelInterpretabilityViewer />
      </section>

      <section className={styles.liveSection} aria-labelledby="live-title">
        <div className={styles.sectionHeading}>
          <div>
            <h2 id="live-title">This is a video game. You play it through brain surgery.</h2>
          </div>
          <p>
            The model never trained on keystrokes. Here, WASD is mapped directly onto the recovered
            ±x and ±y activation directions for the green circle. The white puck moves only when the
            hallucinated physics says it should.
          </p>
        </div>
        <LiveWorldModel />
      </section>

      <section className={styles.emergenceSection} aria-labelledby="emergence-title">
        <div className={styles.sectionHeading}>
          <div>
            <h2 id="emergence-title">The learned direction code has circular geometry.</h2>
          </div>
          <p>
            Finally, following <a href="https://arxiv.org/abs/2602.07050" target="_blank" rel="noreferrer">Interpreting Physics in Video World Models</a>, we ask how motion is arranged inside the network rather than assuming x/y velocity is a factorized state variable. Direction becomes linearly readable after the first block, while a clean circular population code organizes at blocks 4–5. Removing 74 orthogonal residual dimensions is required to collapse direction decoding, compared with 50 for speed.
          </p>
        </div>
        <PhysicsEmergenceViewer />
      </section>

      <footer className={styles.footer}>
        <span>BLOCKET LEAGUE</span>
      </footer>
    </main>
  );
}
