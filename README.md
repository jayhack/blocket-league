# Blocket League

**The money shot: a transformer trained only to predict pixels develops a causal representation of motion. We recover a velocity direction from its hidden activations, write that direction back into the frozen model, and turn its video hallucination into a playable game.**

Blocket League is a small world-model interpretability experiment. The transformer watches raw 64×64 frames from a passive two-disc physics simulation and predicts the next image. It receives no actions, coordinates, or simulator state. Its learned weights nevertheless produce hidden-state directions that predict downstream physical effects. Those directions are not merely readable: intervening on them changes the generated trajectory.

The interactive lab follows the whole argument from observation to intervention. It shows the model rolling forward, recovers physical variables such as velocity from its activations, tests a reusable causal direction across unseen worlds, and finally maps that activation edit to the keyboard. The resulting “game” is still an autoregressive hallucination; the controls operate by performing live surgery on the model's internal physics.

## A physics emergence zone

Following [*Interpreting Physics in Video World Models*](https://arxiv.org/abs/2602.07050), we tested whether motion direction is represented as a circular population code rather than as two privileged x/y variables. It is—and the result separates three ideas that are easy to conflate:

- **Readable:** direction is absent from the patch embedding, jumps to 0.63 held-out R² after the first transformer block, and peaks at 0.88 at block 5.
- **Organized:** a clean ring appears in blocks 4–5. At block 5, 305 of 768 MLP units are direction-tuned; their population trajectory has winding number 1 and circular-distance correlation 0.976.
- **Controllable:** direction is distributed—removing it requires 74 orthogonal residual dimensions, versus 50 for speed—but downstream Jacobian averaging recovers two nearly orthogonal activation writes. Interpolating those writes around the circle steers 12-frame generated trajectories with 5.1° mean angular error across 256 unseen rollouts.

All probe targets come from rendered-pixel motion, not simulator state. The result is from one six-block checkpoint, so the emergence zone is coarse and should be replicated across independently trained seeds before treating its exact depth as universal.

## Run the lab

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The checked-in ONNX model runs locally in the browser with WebGPU when available and falls back to WASM.

## Train and test

The Python package contains the simulator, pixel transformer, Modal training entry point, probes, intervention analysis, and browser exporters.

```bash
uv sync --extra training
uv run python -m unittest discover tests
uv run modal run blocket_league/modal_app.py --stage pixel-direct --preset tiny
uv run --extra training python -m blocket_league.hallucination_assets checkpoint.pt \
  public/blocket-league/hallucinations
uv run --extra training python -m blocket_league.ring_probe checkpoint.pt ring-probe.json \
  --causal-manifest public/blocket-league/interpretability/passive-pixel-manifest.json
```

Training clips are generated deterministically, so no dataset download is required. Large checkpoints and generated outputs are intentionally ignored; the smaller browser graph and exhibit assets live in `public/blocket-league/`.

## Repository map

- `blocket_league/` — simulator, models, training, probes, and export tools
- `components/blocket-league/` — interactive React lab
- `public/blocket-league/` — browser model and generated exhibits
- `tests/` — simulator and model tests

Licensed under the MIT License.
