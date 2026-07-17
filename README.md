# Blocket League

**The money shot: a transformer trained only to predict pixels develops a causal representation of motion. We recover a velocity direction from its hidden activations, write that direction back into the frozen model, and turn its video hallucination into a playable game.**

Blocket League is a small world-model interpretability experiment. The transformer watches raw 64×64 frames from a passive two-disc physics simulation and predicts the next image. It receives no actions, coordinates, or simulator state. Its learned weights nevertheless produce hidden-state directions that predict downstream physical effects. Those directions are not merely readable: intervening on them changes the generated trajectory.

The interactive lab follows the whole argument from observation to intervention. It shows the model rolling forward, recovers physical variables such as velocity from its activations, tests a reusable causal direction across unseen worlds, and finally maps that activation edit to the keyboard. The resulting “game” is still an autoregressive hallucination; the controls operate by performing live surgery on the model's internal physics.

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
```

Training clips are generated deterministically, so no dataset download is required. Large checkpoints and generated outputs are intentionally ignored; the smaller browser graph and exhibit assets live in `public/blocket-league/`.

## Repository map

- `blocket_league/` — simulator, models, training, probes, and export tools
- `components/blocket-league/` — interactive React lab
- `public/blocket-league/` — browser model and generated exhibits
- `tests/` — simulator and model tests

Licensed under the MIT License.
