# Blocket League

Blocket League is a small world-model interpretability experiment. A transformer watches raw 64×64 pixel frames from a passive two-disc physics simulation and predicts the next image. It receives no actions or simulator state.

The interactive lab shows the model rolling forward, probes hidden states for physical quantities such as velocity, and demonstrates a causal intervention: keyboard input writes a recovered velocity direction into the frozen model's activations instead of entering through an action channel.

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
