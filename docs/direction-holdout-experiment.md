# Direction holdout experiment

This addendum tests whether the passive pixel transformer interpolates a
motion rule across directions that are absent from its training support.

## Design

Two 3.67M-parameter `tiny` models are trained from scratch with identical
hyperparameters and seed. The control sees the usual collision-rich passive
cache. The treatment cache rejects an entire 24-frame world whenever the
puck has a nontrivial velocity within 30 degrees of due east in any frame.
This clip-level rejection matters: filtering only the initial state would let
the held-out direction leak back into training after a wall or disc collision.

Goal-centered clips are disabled in both arms because their forced goalward
velocity would violate the support holdout. Each arm uses 16,384 cached worlds,
30,000 optimizer steps, batch size 16, eight history frames, and the same
corruption and model-roll-in recipe.

Evaluation forces the puck's initial direction into eight 30-degree bins
centered every 45 degrees. It uses 32 deterministic held-out worlds per bin and
measures rendered-pixel puck position error over a 12-frame autoregressive
rollout. The simulator state is used only as the evaluation target.

## Result

The direction-held-out model reaches **0.95 px** mean puck-position error in
the unseen due-east bin. Its average across the seven direction bins present in
training is **0.99 px**, so the missing direction is not a catastrophic or
even locally obvious hole in its learned transition rule.

The matched all-angle control reaches **0.56 px** on the same due-east worlds.
The holdout therefore carries a real **71% accuracy penalty** despite landing
within the normal range of the held-out model's seen-direction errors. The
cleanest interpretation is that the transformer learns a direction-general
rule and interpolates it into missing support, while direct experience still
substantially sharpens that rule.

Each training run used one H100 and completed cache construction, 30,000
optimizer steps, and evaluation in about 19 minutes.

## Commands

Control:

```bash
uvx --from modal modal run blocket_league/modal_app.py \
  --stage pixel-direct --preset tiny --steps 30000 --batch-size 16 \
  --learning-rate 0.0003 --seed 73 --workers 8 --eval-samples 64 \
  --pixel-history-frames 8 --patch-size 4 --latent-cache-samples 16384 \
  --latent-rollout-frames 12 --goal-centered-fraction 0 \
  --pixel-corruption-rate 0.06 --pixel-entity-corruption-fraction 0.20 \
  --pixel-model-rollin-fraction 0.35 --pixel-model-rollin-start-step 3000 \
  --pixel-model-rollin-ramp-steps 7000 --late-frame-weight 2 --gpu H100 \
  --direction-eval-samples-per-bin 32 \
  --output-dir blocket_league/outputs/direction-control-tiny-30000
```

Direction holdout:

```bash
uvx --from modal modal run blocket_league/modal_app.py \
  --stage pixel-direct --preset tiny --steps 30000 --batch-size 16 \
  --learning-rate 0.0003 --seed 73 --workers 8 --eval-samples 64 \
  --pixel-history-frames 8 --patch-size 4 --latent-cache-samples 16384 \
  --latent-rollout-frames 12 --goal-centered-fraction 0 \
  --pixel-corruption-rate 0.06 --pixel-entity-corruption-fraction 0.20 \
  --pixel-model-rollin-fraction 0.35 --pixel-model-rollin-start-step 3000 \
  --pixel-model-rollin-ramp-steps 7000 --late-frame-weight 2 --gpu H100 \
  --direction-eval-samples-per-bin 32 \
  --excluded-puck-angle-center-degrees 0 \
  --excluded-puck-angle-width-degrees 60 \
  --output-dir blocket_league/outputs/direction-holdout-east60-tiny-30000
```

Combine the two summaries into the browser manifest:

```bash
uv run python -m blocket_league.compare_direction_holdout \
  blocket_league/outputs/direction-control-tiny-30000/summary.json \
  blocket_league/outputs/direction-holdout-east60-tiny-30000/summary.json \
  public/blocket-league/interpretability/direction-holdout.json
```
