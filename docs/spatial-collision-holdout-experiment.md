# Spatial collision holdout experiment

This addendum tests whether the passive pixel transformer learns collision
dynamics that transfer across arena location.

## Design

The treatment is a 3.67M-parameter `tiny` model trained from scratch. Its cache
rejects an entire 24-frame world whenever a player–puck impact occurs in the
upper-right arena quadrant. Collision location is defined as the midpoint
between the two disc centers on the impact frame. Whole-clip rejection prevents
a collision later in a sampled world from leaking into training.

Free motion, wall bounces, and non-collision frames in the upper-right remain
available. The removed support is therefore the local collision transition, not
the arena region itself. In a 2,000-world audit, 15.95% of ordinary worlds
contained an upper-right collision and were eligible for rejection; collision
worlds were otherwise approximately balanced across quadrants.

The all-angle 30,000-step direction control is reused as the matched
all-location control because its seed, architecture, cache size, optimizer,
history length, corruption recipe, and rollout curriculum are identical. Both
arms use 16,384 cached worlds, batch size 16, eight observed frames, and 12
predicted frames.

Evaluation searches deterministic unseen worlds for a player–puck impact whose
contact midpoint is in each requested quadrant. Each model is evaluated on the
same 32 worlds per quadrant. The primary metric is mean rendered-pixel puck
position error over the 12-frame autoregressive rollout; simulator state is used
only as the evaluation target.

## Result

The upper-right-held-out model reaches **1.257 px** puck error on upper-right
collisions. The matched model trained on collisions everywhere reaches **1.247
px** on the identical worlds, leaving only a **0.8% holdout penalty**.

The held-out model averages **1.168 px** across the three collision quadrants it
did see. Its unseen-quadrant error is 7.5% higher than that average, within the
ordinary variation between locations and much smaller than the 71% penalty in
the direction-support experiment.

The evidence therefore favors a spatially shared collision rule rather than
location-specific memorization. The 0.8% difference should not be interpreted
as statistically resolved from one training seed and 32 evaluation worlds per
quadrant; repeated training seeds would be the appropriate next test.

Training, cache construction, and evaluation took 1,152 seconds on one H100 and
cost approximately $1.72.

## Commands

Train the upper-right collision holdout:

```bash
uvx --from modal modal run blocket_league/modal_app.py \
  --stage pixel-direct --preset tiny --steps 30000 --batch-size 16 \
  --learning-rate 0.0003 --seed 73 --workers 8 --eval-samples 64 \
  --pixel-history-frames 8 --patch-size 4 --latent-cache-samples 16384 \
  --latent-rollout-frames 12 --goal-centered-fraction 0 \
  --pixel-corruption-rate 0.06 --pixel-entity-corruption-fraction 0.20 \
  --pixel-model-rollin-fraction 0.35 --pixel-model-rollin-start-step 3000 \
  --pixel-model-rollin-ramp-steps 7000 --late-frame-weight 2 --gpu H100 \
  --excluded-collision-quadrant upper-right \
  --collision-quadrant-eval-samples 32 \
  --output-dir blocket_league/outputs/collision-holdout-upper-right-tiny-30000
```

Evaluate the existing matched control on the same quadrant suite:

```bash
uv run python -m blocket_league.evaluate_collision_quadrants \
  blocket_league/outputs/direction-control-tiny-30000/checkpoint.pt \
  blocket_league/outputs/direction-control-tiny-30000/summary.json \
  blocket_league/outputs/direction-control-tiny-30000/collision-quadrants.json \
  --samples-per-quadrant 32
```

Build the article comparison and registered sample viewer:

```bash
uv run python -m blocket_league.compare_collision_holdout \
  blocket_league/outputs/direction-control-tiny-30000/collision-quadrants.json \
  blocket_league/outputs/collision-holdout-upper-right-tiny-30000/summary.json \
  public/blocket-league/interpretability/collision-holdout.json

uv run python -m blocket_league.experiment_assets \
  blocket_league/outputs/collision-holdout-upper-right-tiny-30000/checkpoint.pt \
  public/experiments/collision-holdout-upper-right \
  --rollout-frames 36 --scenario-set collision-quadrants \
  --lane-label "Upper-right holdout prediction"
```
