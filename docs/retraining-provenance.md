# Pixel-model retraining provenance

This note records the July 26, 2026 palette-migration experiment, including
failed checkpoints. It is intentionally more detailed than the lab narrative
so the model choice remains auditable.

## Model currently exported to the browser

The browser export was restored to the known-good recovery checkpoint after the
new palette retrain failed the long-rollout acceptance test. The simulator's
nine semantic class IDs did not change, so the old categorical model remains
valid. Rendered PNG assets and the browser manifest palette were remapped
class-for-class to the blue/orange palette; no model output was color-matched or
approximated.

The exported checkpoint has two training phases:

1. A 30,000-step base run (`passive-pixel-direct-deterministic-reset-30000`)
   with batch size 16.
2. A 12,000-step recovery fine-tune
   (`passive-pixel-direct-recovery-finetune-12000`) with batch size 16.

That is 42,000 optimizer steps/minibatches in the full checkpoint lineage,
672,000 sampled training sequences, and 5,376,000 teacher next-frame targets
(eight targets per sequence). The browser manifest reports `checkpointStep:
12000` because that field records the final fine-tuning phase, not the sum of
the two phases.

The exact recovery command was:

```bash
uvx --from modal modal run blocket_league/modal_app.py \
  --stage pixel-direct \
  --preset tiny \
  --steps 12000 \
  --batch-size 16 \
  --learning-rate 0.0001 \
  --workers 6 \
  --eval-samples 128 \
  --latent-cache-samples 16384 \
  --latent-rollout-frames 64 \
  --late-frame-weight 1.75 \
  --warmup-steps 300 \
  --pixel-entity-corruption-fraction 0.20 \
  --pixel-model-rollin-fraction 0.35 \
  --pixel-model-rollin-start-step 0 \
  --pixel-model-rollin-ramp-steps 2000 \
  --goal-centered-fraction 0.35 \
  --init-checkpoint \
    blocket_league/outputs/passive-pixel-direct-deterministic-reset-30000/checkpoint.pt \
  --gpu H100 \
  --output-dir \
    blocket_league/outputs/passive-pixel-direct-recovery-finetune-12000
```

The base phase ran for 963.900 seconds after a 27.231-second cache build. The
final phase ran for 481.785 seconds after a 25.657-second cache build. Together,
the trainer-reported cache, training, and evaluation time was 1,498.572 seconds
(24 minutes 58.6 seconds) on an NVIDIA H100. This excludes Modal image build,
container startup, and artifact transfer time.

The recovery phase's final minibatch loss was 0.02764 and its 50-step loss EMA
was 0.02475. On 128 held-out worlds, the mean entity-position error through
frame 12 was 0.928 pixels; through the 64-frame rollout it was 6.532 pixels.

## Browser-export acceptance test

The exported ONNX model was tested using the same inputs as the browser:
eight categorical 64×64 history frames, a block-5 residual-stream write routed
to the latest player-centroid patch, and the site's four-frames-on /
four-frames-off steering schedule.

- Baseline: 128 unique frames in a 128-frame rollout; all of the final 32
  frames were unique.
- Player shape: median 56.5 pixels, 54 pixels on the final frame, and a largest
  connected-component fraction of 1.0.
- At frame 32, all four held directions displaced the player from its matched
  no-write baseline in the requested direction: right 1.59 px, left 18.89 px,
  down 10.78 px, and up 16.11 px.
- All four steered 128-frame rollouts continued changing and retained a
  connected player component. After collisions, wall contacts, and goal resets,
  absolute centroid differences from the no-write baseline are no longer a
  meaningful sign test because the two rollouts occupy different world phases.

Continuous writes were also tested. They steer correctly, but sustained
vertical writes grow the player class mass. The pulsed schedule is therefore
the safer browser policy.

## Rejected July 26 palette retrain

The new-palette H100 run was:

```bash
uv run modal run blocket_league/modal_app.py \
  --stage pixel-direct \
  --gpu H100 \
  --preset tiny \
  --steps 30000 \
  --batch-size 24 \
  --learning-rate 0.0002 \
  --seed 59 \
  --workers 8 \
  --log-every 250 \
  --eval-samples 128 \
  --pixel-history-frames 8 \
  --latent-rollout-frames 64 \
  --latent-cache-samples 16384 \
  --goal-centered-fraction 0.35 \
  --pixel-corruption-rate 0.06 \
  --pixel-entity-corruption-fraction 0.25 \
  --pixel-model-rollin-fraction 0.35 \
  --pixel-model-rollin-start-step 3000 \
  --pixel-model-rollin-ramp-steps 7000 \
  --late-frame-weight 2.0 \
  --ema-decay 0.9995 \
  --warmup-steps 500 \
  --output-dir blocket_league/outputs/pixel-galactic-30k
```

Accounting: 30,000 optimizer steps/minibatches, batch size 24, 720,000 sampled
sequences, and 5,760,000 teacher next-frame targets. The cache contained 16,384
worlds × 24 frames = 393,216 rendered frames and 262,144 candidate windows.
Training plus evaluation took 1,676.085 seconds after a 29.720-second cache
build. Final loss was 0.02579 (50-step EMA 0.02867).

Although its 12-frame held-out error was 1.376 pixels and its probes remained
strong, the exported rollout reached an exact fixed point: generated frames
72–128 were identical and both entities had zero motion over the final 32
frames. It therefore failed the browser acceptance gate.

Two corrective smoke runs tested motion-weighted loss, foreground-occupancy
loss, and an explicit four-step autoregressive objective:

- 2,000 steps with 10× changed-pixel weighting avoided freezing but produced
  catastrophic entity growth (only 6.5% valid player mass and 2.1% valid puck
  mass).
- 5,000 steps with 4× entity-change weighting, occupancy weight 0.5, and a
  four-step autoregressive loss also avoided freezing, but mean 12-frame error
  rose to 4.526 pixels and valid mass remained only 52.4% for the player and
  21.0% for the puck.

Those checkpoints and summaries remain under
`blocket_league/outputs/pixel-galactic-*` for diagnosis. They are not exported
to the site.
