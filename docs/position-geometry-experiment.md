# Cartesian position geometry and causal writes

## Question

Does the passive pixel transformer construct a coordinate-like internal state,
or can a probe only recover an entity's absolute patch address? If position is
linearly readable, is the corresponding probe direction also causally writable?

## Fixed-token position probe

Each example contains the model's normal eight rendered context frames. Player
and puck targets are centroids measured from categorical pixels; simulator state
is never used as a label.

We compare three readouts at the input embedding and after every transformer
block:

- **Fixed bottom-right token:** the literal last spatial token after the final
  observed frame. It is identical for every example and uses no entity locator.
- **Spatial mean:** the mean of all 256 final-frame tokens.
- **Entity token:** the token containing the rendered entity centroid. This is a
  deliberately permissive baseline and may expose absolute patch embeddings.

A ridge decoder is fit on 2,048 trajectories and evaluated on 1,024 disjoint
trajectories. A second decoder excludes every fitting example in which its target
entity occupies the upper-right quadrant. Its test set contains only held-out
upper-right examples. Large x and small y values both occur during fitting, but
never in that combination.

At block 5, the fixed-token decoder obtains:

| Target | Ordinary R² | Ordinary RMSE | Upper-right R² | Upper-right RMSE |
|---|---:|---:|---:|---:|
| Player x/y | 0.967 | 1.98 px | 0.668 | 3.45 px |
| Puck x/y | 0.980 | 1.72 px | 0.719 | 3.67 px |

There are 220 player and 365 puck examples in the held-out quadrant test.
Spatial-mean upper-right R² is 0.829 and 0.927, respectively. The fixed token's
input embedding is at chance, and a same-shape random-weight transformer fails
the quadrant test. Spatial attention therefore constructs and broadcasts a
learned, linearly decodable, Cartesian-like representation rather than merely
exposing the selected entity's patch address.

This does not mean that two individual units literally store x and y. The two
coordinates are linear directions through a 192-dimensional distributed state.

## Causal write test

For the puck at block 5, we turn each two-output ridge decoder into the
minimum-norm hidden direction predicted to change x or y while holding the other
coordinate fixed. We normalize that orientation and write it at strengths 1, 4,
8, and 16.

The result is negative: the fixed-token direction has essentially no rendered
effect, and the pooled and puck-token probe directions are weak or inconsistently
signed. Linear decodability by itself is correlational.

We then compute a separate Jacobian direction: the average gradient of the next
rendered puck centroid with respect to the block-5 puck-token activation. At
strength 8 on 128 unseen worlds, a one-frame write produces:

| Write | Immediate intended displacement | Intended-sign worlds |
|---|---:|---:|
| +x | +0.473 px | 89.8% |
| −x | −0.340 px | 85.2% |
| +y | +0.432 px | 88.3% |
| −y | −0.360 px | 84.4% |

Immediate player displacement is only 0.01–0.05 px and the puck remains present
in every example. The write is then stopped. Twelve generated frames later,
signed puck separation from the matched baseline reaches +1.227, −1.023, +1.327,
and −1.045 px. The displacement grows after release, showing that the brief edit
changes the evolving predicted world rather than only recoloring one frame.

The implication is a useful separation:

1. Linear probes reveal a compact physical state: position, velocity, and future
   collisions are readable.
2. Probe weights do not automatically identify causal control vectors.
3. Downstream Jacobians convert a readable variable into a writable handle by
   selecting the part of activation space that the remaining network actually
   uses to render that variable.

## What would establish a collision circuit?

These results do not order the readable coordinate code relative to a separate
"real" representation, and they do not yet show that the collision feature
operates on this particular x/y subspace. A probe direction can be noncausal
because it is redundant, distributed, or simply points off the learned state
manifold.

A stronger mediation assay should use the existing matched collide-versus-miss
pairs, which end at identical positions with matched per-object speeds:

1. Fit a joint coordinate/velocity decoder and a future-collision decoder at the
   same block.
2. Split the collide-minus-miss activation difference into its projection onto
   the decoded physical-state subspace and its orthogonal remainder.
3. Patch the full difference, the physical-state component, and the orthogonal
   component separately from a collision history into its matched miss history.
4. Measure immediate decoded x/y and velocity, then measure whether the generated
   post-contact relative velocity acquires the expected elastic impulse.
5. Repeat across collision normals. A genuine geometric operator should flip the
   closing normal component of relative velocity while preserving the tangential
   component, and the effect should rotate equivariantly with the contact normal.

That result would go beyond "collision is decodable." It would identify a
collision-conditioned operator acting on a coordinate/velocity state.

## Reproduction

```bash
uv run --with torch --with numpy --with pillow \
  python -m blocket_league.position_geometry_probe \
  blocket_league/outputs/passive-pixel-direct-recovery-finetune-12000/checkpoint.pt \
  public/blocket-league/interpretability/position-geometry.json

uv run --with torch --with numpy --with pillow \
  python -m blocket_league.position_write_probe \
  blocket_league/outputs/passive-pixel-direct-recovery-finetune-12000/checkpoint.pt \
  public/blocket-league/interpretability/position-write.json

uv run --with torch --with numpy --with pillow \
  python -m blocket_league.position_rollout_assets \
  blocket_league/outputs/passive-pixel-direct-recovery-finetune-12000/checkpoint.pt \
  public/blocket-league/position-rollouts
```

Both assays run against a frozen checkpoint. No training or cloud accelerator is
required; the reported runs were executed locally on Apple MPS. The final
command fits the same fixed-token decoder and recomputes its x/y estimate after
every autoregressive frame, producing the article's overlaid rollout assets.
