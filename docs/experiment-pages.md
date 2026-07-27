# Registered experiment pages

Model checkpoints can be published as durable, statically exported sample viewers at
`/{experiment}`. The experiment registry lives in
`lib/blocket-league/experiments.ts`; each entry binds a URL slug to a checkpoint,
training metadata, headline metrics, and a public sample manifest.

## Publishing a run

1. Add the model preset to `blocket_league/pixel_direct_model.py` and train the
   checkpoint.
2. Export held-out rollouts:

   ```bash
   uv run python -m blocket_league.experiment_assets \
     blocket_league/outputs/<run>/checkpoint.pt \
     public/experiments/<slug>
   ```

3. Register `<slug>` and its checkpoint metadata in
   `lib/blocket-league/experiments.ts`.
4. Add the experiment to the scaling section of the main article when it
   contributes to the story.

The dynamic route uses `generateStaticParams`, so every registered slug is included
in the static Next.js export. Checkpoints remain local/remote training artifacts;
only the compact sample atlases and manifest are shipped with the site.
