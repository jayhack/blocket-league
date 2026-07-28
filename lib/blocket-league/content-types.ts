export const BLOCKET_LEAGUE_COPY_IDS = [
  "play-title",
  "hero-intro",
  "hero-sources",
  "play-intro",
  "play-takeaway",
  "dataset-title",
  "dataset",
  "model-title",
  "model",
  "model-results",
  "linear-position-title",
  "linear-position-intro",
  "generalization-title",
  "generalization-intro",
  "jacobian-title",
  "jacobian-lens",
  "causal-title",
  "causal-intervention",
  "brain-surgery-title",
  "brain-surgery",
  "representation-title",
  "representation-depth",
  "model-scale-title",
  "direction-holdout-title",
  "collision-holdout-title",
  "position-geometry-title",
  "collision-representation-title",
  "collision-representation",
  "experiment-index-title",
] as const;

export type BlocketLeagueCopyId = (typeof BLOCKET_LEAGUE_COPY_IDS)[number];
export type BlocketLeagueCopy = Record<BlocketLeagueCopyId, string>;

export function isBlocketLeagueCopyId(value: string): value is BlocketLeagueCopyId {
  return BLOCKET_LEAGUE_COPY_IDS.includes(value as BlocketLeagueCopyId);
}
