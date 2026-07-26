export const BLOCKET_LEAGUE_COPY_IDS = [
  "hero-intro",
  "hero-sources",
  "play-intro",
  "play-takeaway",
  "dataset",
  "model",
  "model-results",
  "jacobian-lens",
  "causal-intervention",
  "brain-surgery",
  "representation-depth",
] as const;

export type BlocketLeagueCopyId = (typeof BLOCKET_LEAGUE_COPY_IDS)[number];
export type BlocketLeagueCopy = Record<BlocketLeagueCopyId, string>;

export function isBlocketLeagueCopyId(value: string): value is BlocketLeagueCopyId {
  return BLOCKET_LEAGUE_COPY_IDS.includes(value as BlocketLeagueCopyId);
}
