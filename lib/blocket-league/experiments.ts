export type ExperimentMetric = {
  label: string;
  value: string;
  detail: string;
};

export type Experiment = {
  slug: string;
  title: string;
  eyebrow: string;
  description: string;
  verdict: string;
  createdAt: string;
  manifestUrl: string;
  checkpoint: {
    id: string;
    artifact: string;
    parameters: number;
    bytes: number;
    step: number;
  };
  training: {
    preset: string;
    hardware: string;
    seconds: number;
  };
  metrics: ExperimentMetric[];
};

export const experiments = [
  {
    slug: "nano-1p5mb",
    title: "The 1.5 MB capacity cliff",
    eyebrow: "PASSIVE PIXEL · SCALING RUN 001",
    description:
      "A width-72, four-block transformer trained on the same passive Blocket League curriculum as the deployed model, with 10.27× fewer parameters.",
    verdict:
      "The model learns coarse motion and the static arena, but its entities fragment under autoregressive rollout. Tenfold compression crosses the useful capacity threshold for this recipe.",
    createdAt: "2026-07-26",
    manifestUrl: "/experiments/nano-1p5mb/manifest.json",
    checkpoint: {
      id: "passive-pixel-direct-nano-12000",
      artifact:
        "blocket_league/outputs/passive-pixel-direct-nano-12000/checkpoint.pt",
      parameters: 377_136,
      bytes: 1_540_725,
      step: 12_000,
    },
    training: {
      preset: "nano",
      hardware: "NVIDIA H100 80GB HBM3",
      seconds: 257.657,
    },
    metrics: [
      {
        label: "12-frame error",
        value: "4.98 px",
        detail: "deployed model 0.93 px",
      },
      {
        label: "64-frame error",
        value: "19.10 px",
        detail: "deployed model 6.53 px",
      },
      {
        label: "FP32 checkpoint",
        value: "1.54 MB",
        detail: "14.88 MB deployed graph",
      },
      {
        label: "Training time",
        value: "4m 18s",
        detail: "12,000 H100 steps",
      },
    ],
  },
] as const satisfies readonly Experiment[];

export function getExperiment(slug: string): Experiment | undefined {
  return experiments.find((experiment) => experiment.slug === slug);
}

