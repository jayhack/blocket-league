export type ExperimentMetric = {
  label: string;
  value: string;
  detail: string;
};

export type Experiment = {
  slug: string;
  title: string;
  description: string;
  verdict: string;
  createdAt: string;
  manifestUrl: string;
  lossUrl?: string;
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
  {
    slug: "direction-control-tiny-30k",
    title: "The all-angle control",
    description:
      "A matched 3.67M-parameter control trained from scratch on all puck-motion directions, with the same seed, cache size, optimizer schedule, and recovery curriculum as the direction-holdout model.",
    verdict:
      "This is the reference arm for the direction-support experiment. It reaches 0.56 px puck error on the due-east evaluation bin and provides the counterfactual for measuring the cost of removing that direction from training.",
    createdAt: "2026-07-26",
    manifestUrl: "/experiments/direction-control-tiny-30k/manifest.json",
    lossUrl: "/experiments/direction-control-tiny-30k/loss.json",
    checkpoint: {
      id: "direction-control-tiny-30000",
      artifact:
        "blocket_league/outputs/direction-control-tiny-30000/checkpoint.pt",
      parameters: 3_667_992,
      bytes: 14_720_792,
      step: 30_000,
    },
    training: {
      preset: "tiny",
      hardware: "NVIDIA H100 80GB HBM3",
      seconds: 1_151.101,
    },
    metrics: [
      {
        label: "Due-east error",
        value: "0.56 px",
        detail: "32 held-out evaluation worlds",
      },
      {
        label: "12-frame error",
        value: "0.80 px",
        detail: "mean across both entities",
      },
      {
        label: "Training cost",
        value: "$1.72",
        detail: "H100 + CPU + memory",
      },
      {
        label: "Training time",
        value: "19m 11s",
        detail: "30,000 H100 steps",
      },
    ],
  },
  {
    slug: "direction-holdout-east-60",
    title: "A 60° hole in motion",
    description:
      "A 3.67M-parameter transformer trained from scratch after rejecting every 24-frame world in which the puck ever moves within ±30° of due east.",
    verdict:
      "The model interpolates into the missing support: due-east error is 0.95 px, nearly identical to its 0.99 px average on seen directions. But the matched control reaches 0.56 px, leaving a real 71% holdout penalty.",
    createdAt: "2026-07-26",
    manifestUrl: "/experiments/direction-holdout-east-60/manifest.json",
    checkpoint: {
      id: "direction-holdout-east60-tiny-30000",
      artifact:
        "blocket_league/outputs/direction-holdout-east60-tiny-30000/checkpoint.pt",
      parameters: 3_667_992,
      bytes: 14_720_792,
      step: 30_000,
    },
    training: {
      preset: "tiny",
      hardware: "NVIDIA H100 80GB HBM3",
      seconds: 1_151.233,
    },
    metrics: [
      {
        label: "Unseen east",
        value: "0.95 px",
        detail: "inside the excluded wedge",
      },
      {
        label: "Seen directions",
        value: "0.99 px",
        detail: "seven-bin average",
      },
      {
        label: "Holdout penalty",
        value: "+71%",
        detail: "versus matched control",
      },
      {
        label: "Training cost",
        value: "$1.72",
        detail: "H100 + CPU + memory",
      },
    ],
  },
  {
    slug: "collision-holdout-upper-right",
    title: "Collision physics without the upper-right",
    description:
      "A 3.67M-parameter transformer trained after rejecting every 24-frame world containing a player–puck collision in the upper-right arena quadrant.",
    verdict:
      "Collision physics transfers almost perfectly across space. The held-out model reaches 1.257 px puck error on upper-right impacts, versus 1.247 px for the matched all-location control: a difference of just 0.8%.",
    createdAt: "2026-07-27",
    manifestUrl: "/experiments/collision-holdout-upper-right/manifest.json",
    checkpoint: {
      id: "collision-holdout-upper-right-tiny-30000",
      artifact:
        "blocket_league/outputs/collision-holdout-upper-right-tiny-30000/checkpoint.pt",
      parameters: 3_667_992,
      bytes: 14_720_920,
      step: 30_000,
    },
    training: {
      preset: "tiny",
      hardware: "NVIDIA H100 80GB HBM3",
      seconds: 1_152.213,
    },
    metrics: [
      {
        label: "Unseen quadrant",
        value: "1.257 px",
        detail: "upper-right collisions",
      },
      {
        label: "Matched control",
        value: "1.247 px",
        detail: "same upper-right worlds",
      },
      {
        label: "Holdout penalty",
        value: "+0.8%",
        detail: "nearly complete transfer",
      },
      {
        label: "Training cost",
        value: "$1.72",
        detail: "H100 + CPU + memory",
      },
    ],
  },
] as const satisfies readonly Experiment[];

export function getExperiment(slug: string): Experiment | undefined {
  return experiments.find((experiment) => experiment.slug === slug);
}
