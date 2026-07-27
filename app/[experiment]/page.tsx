import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";

import { ExperimentTabs } from "@/components/blocket-league/experiment-tabs";
import styles from "@/components/blocket-league/experiment.module.css";
import {
  experiments,
  getExperiment,
} from "@/lib/blocket-league/experiments";

type ExperimentPageProps = {
  params: Promise<{ experiment: string }>;
};

export const dynamicParams = false;

export function generateStaticParams() {
  return experiments.map((experiment) => ({
    experiment: experiment.slug,
  }));
}

export async function generateMetadata({
  params,
}: ExperimentPageProps): Promise<Metadata> {
  const { experiment: slug } = await params;
  const experiment = getExperiment(slug);
  if (!experiment) notFound();
  return {
    title: `${experiment.title} — Blocket League Experiment`,
    description: experiment.description,
  };
}

export default async function ExperimentPage({
  params,
}: ExperimentPageProps) {
  const { experiment: slug } = await params;
  const experiment = getExperiment(slug);
  if (!experiment) notFound();
  const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

  return (
    <main className={styles.page}>
      <div className={styles.shell}>
        <nav className={styles.nav}>
          <Link href="/">← Blocket League lab</Link>
          <span>EXPERIMENT / {experiment.slug.toUpperCase()}</span>
        </nav>

        <header className={styles.hero}>
          <h1>{experiment.title}</h1>
          <p className={styles.heroCopy}>{experiment.description}</p>
        </header>

        <ExperimentTabs
          manifestUrl={`${basePath}${experiment.manifestUrl}`}
          lossUrl={
            "lossUrl" in experiment && experiment.lossUrl
              ? `${basePath}${experiment.lossUrl}`
              : undefined
          }
          metrics={experiment.metrics}
          verdict={experiment.verdict}
        />

        <footer className={styles.provenance}>
          <div>
            <strong>Registered checkpoint</strong>
            <code>{experiment.checkpoint.id}</code>
            <br />
            <code>{experiment.checkpoint.artifact}</code>
          </div>
          <div>
            <strong>Run</strong>
            {experiment.training.preset} preset ·{" "}
            {experiment.checkpoint.parameters.toLocaleString()} parameters ·{" "}
            {experiment.training.hardware} ·{" "}
            {experiment.training.seconds.toFixed(1)} seconds
          </div>
        </footer>
      </div>
    </main>
  );
}
