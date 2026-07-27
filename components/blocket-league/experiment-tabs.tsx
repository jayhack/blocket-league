"use client";

import { useId, useState } from "react";

import { HallucinationFilmstrip } from "./hallucination-filmstrip";
import styles from "./experiment.module.css";
import { LossViewer } from "./loss-viewer";

type ExperimentTab = "samples" | "results" | "loss";

type ExperimentTabsProps = {
  manifestUrl: string;
  lossUrl?: string;
  metrics: readonly {
    label: string;
    value: string;
    detail: string;
  }[];
  verdict: string;
};

export function ExperimentTabs({
  manifestUrl,
  lossUrl,
  metrics,
  verdict,
}: ExperimentTabsProps) {
  const id = useId().replaceAll(":", "");
  const [activeTab, setActiveTab] = useState<ExperimentTab>("samples");

  const sampleTabId = `${id}-samples-tab`;
  const samplePanelId = `${id}-samples-panel`;
  const resultsTabId = `${id}-results-tab`;
  const resultsPanelId = `${id}-results-panel`;
  const lossTabId = `${id}-loss-tab`;
  const lossPanelId = `${id}-loss-panel`;

  return (
    <>
      <div className={styles.experimentTabs} role="tablist" aria-label="Experiment view">
        <button
          id={sampleTabId}
          type="button"
          role="tab"
          aria-controls={samplePanelId}
          aria-selected={activeTab === "samples"}
          className={activeTab === "samples" ? styles.experimentTabActive : undefined}
          onClick={() => setActiveTab("samples")}
        >
          Samples
        </button>
        <button
          id={resultsTabId}
          type="button"
          role="tab"
          aria-controls={resultsPanelId}
          aria-selected={activeTab === "results"}
          className={activeTab === "results" ? styles.experimentTabActive : undefined}
          onClick={() => setActiveTab("results")}
        >
          Results
        </button>
        {lossUrl ? (
          <button
            id={lossTabId}
            type="button"
            role="tab"
            aria-controls={lossPanelId}
            aria-selected={activeTab === "loss"}
            className={activeTab === "loss" ? styles.experimentTabActive : undefined}
            onClick={() => setActiveTab("loss")}
          >
            Loss
          </button>
        ) : null}
      </div>

      <section className={styles.sampleSection}>
        <div className={styles.experimentPanels}>
          <div
            id={samplePanelId}
            role="tabpanel"
            aria-labelledby={sampleTabId}
            hidden={activeTab !== "samples"}
          >
            <HallucinationFilmstrip manifestUrl={manifestUrl} compact />
          </div>
          <div
            id={resultsPanelId}
            role="tabpanel"
            aria-labelledby={resultsTabId}
            hidden={activeTab !== "results"}
          >
            <table className={styles.resultsTable}>
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Value</th>
                  <th>Notes</th>
                </tr>
              </thead>
              <tbody>
                {metrics.map((metric) => (
                  <tr key={metric.label}>
                    <th scope="row">{metric.label}</th>
                    <td>{metric.value}</td>
                    <td>{metric.detail}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className={styles.resultsSummary}>{verdict}</p>
          </div>
          {lossUrl ? (
            <div
              id={lossPanelId}
              role="tabpanel"
              aria-labelledby={lossTabId}
              hidden={activeTab !== "loss"}
            >
              <LossViewer lossUrl={lossUrl} />
            </div>
          ) : null}
        </div>
      </section>
    </>
  );
}
