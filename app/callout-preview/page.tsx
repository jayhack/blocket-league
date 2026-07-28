import { Callout } from "@/components/blocket-league/callout";

export default function CalloutPreviewPage() {
  return (
    <main
      style={{
        maxWidth: 760,
        margin: "0 auto",
        padding: "64px 24px",
        display: "flex",
        flexDirection: "column",
        gap: 32,
      }}
    >
      <h1 style={{ fontSize: 20, color: "#f4f4f5" }}>Callout preview</h1>

      <Callout
        variant="blue"
        title="Collision physics transfers into the missing quadrant."
      >
        <p>
          Against a matched control trained everywhere, the penalty in the
          unseen quadrant is only 0.8%.
        </p>
      </Callout>

      <Callout variant="orange" title="On-manifold only.">
        <p>
          We make no extrapolation claim. The model interpolates within its
          training regime and breaks outside it — a manipulable interpolative
          model, not a symbolic physics engine.
        </p>
      </Callout>

      <Callout variant="blue" title="Multi-paragraph example.">
        <p>First paragraph establishes the claim.</p>
        <p>Second paragraph gets automatic spacing above it.</p>
      </Callout>
    </main>
  );
}
