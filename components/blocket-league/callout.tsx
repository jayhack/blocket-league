import type { ReactNode } from "react";

import styles from "./callout.module.css";

type CalloutVariant = "blue" | "orange";

interface CalloutProps {
  title: ReactNode;
  children: ReactNode;
  variant?: CalloutVariant;
  className?: string;
}

export function Callout({
  title,
  children,
  variant = "blue",
  className,
}: CalloutProps) {
  return (
    <aside
      className={[styles.callout, styles[variant], className]
        .filter(Boolean)
        .join(" ")}
    >
      <strong className={styles.title}>{title}</strong>
      <div className={styles.body}>{children}</div>
    </aside>
  );
}

export default Callout;
