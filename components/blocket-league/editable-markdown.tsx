"use client";

import { useEffect, useState, type KeyboardEvent, type MouseEvent } from "react";
import ReactMarkdown from "react-markdown";
import remarkBreaks from "remark-breaks";

import type { BlocketLeagueCopyId } from "@/lib/blocket-league/content-types";

import styles from "./editable-markdown.module.css";

type EditableMarkdownProps = {
  blockId: BlocketLeagueCopyId;
  markdown: string;
  editable: boolean;
  className?: string;
};

export function EditableMarkdown({
  blockId,
  markdown,
  editable,
  className,
}: EditableMarkdownProps) {
  const [currentMarkdown, setCurrentMarkdown] = useState(markdown);
  const [draft, setDraft] = useState(markdown);
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!editing) {
      setCurrentMarkdown(markdown);
      setDraft(markdown);
    }
  }, [editing, markdown]);

  const beginEditing = (event?: MouseEvent<HTMLDivElement>) => {
    if (!editable) return;
    if (event?.target instanceof Element && event.target.closest("a")) return;
    setDraft(currentMarkdown);
    setError(null);
    setEditing(true);
  };

  const cancelEditing = () => {
    setDraft(currentMarkdown);
    setError(null);
    setEditing(false);
  };

  const save = async () => {
    if (draft.trim().length === 0) {
      setError("This block cannot be empty.");
      return;
    }

    setSaving(true);
    setError(null);
    try {
      const response = await fetch("/api/local-copy", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ blockId, markdown: draft }),
      });
      const result = await response.json() as { error?: string };

      if (!response.ok) throw new Error(result.error ?? "Could not save this block.");

      setCurrentMarkdown(draft.trim());
      setEditing(false);
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Could not save this block.");
    } finally {
      setSaving(false);
    }
  };

  const handleEditorKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Escape") {
      event.preventDefault();
      cancelEditing();
    }
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      void save();
    }
  };

  if (editing) {
    return (
      <div className={`${styles.editor} ${className ?? ""}`} data-markdown-editor>
        <div className={styles.editorLabel}>
          <strong>Edit Markdown</strong>
          <span>⌘ Enter to save · Esc to cancel</span>
        </div>
        <textarea
          autoFocus
          aria-label={`Markdown for ${blockId}`}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={handleEditorKeyDown}
        />
        {error && <p className={styles.error}>{error}</p>}
        <div className={styles.actions}>
          <button type="button" onClick={cancelEditing} disabled={saving}>Cancel</button>
          <button type="button" onClick={() => void save()} disabled={saving}>
            {saving ? "Saving…" : "Save to file"}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`${styles.block} ${styles.rendered} ${editable ? styles.editable : ""} ${className ?? ""}`}
      onClick={beginEditing}
      onKeyDown={(event) => {
        if (editable && (event.key === "Enter" || event.key === " ")) {
          event.preventDefault();
          setEditing(true);
        }
      }}
      role={editable ? "button" : undefined}
      tabIndex={editable ? 0 : undefined}
      aria-label={editable ? `Edit ${blockId} as Markdown` : undefined}
    >
      <ReactMarkdown
        remarkPlugins={[remarkBreaks]}
        components={{
          a: ({ children, ...props }) => <a {...props} target="_blank" rel="noreferrer">{children}</a>,
        }}
      >
        {currentMarkdown}
      </ReactMarkdown>
      {editable && <span className={styles.badge}>Edit Markdown</span>}
    </div>
  );
}
