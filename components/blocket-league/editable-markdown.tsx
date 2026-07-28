"use client";

import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type KeyboardEvent,
  type MouseEvent,
  type ReactNode,
} from "react";
import ReactMarkdown from "react-markdown";
import remarkBreaks from "remark-breaks";

import type { BlocketLeagueCopyId } from "@/lib/blocket-league/content-types";

import styles from "./editable-markdown.module.css";

type EditableMarkdownProps = {
  blockId: BlocketLeagueCopyId;
  markdown: string;
  editable: boolean;
  className?: string;
  headingId?: string;
};

export function EditableMarkdown({
  blockId,
  markdown,
  editable,
  className,
  headingId,
}: EditableMarkdownProps) {
  const [currentMarkdown, setCurrentMarkdown] = useState(markdown);
  const [draft, setDraft] = useState(markdown);
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pendingSavedMarkdown = useRef<string | null>(null);
  const activeBlockId = useRef(blockId);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (activeBlockId.current !== blockId) {
      activeBlockId.current = blockId;
      pendingSavedMarkdown.current = null;
      setCurrentMarkdown(markdown);
      setDraft(markdown);
      setEditing(false);
      setSaving(false);
      setError(null);
      return;
    }

    if (editing) return;

    if (pendingSavedMarkdown.current !== null) {
      if (markdown === pendingSavedMarkdown.current) {
        pendingSavedMarkdown.current = null;
      } else {
        return;
      }
    }

    setCurrentMarkdown(markdown);
    setDraft(markdown);
  }, [blockId, editing, markdown]);

  useLayoutEffect(() => {
    if (!editing || !textareaRef.current) return;

    const textarea = textareaRef.current;
    const borderHeight = textarea.offsetHeight - textarea.clientHeight;
    textarea.style.height = "0px";
    textarea.style.height = `${textarea.scrollHeight + borderHeight}px`;
  }, [draft, editing]);

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
      const normalizedMarkdown = draft.trim();
      const response = await fetch("/api/local-copy/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ blockId, markdown: normalizedMarkdown }),
      });
      const result = await response.json() as { error?: string };

      if (!response.ok) throw new Error(result.error ?? "Could not save this block.");

      pendingSavedMarkdown.current = normalizedMarkdown;
      setCurrentMarkdown(normalizedMarkdown);
      setDraft(normalizedMarkdown);
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
  };

  if (editing) {
    return (
      <div
        className={`${styles.editor} ${headingId ? styles.titleEditor : ""} ${className ?? ""}`}
        data-markdown-editor
      >
        <div className={styles.editorLabel}>
          <strong>Edit Markdown</strong>
          <span>Esc to cancel</span>
        </div>
        <textarea
          ref={textareaRef}
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
      className={`${styles.block} ${styles.rendered} ${headingId ? styles.title : ""} ${editable ? styles.editable : ""} ${className ?? ""}`}
      onPointerDown={(event) => {
        if (!editable || !headingId) return;
        if (event.target instanceof Element && event.target.closest("a")) return;
        event.preventDefault();
        beginEditing();
      }}
      onClick={headingId ? undefined : beginEditing}
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
          a: ({ children, href, ...props }) => (
            <a
              {...props}
              href={href}
              {...(href?.startsWith("#") ? {} : { target: "_blank", rel: "noreferrer" })}
            >
              {children}
            </a>
          ),
          ...(headingId
            ? {
                p: ({ children }: { children?: ReactNode }) => (
                  <h2 id={headingId}>{children}</h2>
                ),
                h1: ({ children }: { children?: ReactNode }) => (
                  <h2 id={headingId}>{children}</h2>
                ),
                h2: ({ children }: { children?: ReactNode }) => (
                  <h2 id={headingId}>{children}</h2>
                ),
                h3: ({ children }: { children?: ReactNode }) => (
                  <h2 id={headingId}>{children}</h2>
                ),
              }
            : {}),
        }}
      >
        {currentMarkdown}
      </ReactMarkdown>
      {editable && <span className={styles.badge}>Edit Markdown</span>}
    </div>
  );
}
