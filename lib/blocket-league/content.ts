import { readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";

import {
  BLOCKET_LEAGUE_COPY_IDS,
  type BlocketLeagueCopy,
  type BlocketLeagueCopyId,
} from "./content-types";

const CONTENT_PATH = path.join(process.cwd(), "content", "blocket-league-copy.md");

function blockMarkers(id: BlocketLeagueCopyId) {
  return {
    start: `<!-- block:${id} -->`,
    end: "<!-- /block -->",
  };
}

function readBlock(source: string, id: BlocketLeagueCopyId) {
  const { start, end } = blockMarkers(id);
  const startIndex = source.indexOf(start);
  const endIndex = source.indexOf(end, startIndex + start.length);

  if (startIndex < 0 || endIndex < 0) {
    throw new Error(`Missing Markdown content block: ${id}`);
  }

  return source.slice(startIndex + start.length, endIndex).trim();
}

export async function readBlocketLeagueCopy(): Promise<BlocketLeagueCopy> {
  const source = await readFile(CONTENT_PATH, "utf8");

  return Object.fromEntries(
    BLOCKET_LEAGUE_COPY_IDS.map((id) => [id, readBlock(source, id)]),
  ) as BlocketLeagueCopy;
}

export async function updateBlocketLeagueCopy(
  id: BlocketLeagueCopyId,
  markdown: string,
) {
  const source = await readFile(CONTENT_PATH, "utf8");
  const { start, end } = blockMarkers(id);
  const startIndex = source.indexOf(start);
  const endIndex = source.indexOf(end, startIndex + start.length);

  if (startIndex < 0 || endIndex < 0) {
    throw new Error(`Missing Markdown content block: ${id}`);
  }

  const nextSource = [
    source.slice(0, startIndex + start.length),
    `\n${markdown.trim()}\n`,
    source.slice(endIndex),
  ].join("");
  const temporaryPath = `${CONTENT_PATH}.tmp`;

  await writeFile(temporaryPath, nextSource, "utf8");
  await rename(temporaryPath, CONTENT_PATH);
}
