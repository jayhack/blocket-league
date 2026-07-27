import { revalidatePath } from "next/cache";

import { updateBlocketLeagueCopy } from "@/lib/blocket-league/content";
import { isBlocketLeagueCopyId } from "@/lib/blocket-league/content-types";

export async function POST(request: Request) {
  if (process.env.NODE_ENV === "production") {
    return Response.json({ error: "Local authoring is disabled." }, { status: 404 });
  }

  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return Response.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  if (
    typeof payload !== "object"
    || payload === null
    || !("blockId" in payload)
    || !("markdown" in payload)
    || typeof payload.blockId !== "string"
    || typeof payload.markdown !== "string"
    || !isBlocketLeagueCopyId(payload.blockId)
  ) {
    return Response.json({ error: "Invalid content block." }, { status: 400 });
  }

  if (payload.markdown.trim().length === 0 || payload.markdown.length > 12_000) {
    return Response.json({ error: "Markdown must be between 1 and 12,000 characters." }, { status: 400 });
  }

  await updateBlocketLeagueCopy(payload.blockId, payload.markdown);
  revalidatePath("/");
  revalidatePath("/blocket-league");
  return Response.json({ ok: true });
}
