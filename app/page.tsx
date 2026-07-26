import { BlocketLeagueLab } from "@/components/blocket-league/blocket-league-lab";
import { readBlocketLeagueCopy } from "@/lib/blocket-league/content";

export default async function HomePage() {
  const copy = await readBlocketLeagueCopy();
  return <BlocketLeagueLab copy={copy} editable={process.env.NODE_ENV === "development"} />;
}
