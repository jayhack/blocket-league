import type { Metadata } from "next";
import "./globals.css";

const description =
  "A pixel transformer learns a tiny physical world, reveals a writable velocity direction, and becomes playable through activation edits.";

export const metadata: Metadata = {
  metadataBase: new URL("https://blocket-league.vercel.app"),
  title: "Blocket League | World Model Lab",
  description,
  openGraph: {
    title: "Blocket League — steer a video model's hallucinations",
    description,
    url: "/",
    siteName: "Blocket League",
    images: [{ url: "/og-image.png", width: 1200, height: 630 }],
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Blocket League — steer a video model's hallucinations",
    description,
    images: ["/og-image.png"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
