import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Blocket League — World Model Lab",
  description:
    "A pixel transformer learns a tiny physical world, reveals a writable velocity direction, and becomes playable through activation edits.",
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
