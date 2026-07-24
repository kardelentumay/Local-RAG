import type { Metadata } from "next";
import { Geist, Nunito } from "next/font/google";
import "./globals.css";

const geist = Geist({ variable: "--font-geist", subsets: ["latin"] });
const nunito = Nunito({ variable: "--font-nunito", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Nivora · Local Document Assistant",
  description: "A source-grounded local RAG assistant interface demo.",
  icons: { icon: "/assistant-icon.png" },
  openGraph: {
    title: "Nivora · Local Document Assistant",
    description: "Offline, source-grounded document chat.",
    images: [{ url: "/og.png", width: 1200, height: 630 }],
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${geist.variable} ${nunito.variable}`}>{children}</body>
    </html>
  );
}
