import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Retinal Image Analysis Platform",
  description:
    "Classical Digital Image Processing · Blood Vessel Segmentation · Hard-Exudate Candidate Detection",
  icons: { icon: "/favicon.ico" },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-[#0d1117] text-white antialiased">
        {children}
      </body>
    </html>
  );
}
