import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "sans-serif"],
      },
      colors: {
        brand: {
          sky: "#38bdf8",
          indigo: "#818cf8",
          dark: "#0d1117",
          surface: "#161b22",
          card: "#1e2d3d",
        },
      },
      backgroundImage: {
        "app-gradient":
          "linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%)",
        "header-gradient":
          "linear-gradient(135deg, #1a2a3a 0%, #0d2137 100%)",
        "button-gradient":
          "linear-gradient(135deg, #0ea5e9, #6366f1)",
      },
      keyframes: {
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        fadeIn: {
          from: { opacity: "0", transform: "translateY(8px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        shimmer: "shimmer 2s linear infinite",
        fadeIn: "fadeIn 0.3s ease forwards",
      },
    },
  },
  plugins: [],
};
export default config;
