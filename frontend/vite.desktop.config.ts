import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  plugins: [react()],
  define: {
    "process.env.NEXT_PUBLIC_LOCAL_RAG_API_URL": JSON.stringify("http://127.0.0.1:8000"),
  },
  build: {
    outDir: "dist-desktop",
    emptyOutDir: true,
    rollupOptions: {
      input: "desktop.html",
    },
  },
});
